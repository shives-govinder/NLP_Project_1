"""Model-collapse pipeline for the ICL symbol task.

We train an initial ("generation 0") model on real, uniform data, then
repeatedly: (1) sample the current model's outputs to estimate its output
distribution, (2) train the next generation on data drawn from *that*
distribution. Over generations, information in the tails of the distribution
is progressively lost (Shumailov et al., 2024) and the output entropy should
fall.

Two variants (both required by the brief):

* ``base``     - the model predicts only the query's label. Because the correct
                 label is fully determined by the (uniform) context, a model
                 that has learned in-context learning tends to keep the marginal
                 label distribution ~uniform. Collapse is therefore expected to
                 be weak here: treat it as a CONTROL.

* ``extended`` - the model additionally predicts the next symbol and its value,
                 so its *own* preferences over symbols/labels feed back into the
                 data and can skew the distribution. Collapse is expected to be
                 stronger. Protocol: contexts stay real; the previous
                 generation samples the three continuation tokens (temperature
                 ``collapse_temperature``); each generation trains only on the
                 previous generation's fixed dataset of ``dataset_size`` sequences.

The functions here are deliberately small and composable so you can swap in
alternative collapse protocols (e.g. feeding whole generated sequences,
mixing in a fraction of real data, adding temperature, etc.).
"""
from __future__ import annotations

import dataclasses
import json
import os
from typing import List, Optional

import torch

from .config import Config
from .data import make_extended_batch, make_icl_batch
from .metrics import distribution_entropy
from .train import auto_device, evaluate, fixed_split, save_checkpoint, train_model, TEST_SEED


@torch.no_grad()
def estimate_label_distribution(model, cfg: Config) -> torch.Tensor:
    """Estimate the model's marginal distribution over label tokens.

    Runs the model on many fresh (uniform) contexts and counts its predicted
    labels at the query position. Returns a [n_labels] probability vector,
    optionally sharpened/softened by ``cfg.collapse_temperature``.
    """
    model.eval()
    counts = torch.zeros(cfg.n_labels)
    for _ in range(cfg.collapse_batches):
        seq, _ = make_icl_batch(
            cfg.batch_size, cfg.n_pairs, cfg.n_symbols, cfg.n_labels,
            device=cfg.device, n_unique=cfg.n_unique,
        )
        logits = model(seq)
        preds = logits[:, -1, :].argmax(dim=-1) - cfg.n_symbols
        valid = preds[(preds >= 0) & (preds < cfg.n_labels)]
        counts += torch.bincount(valid.cpu(), minlength=cfg.n_labels).float()

    probs = counts.clamp(min=1e-9)
    if cfg.collapse_temperature != 1.0:
        probs = probs.pow(1.0 / cfg.collapse_temperature)
    probs = probs / probs.sum()
    return probs


EXT_DATA_SEED = 30_003   # seeds for building / sampling extended datasets


def real_extended_dataset(cfg: Config, seed: int) -> torch.Tensor:
    """Generation 0's training data: ``cfg.dataset_size`` real extended sequences (CPU)."""
    g = torch.Generator().manual_seed(seed)
    chunks, left = [], cfg.dataset_size
    while left > 0:
        n = min(4096, left)
        chunks.append(make_extended_batch(
            n, cfg.n_pairs, cfg.n_symbols, cfg.n_labels, generator=g, n_unique=cfg.n_unique,
        ))
        left -= n
    return torch.cat(chunks)


@torch.no_grad()
def generate_extended_dataset(model, cfg: Config, seed: int) -> torch.Tensor:
    """The next generation's training data, written by ``model``.

    Contexts and queries are fresh real samples; the model then generates the
    three continuation tokens itself (query label, next symbol, next label) by
    sampling at ``cfg.collapse_temperature`` (<= 0 means argmax). Each token is
    restricted to the right type (label / symbol / label) so every sequence
    stays well-formed; which label or symbol is chosen is up to the model.
    Returns a CPU LongTensor [cfg.dataset_size, 2*n_pairs + 4].
    """
    model.eval()
    g = torch.Generator().manual_seed(seed)
    is_label = torch.arange(cfg.vocab_size) >= cfg.n_symbols
    chunks, left = [], cfg.dataset_size
    while left > 0:
        n = min(4096, left)
        ctx, _ = make_icl_batch(
            n, cfg.n_pairs, cfg.n_symbols, cfg.n_labels,
            device="cpu", generator=g, n_unique=cfg.n_unique,
        )
        seq = ctx.to(cfg.device)
        for want_label in (True, False, True):
            logits = model(seq)[:, -1, :].float().cpu()
            logits = logits.masked_fill(~(is_label if want_label else ~is_label), float("-inf"))
            if cfg.collapse_temperature <= 0:
                tok = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = torch.softmax(logits / cfg.collapse_temperature, dim=-1)
                tok = torch.multinomial(probs, 1, generator=g)
            seq = torch.cat([seq, tok.to(cfg.device)], dim=1)
        chunks.append(seq.cpu())
        left -= n
    return torch.cat(chunks)


@torch.no_grad()
def extended_stats(data: torch.Tensor, cfg: Config) -> dict:
    """Describe a set of extended sequences (real or generated).

    query_label_correct     generated query label matches the context
    next_symbol_in_context  chosen next symbol appears in the context (real data: 1.0)
    next_symbol_is_query    chosen next symbol is the query symbol itself
    next_label_correct      label given to the next symbol matches the context
                            (only over next symbols that are in the context)
    symbol_id_entropy       entropy of which symbol ids are chosen (uniform = ln n_symbols)
    choice_rank_entropy     entropy of *which* context symbol is chosen, by order of
                            first appearance (plus one bucket for "not in context")
    label_entropy           entropy of all generated labels (uniform = ln n_labels)
    """
    data = data.cpu()
    K, S, NL = cfg.n_pairs, cfg.n_symbols, cfg.n_labels
    rows = torch.arange(data.shape[0])
    ctx_syms, ctx_labs = data[:, 0:2 * K:2], data[:, 1:2 * K:2]
    q, lq, sn, ln = data[:, 2 * K], data[:, 2 * K + 1], data[:, 2 * K + 2], data[:, 2 * K + 3]

    q_slot = (ctx_syms == q[:, None]).float().argmax(dim=1)
    sn_match = ctx_syms == sn[:, None]
    sn_in = sn_match.any(dim=1)
    sn_slot = sn_match.float().argmax(dim=1)
    true_lq = ctx_labs[rows, q_slot]
    true_ln = ctx_labs[rows, sn_slot]

    same = ctx_syms[:, :, None] == ctx_syms[:, None, :]
    earlier = torch.tril(torch.ones(K, K), diagonal=-1).bool()
    is_first = ~(same & earlier).any(dim=2)                         # first appearance of each symbol
    rank = (is_first & (torch.arange(K)[None, :] < sn_slot[:, None])).sum(dim=1)
    rank = torch.where(sn_in, rank, torch.full_like(rank, K))       # K = "not in context"
    labels = (torch.cat([lq, ln]) - S).clamp(0, NL - 1)
    n_in = int(sn_in.sum())

    return {
        "query_label_correct": float((lq == true_lq).float().mean()),
        "next_symbol_in_context": float(sn_in.float().mean()),
        "next_symbol_is_query": float((sn == q).float().mean()),
        "next_label_correct": float(((ln == true_ln) & sn_in).sum()) / max(n_in, 1),
        "symbol_id_entropy": distribution_entropy(torch.bincount(sn.clamp(0, S - 1), minlength=S)),
        "choice_rank_entropy": distribution_entropy(torch.bincount(rank, minlength=K + 1)),
        "label_entropy": distribution_entropy(torch.bincount(labels, minlength=NL)),
    }


def format_stats(st: dict) -> str:
    return (f"q-label ok {st['query_label_correct']:.3f} | next-sym in ctx {st['next_symbol_in_context']:.3f} "
            f"| =query {st['next_symbol_is_query']:.3f} | next-label ok {st['next_label_correct']:.3f} "
            f"| H(sym id) {st['symbol_id_entropy']:.3f} | H(choice) {st['choice_rank_entropy']:.3f} "
            f"| H(labels) {st['label_entropy']:.3f}")


def run_collapse(cfg: Config, log=print, save_dir: Optional[str] = None):
    """Run the full multi-generation collapse experiment.

    base      each generation trains on fresh data whose label distribution is
              the previous generation's predicted label distribution.
    extended  each generation trains on a fixed dataset of ``cfg.dataset_size``
              sequences: real ones for generation 0, then sequences whose three
              continuation tokens were generated by the previous generation.

    If ``save_dir`` is given, each generation's model is saved there as
    ``gen{g}.pt`` (for scripts/analyse_generations.py).

    Returns (history, real_reference): one record per generation, and for the
    extended variant the statistics of real data to compare against (else None).
    """
    history: List[dict] = []
    label_probs: Optional[torch.Tensor] = None  # base: gen 0 = real/uniform data
    test_set = fixed_split(cfg, TEST_SEED)

    dataset, real_ref = None, None
    if cfg.variant == "extended":
        dataset = real_extended_dataset(cfg, EXT_DATA_SEED + cfg.seed)
        real_ref = extended_stats(dataset, cfg)
        log(f"real-data reference | {format_stats(real_ref)}")
    elif cfg.variant != "base":
        raise ValueError(f"unknown variant {cfg.variant!r}")

    for gen in range(cfg.n_generations):
        # Each generation gets its own seed (fresh init + fresh data order).
        # Reusing one seed makes generations near-identical re-runs, which
        # confounds generation effects with seed effects.
        gen_cfg = dataclasses.replace(cfg, seed=cfg.seed + gen)
        log(f"\n=== generation {gen} (variant={cfg.variant}, seed={gen_cfg.seed}) ===")
        if cfg.variant == "extended":
            model, train_hist = train_model(gen_cfg, dataset=dataset, log=log)
        else:
            model, train_hist = train_model(gen_cfg, label_probs=label_probs, log=log)
        test = evaluate(model, gen_cfg, test_set)
        if save_dir:
            save_checkpoint(model, gen_cfg, os.path.join(save_dir, f"gen{gen}.pt"), train_hist)

        rec = {
            "gen": gen,
            "seed": gen_cfg.seed,
            "train_label_probs": None if label_probs is None else label_probs.tolist(),
            "train_history": train_hist,
            "test_acc": test["acc"],
            "test_loss": test["loss"],
            "test_ppl": test["ppl"],
        }
        if cfg.variant == "base":
            next_probs = estimate_label_distribution(model, gen_cfg)
            rec["next_label_entropy"] = distribution_entropy(next_probs)
            rec["label_probs"] = next_probs.tolist()
            label_probs = next_probs  # feed forward into the next generation
            log(f"gen {gen} | test acc {test['acc']:.3f} | next-dist entropy {rec['next_label_entropy']:.4f} "
                f"(uniform = {torch.log(torch.tensor(float(cfg.n_labels))):.4f})")
        else:
            # This generation writes the data the next generation trains on.
            dataset = generate_extended_dataset(model, gen_cfg, EXT_DATA_SEED + 1000 * (gen + 1) + cfg.seed)
            stats = extended_stats(dataset, gen_cfg)
            rec["extended_stats"] = stats
            rec["next_label_entropy"] = stats["label_entropy"]
            log(f"gen {gen} | test acc {test['acc']:.3f} | generated: {format_stats(stats)}")
        history.append(rec)

    return history, real_ref


def main():
    import argparse
    p = argparse.ArgumentParser(description="Run the model-collapse pipeline.")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--variant", choices=["base", "extended"], default=None)
    p.add_argument("--n_generations", type=int, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--n_unique", type=int, default=None)
    p.add_argument("--dense_loss", action="store_true")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=None, help="seed of generation 0 (gen g uses seed+g)")
    p.add_argument("--temperature", type=float, default=None,
                   help="sampling temperature for generated data (extended; <= 0 = argmax)")
    p.add_argument("--dataset_size", type=int, default=None, help="sequences per generation (extended)")
    p.add_argument("--device", type=str, default=None, help="cpu, cuda or mps")
    p.add_argument("--out", type=str, default=None, help="save full results as JSON, e.g. results/collapse_base.json")
    p.add_argument("--save_dir", type=str, default=None,
                   help="save every generation's model (gen0.pt, gen1.pt, ...) and collapse.json here")
    args = p.parse_args()

    cfg = Config.from_yaml(args.config) if args.config else Config()
    for field in ("variant", "n_generations", "steps", "n_unique", "lr", "seed", "dataset_size"):
        val = getattr(args, field)
        if val is not None:
            setattr(cfg, field, val)
    if args.temperature is not None:
        cfg.collapse_temperature = args.temperature
    if args.dense_loss:
        cfg.dense_loss = True
    cfg.device = auto_device(args.device or cfg.device)

    history, real_ref = run_collapse(cfg, save_dir=args.save_dir)
    if args.save_dir and not args.out:
        args.out = os.path.join(args.save_dir, "collapse.json")

    print("\ngen,seed,test_acc,next_label_entropy")
    for r in history:
        print(f"{r['gen']},{r['seed']},{r['test_acc']:.4f},{r['next_label_entropy']:.4f}")
    if real_ref is not None:
        print(f"\nreal data | {format_stats(real_ref)}")
        for r in history:
            print(f"gen {r['gen']}    | {format_stats(r['extended_stats'])}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"config": dataclasses.asdict(cfg), "generations": history,
                       "real_reference": real_ref}, f, indent=2)
        print(f"saved results to {args.out}")


if __name__ == "__main__":
    main()
