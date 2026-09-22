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
                 stronger. The scaffold below marks the exact feedback protocol
                 as a TODO for your group to design and justify - that design
                 choice is a large part of the marks.

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
from .data import make_icl_batch
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


def sample_extended_distribution(model, cfg: Config) -> torch.Tensor:
    """TODO (Project 1, extended variant).

    Implement the extended feedback loop: let the model generate not just the
    query label but also the next symbol and its value, then fold the model's
    *symbol* preferences back into the next generation's data (e.g. by biasing
    which symbols appear, or how labels are assigned). Return whatever object
    your training loop consumes (here we return a label distribution to keep
    the interface identical, but you may want to also return a symbol
    distribution and extend ``make_icl_batch`` to accept it).

    Designing and justifying this protocol is a graded part of the project -
    keep the base variant above as your control and compare against it.
    """
    raise NotImplementedError(
        "Extended-variant feedback loop is intentionally left for the group to design. "
        "See the docstring and README roadmap."
    )


def run_collapse(cfg: Config, log=print, save_dir: Optional[str] = None) -> List[dict]:
    """Run the full multi-generation collapse experiment.

    If ``save_dir`` is given, each generation's model is saved there as
    ``gen{g}.pt`` so its circuit can be analysed afterwards
    (scripts/analyse_generations.py).

    Returns a history list with one record per generation containing the
    validation metrics and the entropy of the distribution the *next*
    generation will be trained on.
    """
    history: List[dict] = []
    label_probs: Optional[torch.Tensor] = None  # gen 0 = real/uniform data
    test_set = fixed_split(cfg, TEST_SEED)

    for gen in range(cfg.n_generations):
        # Each generation gets its own seed (fresh init + fresh data order).
        # Reusing one seed makes generations near-identical re-runs, which
        # confounds generation effects with seed effects.
        gen_cfg = dataclasses.replace(cfg, seed=cfg.seed + gen)
        log(f"\n=== generation {gen} (variant={cfg.variant}, seed={gen_cfg.seed}) ===")
        model, train_hist = train_model(gen_cfg, label_probs=label_probs, log=log)
        test = evaluate(model, gen_cfg, test_set)
        if save_dir:
            save_checkpoint(model, gen_cfg, os.path.join(save_dir, f"gen{gen}.pt"), train_hist)

        if cfg.variant == "base":
            next_probs = estimate_label_distribution(model, gen_cfg)
        elif cfg.variant == "extended":
            next_probs = sample_extended_distribution(model, gen_cfg)
        else:
            raise ValueError(f"unknown variant {cfg.variant!r}")

        ent = distribution_entropy(next_probs)
        rec = {
            "gen": gen,
            "seed": gen_cfg.seed,
            "train_label_probs": None if label_probs is None else label_probs.tolist(),
            "train_history": train_hist,
            "test_acc": test["acc"],
            "test_loss": test["loss"],
            "test_ppl": test["ppl"],
            "next_label_entropy": ent,
            "label_probs": next_probs.tolist(),
        }
        history.append(rec)
        log(f"gen {gen} | test acc {test['acc']:.3f} | next-dist entropy {ent:.4f} "
            f"(uniform = {torch.log(torch.tensor(float(cfg.n_labels))):.4f})")

        label_probs = next_probs  # feed forward into the next generation

    return history


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
    p.add_argument("--device", type=str, default=None, help="cpu, cuda or mps")
    p.add_argument("--out", type=str, default=None, help="save full results as JSON, e.g. results/collapse_base.json")
    p.add_argument("--save_dir", type=str, default=None,
                   help="save every generation's model (gen0.pt, gen1.pt, ...) and collapse.json here")
    args = p.parse_args()

    cfg = Config.from_yaml(args.config) if args.config else Config()
    for field in ("variant", "n_generations", "steps", "n_unique", "lr", "seed"):
        val = getattr(args, field)
        if val is not None:
            setattr(cfg, field, val)
    if args.dense_loss:
        cfg.dense_loss = True
    cfg.device = auto_device(args.device or cfg.device)

    history = run_collapse(cfg, save_dir=args.save_dir)
    if args.save_dir and not args.out:
        args.out = os.path.join(args.save_dir, "collapse.json")

    print("\ngen,seed,test_acc,next_label_entropy")
    for r in history:
        print(f"{r['gen']},{r['seed']},{r['test_acc']:.4f},{r['next_label_entropy']:.4f}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"config": dataclasses.asdict(cfg), "generations": history}, f, indent=2)
        print(f"saved results to {args.out}")


if __name__ == "__main__":
    main()
