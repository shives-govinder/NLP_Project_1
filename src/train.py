"""Training and evaluation for a single generation of the ICL model.

``train_model`` is imported by the model-collapse pipeline; the ``__main__``
block lets you train, evaluate and save one model from the command line:

    python -m src.train --steps 20000 --n_unique 4 --dense_loss --save results/base.pt

Weights & Biases logging is optional and off by default (see --wandb).
"""
from __future__ import annotations

import argparse
import dataclasses
import os
from typing import List, Optional, Tuple

import torch

from .config import Config
from .data import dense_targets, extended_targets, make_fixed_set, make_icl_batch
from .metrics import dense_loss, query_accuracy, query_loss, perplexity_from_loss

# Fixed seeds give disjoint, reproducible validation / test splits.
VAL_SEED = 10_001
TEST_SEED = 20_002


def fixed_split(cfg: Config, seed: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Frozen val/test batches drawn from the *true* (uniform-label) distribution."""
    return make_fixed_set(
        cfg.eval_batches, cfg.batch_size, cfg.n_pairs, cfg.n_symbols, cfg.n_labels,
        seed=seed, device=cfg.device, n_unique=cfg.n_unique,
    )


@torch.no_grad()
def evaluate(model, cfg: Config, batches: List[Tuple[torch.Tensor, torch.Tensor]]) -> dict:
    """Query-position loss / accuracy / perplexity over a frozen set of batches.

    Metrics are always measured at the query, even when training with the
    dense loss, so numbers stay comparable across settings.
    """
    model.eval()
    tot_loss, tot_acc, n = 0.0, 0.0, 0
    for seq, tgt in batches:
        logits = model(seq)
        tot_loss += float(query_loss(logits, tgt).item())
        tot_acc += float(query_accuracy(logits, tgt).item())
        n += 1
    n = max(n, 1)
    mean_loss = tot_loss / n
    return {
        "loss": mean_loss,
        "acc": tot_acc / n,
        "ppl": perplexity_from_loss(torch.tensor(mean_loss)),
    }


def train_model(
    cfg: Config,
    label_probs: Optional[torch.Tensor] = None,
    log=print,
    wandb_run=None,
    dataset: Optional[torch.Tensor] = None,
) -> Tuple[torch.nn.Module, List[dict]]:
    """Train one model. Returns (model, history).

    ``dataset`` (extended variant): a fixed LongTensor [N, 2*n_pairs + 4] of
    extended sequences to sample training batches from, instead of generating
    fresh data every step. Generation 0 gets real sequences; later generations
    get sequences written by the previous generation's model.

    ``label_probs`` (a [n_labels] vector) skews the label distribution of the
    *training* data; the model-collapse pipeline passes the previous
    generation's distribution here. Validation/test always use the true
    (uniform) distribution so we measure degradation against reality.
    """
    from .model import Transformer

    torch.manual_seed(cfg.seed)
    device = cfg.device
    model = Transformer(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    val_set = fixed_split(cfg, VAL_SEED)

    if dataset is not None:
        dataset = dataset.to(device)

    history: List[dict] = []
    for step in range(cfg.steps + 1):
        model.train()
        if dataset is not None:
            idx = torch.randint(0, dataset.shape[0], (cfg.batch_size,), device=device)
            seq = dataset[idx]
            logits = model(seq)
            loss = dense_loss(logits, extended_targets(seq, cfg.n_pairs, context_targets=cfg.dense_loss))
        else:
            seq, tgt = make_icl_batch(
                cfg.batch_size, cfg.n_pairs, cfg.n_symbols, cfg.n_labels,
                device=device, label_probs=label_probs, n_unique=cfg.n_unique,
            )
            logits = model(seq)
            if cfg.dense_loss:
                loss = dense_loss(logits, dense_targets(seq, tgt))
            else:
                loss = query_loss(logits, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % cfg.eval_every == 0:
            val = evaluate(model, cfg, val_set)
            rec = {"step": step, "train_loss": float(loss.item()), **{f"val_{k}": v for k, v in val.items()}}
            history.append(rec)
            log(f"step {step:5d} | train loss {loss.item():.4f} | "
                f"val acc {val['acc']:.3f} | val loss {val['loss']:.4f} | val ppl {val['ppl']:.2f}")
            if wandb_run is not None:
                wandb_run.log(rec)

    return model, history


def save_checkpoint(model, cfg: Config, path: str, history: Optional[List[dict]] = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {"config": dataclasses.asdict(cfg), "state_dict": model.state_dict(), "history": history or []},
        path,
    )


def load_checkpoint(path: str, device: str = "cpu"):
    """Returns (model, cfg, history) from a file written by save_checkpoint."""
    from .model import Transformer

    ckpt = torch.load(path, map_location="cpu")
    cfg = Config(**ckpt["config"])
    cfg.device = device
    model = Transformer(cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg, ckpt.get("history", [])


def auto_device(requested: Optional[str]) -> str:
    if requested and requested != "cpu":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train one ICL induction-head model.")
    p.add_argument("--config", type=str, default=None, help="path to a YAML config")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--n_layers", type=int, default=None)
    p.add_argument("--n_pairs", type=int, default=None)
    p.add_argument("--n_unique", type=int, default=None, help="distinct symbols per sequence")
    p.add_argument("--n_labels", type=int, default=None)
    p.add_argument("--dense_loss", action="store_true", help="supervise every induction opportunity")
    p.add_argument("--attention_only", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None, help="cpu, cuda or mps")
    p.add_argument("--save", type=str, default=None, help="path to save the trained model, e.g. results/base.pt")
    p.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    return p


def main():
    args = build_argparser().parse_args()
    cfg = Config.from_yaml(args.config) if args.config else Config()
    for field in ("steps", "batch_size", "lr", "n_layers", "n_pairs", "n_unique", "n_labels", "seed"):
        val = getattr(args, field)
        if val is not None:
            setattr(cfg, field, val)
    if args.dense_loss:
        cfg.dense_loss = True
    if args.attention_only:
        cfg.attention_only = True
    cfg.device = auto_device(args.device or cfg.device)

    print(cfg.describe(), f"| n_unique={cfg.n_unique} | dense_loss={cfg.dense_loss} | device={cfg.device}")

    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(project="nlp-model-collapse", config=dataclasses.asdict(cfg))

    model, history = train_model(cfg, wandb_run=wandb_run)

    test = evaluate(model, cfg, fixed_split(cfg, TEST_SEED))
    print(f"FINAL TEST | acc {test['acc']:.3f} | loss {test['loss']:.4f} | ppl {test['ppl']:.2f}")
    print(f"model params: {model.num_params():,}")

    if args.save:
        save_checkpoint(model, cfg, args.save, history)
        print(f"saved model to {args.save}")


if __name__ == "__main__":
    main()
