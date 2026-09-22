"""Training and evaluation for a single generation of the ICL model.

``train_model`` is imported by the model-collapse pipeline; the ``__main__``
block lets you train and evaluate one model from the command line:

    python -m src.train --steps 3000 --attention_only

Weights & Biases logging is optional and off by default (see --wandb).
"""
from __future__ import annotations

import argparse
from typing import List, Optional, Tuple

import torch

from .config import Config
from .data import make_fixed_set, make_icl_batch
from .metrics import query_accuracy, query_loss, perplexity_from_loss

# Fixed seeds give disjoint, reproducible validation / test splits.
VAL_SEED = 10_001
TEST_SEED = 20_002


@torch.no_grad()
def evaluate(model, cfg: Config, batches: List[Tuple[torch.Tensor, torch.Tensor]]) -> dict:
    """Average loss / accuracy / perplexity over a frozen set of batches."""
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
) -> Tuple[torch.nn.Module, List[dict]]:
    """Train one model. Returns (model, history).

    ``label_probs`` (a [n_labels] vector) skews the label distribution of the
    *training* data; the model-collapse pipeline passes the previous
    generation's distribution here. Validation/test always use the true
    (uniform) distribution so we measure degradation against reality.
    """
    # Local import so the module imports even without torch installed elsewhere.
    from .model import Transformer

    torch.manual_seed(cfg.seed)
    device = cfg.device
    model = Transformer(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    val_set = make_fixed_set(
        cfg.eval_batches, cfg.batch_size, cfg.n_pairs, cfg.n_symbols, cfg.n_labels,
        seed=VAL_SEED, device=device,
    )

    history: List[dict] = []
    for step in range(cfg.steps + 1):
        model.train()
        seq, tgt = make_icl_batch(
            cfg.batch_size, cfg.n_pairs, cfg.n_symbols, cfg.n_labels,
            device=device, label_probs=label_probs,
        )
        logits = model(seq)
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


def _auto_device(requested: str) -> str:
    if requested != "cpu":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train one ICL induction-head model.")
    p.add_argument("--config", type=str, default=None, help="path to a YAML config")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--n_layers", type=int, default=None)
    p.add_argument("--attention_only", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    return p


def main():
    args = build_argparser().parse_args()
    cfg = Config.from_yaml(args.config) if args.config else Config()
    for field in ("steps", "batch_size", "lr", "n_layers", "seed", "device"):
        val = getattr(args, field)
        if val is not None:
            setattr(cfg, field, val)
    if args.attention_only:
        cfg.attention_only = True
    cfg.device = _auto_device(cfg.device)

    print(cfg.describe(), f"| device={cfg.device}")

    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(project="nlp-model-collapse", config=vars(cfg))

    model, _ = train_model(cfg, wandb_run=wandb_run)

    test_set = make_fixed_set(
        cfg.eval_batches, cfg.batch_size, cfg.n_pairs, cfg.n_symbols, cfg.n_labels,
        seed=TEST_SEED, device=cfg.device,
    )
    test = evaluate(model, cfg, test_set)
    print(f"FINAL TEST | acc {test['acc']:.3f} | loss {test['loss']:.4f} | ppl {test['ppl']:.2f}")
    print(f"model params: {model.num_params():,}")


if __name__ == "__main__":
    main()
