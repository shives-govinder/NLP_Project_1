"""Data interventions on one generation's training data (in the spirit of Chen et al., 2026).

Trains fresh extended-variant models on a chosen dataset and records (1) when
the induction circuit forms and (2) how far the trained model's own output
drifts from real data. Comparing datasets and deletions shows which training
samples cause collapse and how the data affects circuit formation.

    # real data vs generation-7 synthetic data, 3 seeds each
    python scripts/formation.py --data real --seeds 0 1 2 --out results/formation/real.json
    python scripts/formation.py --data results/ext/data_gen7.pt --seeds 0 1 2 --out results/formation/gen7.json

    # delete the erroneous synthetic rows, or the same number of random rows (control)
    python scripts/formation.py --data results/ext/data_gen7.pt --delete errors --out ...
    python scripts/formation.py --data results/ext/data_gen7.pt --delete random --out ...

    # which training target speeds up induction formation (real data)
    python scripts/formation.py --data real --targets query --out ...

--delete errors removes rows whose generated tokens contradict the context: a
next symbol that is not in the context, or a query / next label that does not
match the context. --delete random removes the same number of rows at random.
"""
import argparse
import dataclasses
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.collapse import (EXT_DATA_SEED, REAL_EVAL_SEED, REAL_EVAL_SIZE, extended_stats,
                          generate_extended_dataset, real_dataset, real_eval)
from src.config import Config
from src.interpret import induction_score, previous_token_score
from src.train import auto_device, set_threads, train_model


def phase_step(history, threshold=0.9):
    for rec in history:
        if rec["val_acc"] >= threshold:
            return rec["step"]
    return None


def error_rows(data: torch.Tensor, K: int) -> torch.Tensor:
    """Rows whose generated tokens contradict the context."""
    ctx_s, ctx_l = data[:, 0:2 * K:2], data[:, 1:2 * K:2]
    q, lq, sn, ln = data[:, 2 * K], data[:, 2 * K + 1], data[:, 2 * K + 2], data[:, 2 * K + 3]
    rows = torch.arange(data.shape[0])
    q_slot = (ctx_s == q[:, None]).float().argmax(1)
    sn_match = ctx_s == sn[:, None]
    sn_in = sn_match.any(1)
    sn_slot = sn_match.float().argmax(1)
    return (~sn_in) | (lq != ctx_l[rows, q_slot]) | (sn_in & (ln != ctx_l[rows, sn_slot]))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", default="real", help='"real" or a data_gen{g}.pt file saved by src.collapse')
    p.add_argument("--delete", choices=["none", "errors", "random"], default="none")
    p.add_argument("--targets", choices=["all", "query+symbol", "query"], default="all")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--gen_samples", type=int, default=20000, help="sequences generated to measure drift")
    p.add_argument("--device", default=None)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    set_threads(args.threads)
    base = Config(variant="extended", n_unique=4, dense_loss=True, lr=1e-3, steps=args.steps,
                  ext_targets=args.targets)
    base.device = auto_device(args.device)
    data = real_dataset(base, EXT_DATA_SEED) if args.data == "real" else torch.load(args.data).long()
    eval_set = real_dataset(base, REAL_EVAL_SEED, REAL_EVAL_SIZE)

    bad = error_rows(data, base.n_pairs)
    n_bad = int(bad.sum())
    if args.delete == "errors":
        data = data[~bad]
    elif args.delete == "random":
        keep = torch.randperm(data.shape[0], generator=torch.Generator().manual_seed(123))[n_bad:]
        data = data[keep]
    print(f"data {args.data} | {n_bad} erroneous rows ({100 * n_bad / bad.numel():.2f}%) | "
          f"delete {args.delete} -> {data.shape[0]} rows | targets {args.targets}")

    runs = []
    for seed in args.seeds:
        cfg = dataclasses.replace(base, seed=seed)
        t0 = time.time()
        model, hist = train_model(cfg, dataset=data, log=lambda *a: None)
        gen_cfg = dataclasses.replace(cfg, dataset_size=args.gen_samples)
        st = extended_stats(generate_extended_dataset(model, gen_cfg, EXT_DATA_SEED + 999), gen_cfg)
        rec = {
            "seed": seed,
            "phase_step": phase_step(hist),
            "final_val_acc": hist[-1]["val_acc"],
            "real_eval": real_eval(model, cfg, eval_set),
            "generated_out_of_context": 1 - st["next_symbol_in_context"],
            "generated_stats": st,
            "induction_score": induction_score(model, cfg).tolist(),
            "prev_token_score": previous_token_score(model, cfg).tolist(),
            "history": hist,
            "minutes": (time.time() - t0) / 60,
        }
        runs.append(rec)
        print(f"seed {seed} | phase@{rec['phase_step']} | val acc {rec['final_val_acc']:.3f} | "
              f"generated not-in-context {100 * rec['generated_out_of_context']:.2f}% | "
              f"real next-symbol KL {rec['real_eval']['next_symbol_kl']:.4f} | {rec['minutes']:.1f} min")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "n_rows": int(data.shape[0]), "n_erroneous": n_bad, "runs": runs}, f, indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
