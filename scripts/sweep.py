"""Hyper-parameter sweep on the extended variant's generation 0 (real data only).

    python scripts/sweep.py --device mps
    python scripts/sweep.py --device mps --lrs 0.0003 0.001 0.003 --d_models 32 64 128 --steps 8000

For every (learning rate, d_model) pair it trains one model on real extended
data and records, using the VALIDATION set only (test is never touched):

  phase_step     first logged step with validation accuracy >= 0.9 (None = never)
  val_acc/loss   at the end of training
  gen_out_ctx    share of generated next symbols NOT in the context, measured on
                 a fresh sample written by the trained model. This is how far its
                 own generations drift from real data, and so how fast a collapse
                 chain started from it would drift. Lower is better.

Results go to results/sweep/sweep.json and a table is printed.
"""
import argparse
import dataclasses
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.collapse import EXT_DATA_SEED, extended_stats, generate_extended_dataset, real_extended_dataset
from src.config import Config
from src.train import auto_device, train_model


def phase_step(history, threshold=0.9):
    for rec in history:
        if rec["val_acc"] >= threshold:
            return rec["step"]
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--lrs", type=float, nargs="+", default=[3e-4, 1e-3, 3e-3])
    p.add_argument("--d_models", type=int, nargs="+", default=[32, 64, 128])
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gen_samples", type=int, default=20000, help="sequences generated to measure drift")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out", type=str, default="results/sweep/sweep.json")
    args = p.parse_args()

    base = Config(variant="extended", n_unique=4, dense_loss=True, steps=args.steps, seed=args.seed)
    base.device = auto_device(args.device)
    data = real_extended_dataset(base, EXT_DATA_SEED + args.seed)   # same training data for every config

    results = []
    if os.path.exists(args.out):                                    # resume: skip finished configs
        results = json.load(open(args.out))
    done = {(r["lr"], r["d_model"]) for r in results}

    for lr in args.lrs:
        for d in args.d_models:
            if (lr, d) in done:
                continue
            cfg = dataclasses.replace(base, lr=lr, d_model=d, d_mlp=4 * d)
            print(f"\n=== lr {lr:g}  d_model {d} ===")
            t0 = time.time()
            model, hist = train_model(cfg, dataset=data, log=lambda *a: None)
            gen_cfg = dataclasses.replace(cfg, dataset_size=args.gen_samples)
            st = extended_stats(generate_extended_dataset(model, gen_cfg, EXT_DATA_SEED + 999), gen_cfg)
            rec = {"lr": lr, "d_model": d, "params": model.num_params(),
                   "phase_step": phase_step(hist), "val_acc": hist[-1]["val_acc"],
                   "val_loss": hist[-1]["val_loss"], "gen_out_ctx": 1 - st["next_symbol_in_context"],
                   "gen_query_share": st["next_symbol_is_query"], "minutes": (time.time() - t0) / 60,
                   "history": hist}
            results.append(rec)
            print(f"phase@{rec['phase_step']} | val acc {rec['val_acc']:.3f} | val loss {rec['val_loss']:.4f} "
                  f"| generated not-in-context {100 * rec['gen_out_ctx']:.2f}% | {rec['minutes']:.1f} min")
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            json.dump(results, open(args.out, "w"), indent=2)          # save after every config

    print(f"\n{'lr':>8} {'d_model':>7} {'params':>8} {'phase@':>7} {'val acc':>8} {'val loss':>9} {'not-in-ctx':>11}")
    for r in sorted(results, key=lambda r: (r["lr"], r["d_model"])):
        ph = "-" if r["phase_step"] is None else r["phase_step"]
        print(f"{r['lr']:>8g} {r['d_model']:>7} {r['params']:>8,} {ph:>7} {r['val_acc']:>8.3f} "
              f"{r['val_loss']:>9.4f} {100 * r['gen_out_ctx']:>10.2f}%")
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
