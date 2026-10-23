"""Fit the one-step recursion x_g = (1 - alpha) x_{g-1} + delta to leakage and misallocation.

    python scripts/leakage_fit.py [results/v2]

x is measured on held-out real data each generation (collapse.json real_eval):
leakage = -log(1 - p_out) in nats, misallocation = KL - leakage (Eq. 1).
alpha is the restoring force: 0 means nothing pulls x back, so it grows by delta
per generation; mixing a share r of real rows (p_out = 0) predicts alpha ~ r
(exactly for p_out, and to first order for leakage in nats, which is ~p_out when small).
When alpha > 0 the fixed point is delta / alpha.

Writes leakage_fit.json into the results folder and prints a table.
"""
import json
import os
import sys

import numpy as np

CHAINS = ["ext_t1", "ext_t1_s100", "ext_real10", "ext_t07",
          "ext_long", "ext_t07_s100", "ext_real10_s100", "ext_genctx", "ext_accum",
          "ext_t1_s200", "ext_t1_s300", "ext_t1_s400", "ext_t07_s200", "ext_t07_s300", "ext_t07_s400",
          "ext_real10_s200", "ext_real10_s300", "ext_real10_s400"]   # missing ones are skipped


def fit(x):
    """Least-squares (alpha, delta, R^2) of x[1:] on x[:-1]."""
    X = np.c_[x[:-1], np.ones(len(x) - 1)]
    (a, delta), *_ = np.linalg.lstsq(X, x[1:], rcond=None)
    pred = X @ [a, delta]
    r2 = 1 - ((x[1:] - pred) ** 2).sum() / ((x[1:] - x[1:].mean()) ** 2).sum()
    alpha = 1 - a
    return {"alpha": float(alpha), "delta": float(delta), "r2": float(r2),
            "fixed_point": float(delta / alpha) if alpha > 0.03 else None}


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "results/v2"
    out = {}
    print(f"{'chain':<12} {'measure':<14} {'alpha':>7} {'delta':>8} {'R2':>6} {'fixed pt':>8}")
    for run in CHAINS:
        if not os.path.exists(os.path.join(root, run, "collapse.json")):
            continue
        with open(os.path.join(root, run, "collapse.json")) as f:
            gens = json.load(f)["generations"]
        p_out = np.array([g["real_eval"]["next_symbol_mass_out"] for g in gens])
        kl = np.array([g["real_eval"]["next_symbol_kl"] for g in gens])
        leak = -np.log(1 - p_out)
        series = {"leakage": leak, "misallocation": kl - leak}
        out[run] = {}
        for name, x in series.items():
            r = fit(x)
            r["values"] = x.tolist()
            out[run][name] = r
            fp = "-" if r["fixed_point"] is None else f"{r['fixed_point']:.3f}"
            print(f"{run:<12} {name:<14} {r['alpha']:>7.3f} {r['delta']:>8.4f} {r['r2']:>6.3f} {fp:>8}")
    path = os.path.join(root, "leakage_fit.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
