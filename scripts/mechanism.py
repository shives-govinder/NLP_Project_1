"""Causal circuit tests on every generation of a collapse run (see src/patching.py).

    python scripts/mechanism.py results/v2/ext_t1 [--threads 6]

For each gen*.pt it runs, on held-out real data:

  query circuit    resample vs zero ablation of every head, resampling a whole
                   layer, and path patching of each layer-0 head into layer 1's
                   queries / keys / values (scored at the query label)
  choice mechanism (extended models) where each head attends from the position
                   that picks the next symbol, and how resampling each head
                   changes that choice (KL to the true choice, probability on
                   out-of-context symbols, probability on the query symbol)

Writes mechanism.json in the run folder and prints a summary per generation.
"""
import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.collapse import REAL_EVAL_SEED, real_dataset
from src.patching import check_run_matches, choice_mechanism, query_circuit_tests
from src.train import TEST_SEED, auto_device, fixed_split, load_checkpoint, set_threads


def gen_number(path):
    return int(re.search(r"gen(\d+)\.pt$", path).group(1))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("run_dir")
    p.add_argument("--device", default=None)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--n_choice", type=int, default=3000, help="real sequences for the choice analysis")
    args = p.parse_args()
    set_threads(args.threads)
    device = auto_device(args.device)

    paths = sorted(glob.glob(os.path.join(args.run_dir, "gen*.pt")), key=gen_number)
    rows = []
    for path in paths:
        model, cfg, _ = load_checkpoint(path, device=device)
        batches = fixed_split(cfg, TEST_SEED)[:10]
        assert check_run_matches(model, batches[0][0]) < 1e-3, "patched forward does not match the model"
        rec = {"gen": gen_number(path), "query_circuit": query_circuit_tests(model, cfg, batches)}
        if cfg.variant == "extended":
            data = real_dataset(cfg, REAL_EVAL_SEED, args.n_choice)
            rec["choice"] = choice_mechanism(model, cfg, data)
        rows.append(rec)

        acc = rec["query_circuit"]["acc"]
        res = torch.tensor(acc["resample"])
        zero = torch.tensor(acc["zero"])
        path_ = torch.tensor(acc["path"])
        print(f"\n=== gen {rec['gen']} ({path}) | intact acc {acc['intact']:.3f}")
        print("  resample one head [layer x head]:", [[round(v, 3) for v in r] for r in res.tolist()])
        print("  zero one head     [layer x head]:", [[round(v, 3) for v in r] for r in zero.tolist()])
        print("  resample whole layer:", [round(v, 3) for v in acc["resample_layer"]])
        print("  path patch L0 head -> L1 [head x (q,k,v)]:", [[round(v, 3) for v in r] for r in path_.tolist()])
        if "choice" in rec:
            ch = rec["choice"]
            print(f"  choice intact: " + ", ".join(f"{k} {v:.4f}" for k, v in ch["intact"].items()))
            for l in range(cfg.n_layers):
                for h in range(cfg.n_heads):
                    a = ch["attn"][l][h]
                    r = ch["resample"][l][h]
                    print(f"  L{l}H{h} attn q/sym/lab/self {a[0]:.2f}/{a[1]:.2f}/{a[2]:.2f}/{a[3]:.2f} | "
                          f"resampled: KL {r['kl']:.3f} out {r['mass_out']:.3f} query p {r['query_prob']:.3f}")

    out = os.path.join(args.run_dir, "mechanism.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
