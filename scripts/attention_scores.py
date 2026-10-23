"""Previous-token and induction scores of every model, on the real evaluation set.

    python scripts/attention_scores.py [results/v2]

Same definitions as src/interpret.py, but on the 10,000 held-out real
sequences every other analysis uses (REAL_EVAL_SEED) instead of fresh draws:

  previous-token  attention from each context label to the symbol right before it
  induction       attention from the query to the label(s) that followed earlier
                  occurrences of the query symbol

For each gen*.pt of each chain: the best layer-0 previous-token score and the
best layer-1 induction score. Writes attention_scores.json in the results
folder and prints a table.
"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.collapse import REAL_EVAL_SEED, REAL_EVAL_SIZE, real_dataset
from src.patching import run
from src.train import load_checkpoint

CHAINS = ["base", "ext_t1", "ext_t1_s100", "ext_real10", "ext_t07",
          "ext_long", "ext_t07_s100", "ext_real10_s100", "ext_genctx", "ext_accum",
          "ext_t1_s200", "ext_t1_s300", "ext_t1_s400", "ext_t07_s200", "ext_t07_s300", "ext_t07_s400",
          "ext_real10_s200", "ext_real10_s300", "ext_real10_s400"]   # missing ones are skipped


@torch.no_grad()
def scores(model, cfg, data, batch=1000):
    K = cfg.n_pairs
    q = 2 * K
    lab = torch.arange(1, q, 2)
    prev = torch.zeros(cfg.n_layers, cfg.n_heads)
    ind = torch.zeros(cfg.n_layers, cfg.n_heads)
    for i in range(0, data.shape[0], batch):
        seq = data[i:i + batch, : q + 1]
        _, cache = run(model, seq)
        hit = (seq[:, 0:q:2] == seq[:, q:q + 1]).float()             # [B, K] earlier occurrences of the query
        for l, att in enumerate(cache["att"]):
            prev[l] += att[:, :, lab, lab - 1].mean(2).sum(0)
            ind[l] += (att[:, :, q, 1:q:2] * hit[:, None]).sum(2).sum(0)
    n = data.shape[0]
    return prev / n, ind / n


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "results/v2"
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 2)))
    out, data = {}, None
    for run_name in CHAINS:
        paths = sorted(glob.glob(os.path.join(root, run_name, "gen*.pt")),
                       key=lambda p: int(re.search(r"gen(\d+)\.pt$", p).group(1)))
        if not paths:
            continue
        rows = []
        for path in paths:
            model, cfg, _ = load_checkpoint(path, device="cpu")
            if data is None:
                data = real_dataset(cfg, REAL_EVAL_SEED, REAL_EVAL_SIZE)
            prev, ind = scores(model, cfg, data)
            rows.append({"gen": int(re.search(r"gen(\d+)\.pt$", path).group(1)),
                         "prev_token": float(prev[0].max()), "induction": float(ind[1].max())})
        out[run_name] = rows
        print(f"{run_name:<16} prev-token " + " ".join(f"{r['prev_token']:.2f}" for r in rows)
              + " | induction " + " ".join(f"{r['induction']:.2f}" for r in rows))
    with open(os.path.join(root, "attention_scores.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
