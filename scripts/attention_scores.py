"""Previous-token and induction scores of every model (src/interpret.py).

    python scripts/attention_scores.py [results/v2]

For each gen*.pt of each chain: the best layer-0 previous-token score and the
best layer-1 induction score, averaged over 10 batches of real sequences.
Writes attention_scores.json in the results folder and prints a table.
"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.interpret import induction_score, previous_token_score
from src.train import load_checkpoint

CHAINS = ["base", "ext_t1", "ext_t1_s100", "ext_real10", "ext_t07",
          "ext_long", "ext_t07_s100", "ext_real10_s100", "ext_genctx", "ext_accum"]   # missing ones are skipped


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "results/v2"
    torch.set_num_threads(2)
    torch.manual_seed(0)
    out = {}
    for run in CHAINS:
        rows = []
        paths = sorted(glob.glob(os.path.join(root, run, "gen*.pt")),
                       key=lambda p: int(re.search(r"gen(\d+)\.pt$", p).group(1)))
        if not paths:
            continue
        for path in paths:
            model, cfg, _ = load_checkpoint(path, device="cpu")
            prev = previous_token_score(model, cfg)[0]
            ind = induction_score(model, cfg)[1]
            rows.append({"gen": int(re.search(r"gen(\d+)", path).group(1)),
                         "prev_token": float(prev.max()), "induction": float(ind.max())})
        out[run] = rows
        print(f"{run:<12} prev-token " + " ".join(f"{r['prev_token']:.2f}" for r in rows)
              + " | induction " + " ".join(f"{r['induction']:.2f}" for r in rows))
    with open(os.path.join(root, "attention_scores.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
