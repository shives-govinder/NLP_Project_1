"""Null model for temperature sampling: what a perfect learner would do.

    python scripts/sampling_null.py [results/v2/ext_t07]

If every generation fitted its training data exactly, sampling at temperature T
would compound as p_g = normalise(p_{g-1}^(1/T)), so p_g is proportional to
p_0^((1/T)^g). Two versions, both scored on the same held-out real sequences
with choice_profile (Eq. 1):

  pure      p_0 = q, the true choice (no leakage at any generation)
  anchored  p_0 = the chain's generation-0 model, errors included

At T = 1 both nulls are flat, so any drift there is learner error. Compares
each with the trained chain (choice_profiles.json). Writes sampling_null.json
into the run folder and prints a table.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.collapse import REAL_EVAL_SEED, REAL_EVAL_SIZE, real_dataset
from src.patching import choice_profile
from src.train import load_checkpoint


class FixedChoice:
    """Stands in for a model: puts log p at the next-symbol position."""

    def __init__(self, logp, pos, vocab):
        self.logp, self.pos, self.vocab = logp, pos, vocab
        self.i = 0

    def eval(self):
        self.i = 0

    def __call__(self, seq):
        B = seq.shape[0]
        out = torch.zeros(B, seq.shape[1], self.vocab)
        out[:, self.pos, : self.logp.shape[1]] = self.logp[self.i:self.i + B]
        self.i += B
        return out


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "results/v2/ext_t07"
    model, cfg, _ = load_checkpoint(os.path.join(run, "gen0.pt"), device="cpu")
    with open(os.path.join(run, "collapse.json")) as f:
        meta = json.load(f)
    T = meta["config"]["collapse_temperature"]
    n_gen = len(meta["generations"])
    K, S = cfg.n_pairs, cfg.n_symbols
    q = 2 * K
    data = real_dataset(cfg, REAL_EVAL_SEED, REAL_EVAL_SIZE)

    ctx = data[:, 0:q:2]
    log_q = (torch.zeros(len(data), S).scatter_add_(1, ctx, torch.ones(ctx.shape)) / K).log()
    with torch.no_grad():
        log_p0 = torch.cat([model(data[i:i + 2000])[:, q + 1, :S].float().log_softmax(-1)
                            for i in range(0, len(data), 2000)])

    trained = {}
    for name in ("choice_profiles.json", "revision_profiles.json", "seed_profiles.json"):
        path = os.path.join(os.path.dirname(run), name)
        if os.path.exists(path):
            with open(path) as f:
                rows = json.load(f).get(os.path.basename(run))
            if rows:
                trained = {r["gen"]: r for r in rows}

    out = {"temperature": T, "pure": [], "anchored": []}
    print(f"T = {T}: exponent (1/T)^g; columns are p_out / misallocation / slope / 1-slot")
    for g in range(n_gen):
        e = (1 / T) ** g
        row = {}
        for name, base in (("pure", log_q), ("anchored", log_p0)):
            logp = (base * e).log_softmax(-1)          # -inf stays -inf for the pure null
            prof = choice_profile(FixedChoice(logp, q + 1, cfg.vocab_size), cfg, data)
            prof["gen"] = g
            out[name].append(prof)
            row[name] = prof
        fmt = lambda r: f"{r['mass_out']:.3f}/{r['kl_in']:.3f}/{r['slope']:.2f}/{r['by_count'].get(1, float('nan')):.3f}"
        t = trained.get(g)
        print(f"gen {g} exp {e:5.2f} | pure {fmt(row['pure'])} | anchored {fmt(row['anchored'])}"
              + (f" | trained {fmt({**t, 'by_count': {int(k): v for k, v in t['by_count'].items()}})}" if t else ""))

    path = os.path.join(run, "sampling_null.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
