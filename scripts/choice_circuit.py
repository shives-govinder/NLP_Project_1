"""Causal tests of the next-symbol choice (the behaviour that collapses).

    python scripts/choice_circuit.py [results/v2] [--n 3000]

mechanism.py scores layer ablations and path patching on the query label, which
never degrades. This runs the same interventions on every extended model but
scores the free next-symbol choice on held-out real data, split as in Eq. 1
into leakage and misallocation (kl_in):

  intact          no intervention
  resample_layer  every head of layer l resampled (another sequence's output)
  mean_layer      every head of layer l replaced by its batch-mean output
  induction       the layer-1 head that matters most for the query label
                  (largest resampled drop in mechanism.json), resampled
  other_l1        mean over the other layer-1 heads, each resampled alone
  prev_path       the query-critical layer-0 head resampled only where layer 1
                  reads q / k / v, including the route via the layer-0 MLP
  attention       the choice implied by layer-1 attention alone: each head's
                  attention over the 8 context-symbol slots, pooled by symbol.
                  Uniform attention plus copying gives exactly q = c/8, so
                  att_kl_in = KL(q || attention-implied) measures how far the
                  attention itself misallocates, and att_corr its correlation
                  with the model's renormalised choice

Writes choice_circuit.json in the results folder and prints a table.
"""
import argparse
import glob
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.collapse import REAL_EVAL_SEED, real_dataset
from src.patching import corrupt, run
from src.train import load_checkpoint, set_threads

CHAINS = ["ext_t1", "ext_t1_s100", "ext_real10", "ext_t07",
          "ext_long", "ext_t07_s100", "ext_real10_s100", "ext_genctx", "ext_accum"]   # missing ones are skipped


def gen_number(path):
    return int(re.search(r"gen(\d+)\.pt$", path).group(1))


def split(logits, seq, cfg):
    """(leakage, misallocation) of the next-symbol choice, as in choice_profile."""
    K, S = cfg.n_pairs, cfg.n_symbols
    q = 2 * K
    p = logits[:, q + 1, :S].float().softmax(-1)
    ctx = seq[:, 0:q:2]
    p_true = torch.zeros(seq.shape[0], S).scatter_add_(1, ctx, torch.ones(ctx.shape)) / K
    kl = float((p_true * (p_true.clamp_min(1e-12).log() - p.clamp_min(1e-30).log())).sum(1).mean())
    mass_out = float((p * (p_true == 0)).sum(1).mean())
    leak = -math.log(max(1 - mass_out, 1e-12))
    return {"leak": leak, "kl_in": kl - leak, "p_out": mass_out}


def attention_choice(att, seq, logits, cfg):
    """Choice implied by layer-1 attention over the context-symbol slots."""
    K, S = cfg.n_pairs, cfg.n_symbols
    q = 2 * K
    ctx = seq[:, 0:q:2]                                              # [B, K]
    a = att[:, :, q + 1, 0:q:2].mean(1)                              # heads pooled, [B, K]
    a = a / a.sum(1, keepdim=True)
    p_att = torch.zeros(seq.shape[0], S).scatter_add_(1, ctx, a)
    counts = torch.zeros(seq.shape[0], S).scatter_add_(1, ctx, torch.ones(ctx.shape))
    p_true = counts / K
    inside = counts > 0
    p = logits[:, q + 1, :S].float().softmax(-1)
    p_in = p * inside / (p * inside).sum(1, keepdim=True)
    kl_in = float((p_true * (p_true.clamp_min(1e-12).log() - p_att.clamp_min(1e-12).log())).sum(1).mean())
    x, y = p_att[inside], p_in[inside]
    corr = float(torch.corrcoef(torch.stack([x, y]))[0, 1])
    t = p_true[inside]
    slope = float(((t - t.mean()) * (x - x.mean())).sum() / ((t - t.mean()) ** 2).sum())
    return {"att_kl_in": kl_in, "att_corr": corr, "att_slope": slope}


@torch.no_grad()
def analyse(model, cfg, seq, mech):
    L, H = cfg.n_layers, cfg.n_heads
    logits, clean = run(model, seq)
    _, bad = run(model, corrupt(seq))
    out = {"intact": split(logits, seq, cfg)}
    out.update(attention_choice(clean["att"][1], seq, logits, cfg))

    res = torch.tensor(mech["query_circuit"]["acc"]["resample"])    # [L, H] query accuracy
    ind = int(res[1].argmin())
    prev = int(res[0].argmin())
    out["induction_head"], out["prev_head"] = ind, prev
    out["induction_query_acc"], out["prev_query_acc"] = float(res[1, ind]), float(res[0, prev])

    out["resample_layer"], out["mean_layer"] = [], []
    for l in range(L):
        z = bad["z"][l]
        out["resample_layer"].append(split(run(model, seq, z_patch={(l, h): z[:, h] for h in range(H)})[0], seq, cfg))
        zm = clean["z"][l].mean(0, keepdim=True).expand_as(clean["z"][l])
        out["mean_layer"].append(split(run(model, seq, z_patch={(l, h): zm[:, h] for h in range(H)})[0], seq, cfg))

    heads = [split(run(model, seq, z_patch={(1, h): bad["z"][1][:, h]})[0], seq, cfg) for h in range(H)]
    out["induction"] = heads[ind]
    others = [heads[h] for h in range(H) if h != ind]
    out["other_l1"] = {k: sum(o[k] for o in others) / len(others) for k in others[0]}

    _, patched = run(model, seq, z_patch={(0, prev): bad["z"][0][:, prev]})
    total = patched["resid_in"][1] - clean["resid_in"][1]
    out["prev_path"] = {p: split(run(model, seq, path_delta={(1, p): total})[0], seq, cfg) for p in "qkv"}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("root", nargs="?", default="results/v2")
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--threads", type=int, default=None)
    args = p.parse_args()
    set_threads(args.threads)

    out = {}
    print(f"{'model':<16} {'intact':>13} {'L0 resamp':>13} {'L1 resamp':>13} {'L1 mean':>13} "
          f"{'induction':>13} {'other L1':>13} {'prev->k':>13} {'att mis':>7} {'att r':>6}")
    for run_name in CHAINS:
        run_dir = os.path.join(args.root, run_name)
        if not os.path.exists(os.path.join(run_dir, "mechanism.json")):
            continue
        with open(os.path.join(run_dir, "mechanism.json")) as f:
            mech = {r["gen"]: r for r in json.load(f)}
        rows = []
        for path in sorted(glob.glob(os.path.join(run_dir, "gen*.pt")), key=gen_number):
            g = gen_number(path)
            model, cfg, _ = load_checkpoint(path, device="cpu")
            seq = real_dataset(cfg, REAL_EVAL_SEED, args.n)
            rec = analyse(model, cfg, seq, mech[g])
            rec["gen"] = g
            rows.append(rec)
            f2 = lambda d: f"{d['leak']:.2f}/{d['kl_in']:.2f}"
            print(f"{run_name + ' g' + str(g):<16} {f2(rec['intact']):>13} {f2(rec['resample_layer'][0]):>13} "
                  f"{f2(rec['resample_layer'][1]):>13} {f2(rec['mean_layer'][1]):>13} {f2(rec['induction']):>13} "
                  f"{f2(rec['other_l1']):>13} {f2(rec['prev_path']['k']):>13} "
                  f"{rec['att_kl_in']:>7.3f} {rec['att_corr']:>6.2f}")
        out[run_name] = rows

    path = os.path.join(args.root, "choice_circuit.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {path}  (cells are leakage/misallocation in nats)")


if __name__ == "__main__":
    main()
