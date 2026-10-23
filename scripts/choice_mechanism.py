"""How extended models make the next-symbol choice, and where it goes wrong.

    python scripts/choice_mechanism.py [results/v2] [--threads 6]

At the position that predicts the next symbol (the query label, 2K+1), on the
10,000 held-out real sequences (REAL_EVAL_SEED), for every gen*.pt of every
extended chain:

  attention     each layer-1 head's mean attention to the 8 context-symbol
                slots, the context labels, the query and itself
  dla           direct logit attribution: the final residual is split into
                embedding + layer 0, each layer-1 head, the layer-1 MLP and the
                final LayerNorm's bias; each is passed through the final
                LayerNorm with its scale frozen and the unembedding, so the
                parts add up exactly to the model's symbol logits. Per part:
                  count_slope  slope of its logit on log c over in-context
                               symbols (demeaned within each sequence); the
                               parts' slopes sum to the model's, and exact
                               frequency copying (p ~ c) needs a total of 1
                  in_out       mean logit on in-context minus out-of-context
                               symbols (positive = keeps the choice in context)
                  ov_corr      (heads) correlation over in-context symbols of
                               the head's attention to a symbol's slots with
                               its logit for that symbol (copying what it reads)
  misallocation exact additive split of KL(q || p~) (Eq. 1): each in-context
                symbol's term q(s) log(q(s)/p~(s)) is credited to
                  by_slot      its slots, equally (8 values)
                  by_last      the slot of its last occurrence
                  by_count     its slot count c
                  by_symbol    its symbol id (32 values)
                  query        whether it is the query symbol
                each sums to the misallocation; log_ratio_by_last is the mean
                log(p~/q) by last-occurrence slot (preference, not KL)

Writes choice_mechanism.json in the results folder and prints a summary.
"""
import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.collapse import REAL_EVAL_SEED, REAL_EVAL_SIZE, real_dataset
from src.patching import run
from src.train import load_checkpoint, set_threads

CHAINS = ["ext_t1", "ext_t1_s100", "ext_real10", "ext_t07",
          "ext_long", "ext_t07_s100", "ext_real10_s100", "ext_genctx", "ext_accum"]   # missing ones are skipped


def gen_number(path):
    return int(re.search(r"gen(\d+)\.pt$", path).group(1))


@torch.no_grad()
def analyse(model, cfg, data, batch=1000):
    K, S, H = cfg.n_pairs, cfg.n_symbols, cfg.n_heads
    q = 2 * K
    p = q + 1
    sym_pos = torch.arange(0, q, 2)
    lab_pos = torch.arange(1, q, 2)
    parts = ["embed+L0"] + [f"L1H{h}" for h in range(H)] + ["L1 MLP", "LN bias"]
    W = model.head.weight[:S]                                          # [S, C] symbol unembedding
    gamma, beta = model.ln_f.weight, model.ln_f.bias
    blk = model.blocks[1]

    attn = torch.zeros(H, K + 3)                                       # 8 slots, labels, query, self
    slope_num = torch.zeros(len(parts)); slope_den = 0.0
    in_out = torch.zeros(len(parts))
    ov = [[] for _ in range(H)]
    mis = {"by_slot": torch.zeros(K), "by_last": torch.zeros(K), "by_count": torch.zeros(K + 1),
           "by_symbol": torch.zeros(S), "query": torch.zeros(2)}
    lr_sum, lr_n = torch.zeros(K), torch.zeros(K)
    total_mis, n = 0.0, 0

    for i in range(0, data.shape[0], batch):
        seq = data[i:i + batch]
        B = seq.shape[0]
        rows = torch.arange(B)
        logits, cache = run(model, seq)
        a = cache["att"][1][:, :, p]                                   # [B, H, T]
        attn[:, :K] += a[:, :, sym_pos].sum(0)
        attn[:, K] += a[:, :, lab_pos].sum((0, 2))
        attn[:, K + 1] += a[:, :, q].sum(0)
        attn[:, K + 2] += a[:, :, p].sum(0)

        # residual parts at the choice position
        r0 = cache["resid_in"][1][:, p]                                # embedding + layer 0
        heads = cache["contrib"][1][:, :, p]                           # [B, H, C]
        mid = cache["resid_in"][1] + cache["contrib"][1].sum(1)
        mlp = blk.mlp(blk.ln2(mid))[:, p]
        x = r0 + heads.sum(1) + mlp
        sigma = (x.var(-1, unbiased=False, keepdim=True) + model.ln_f.eps).sqrt()
        comps = [r0] + [heads[:, h] for h in range(H)] + [mlp]
        dla = [((c - c.mean(-1, keepdim=True)) * gamma / sigma) @ W.T for c in comps]
        dla.append((beta @ W.T).expand(B, S))                          # [B, S] each
        check = sum(dla) - logits[:, p, :S]
        assert check.abs().max() < 1e-3, f"DLA does not add up ({check.abs().max():.2e})"

        ctx = seq[:, 0:q:2]
        counts = torch.zeros(B, S).scatter_add_(1, ctx, torch.ones(ctx.shape))
        inside = counts > 0
        logc = torch.where(inside, counts.clamp_min(1).log(), torch.zeros_like(counts))
        n_in = inside.sum(1, keepdim=True)
        x_c = torch.where(inside, logc - (logc * inside).sum(1, keepdim=True) / n_in, torch.zeros_like(logc))
        slope_den += float((x_c ** 2).sum())
        for k, d in enumerate(dla):
            d_c = torch.where(inside, d - (d * inside).sum(1, keepdim=True) / n_in, torch.zeros_like(d))
            slope_num[k] += float((x_c * d_c).sum())
            in_out[k] += float(((d * inside).sum(1) / n_in.squeeze(1)
                                - (d * ~inside).sum(1) / (~inside).sum(1)).sum())
        # attention to each symbol (summed over its slots) against the head's logit for it
        for h in range(H):
            att_sym = torch.zeros(B, S).scatter_add_(1, ctx, a[:, h, sym_pos])
            ov[h].append(torch.stack([att_sym[inside], dla[1 + h][inside]]))

        # exact misallocation terms
        ps = logits[:, p, :S].float().softmax(-1)
        pt = ps * inside / (ps * inside).sum(1, keepdim=True)
        qs = counts / K
        term = torch.where(inside, qs * (qs.clamp_min(1e-12).log() - pt.clamp_min(1e-30).log()),
                           torch.zeros_like(qs))
        total_mis += float(term.sum())
        slot_term = term.gather(1, ctx) / counts.gather(1, ctx)        # [B, K] each slot's share
        mis["by_slot"] += slot_term.sum(0)
        last = torch.full((B, S), -1, dtype=torch.long)
        for j in range(K):
            last[rows, ctx[:, j]] = j
        for j in range(K):
            m = last == j
            mis["by_last"][j] += float(term[m].sum())
            lr = (pt.clamp_min(1e-30).log() - qs.clamp_min(1e-12).log())[m]
            lr_sum[j] += float(lr.sum()); lr_n[j] += m.sum()
        for c in range(1, K + 1):
            mis["by_count"][c] += float(term[counts == c].sum())
        mis["by_symbol"] += term.sum(0)
        isq = torch.zeros(B, S, dtype=torch.bool); isq[rows, seq[:, q]] = True
        mis["query"][1] += float(term[isq].sum()); mis["query"][0] += float(term[~isq].sum())
        n += B

    out = {
        "parts": parts,
        "attention_labels": [f"slot {j + 1}" for j in range(K)] + ["labels", "query", "self"],
        "attention": (attn / n).tolist(),
        "count_slope": (slope_num / slope_den).tolist(),
        "in_out": (in_out / n).tolist(),
        "ov_corr": [float(torch.corrcoef(torch.cat(v, 1))[0, 1]) for v in ov],
        "misallocation": total_mis / n,
        "log_ratio_by_last": (lr_sum / lr_n.clamp_min(1)).tolist(),
    }
    for k, v in mis.items():
        out["mis_" + k] = (v / n).tolist()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root", nargs="?", default="results/v2")
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args()
    set_threads(args.threads)
    out, data = {}, None
    for run_name in CHAINS:
        paths = sorted(glob.glob(os.path.join(args.root, run_name, "gen*.pt")), key=gen_number)
        if not paths:
            continue
        rows = []
        for path in paths:
            model, cfg, _ = load_checkpoint(path, device="cpu")
            if data is None:
                data = real_dataset(cfg, REAL_EVAL_SEED, REAL_EVAL_SIZE)
            rec = analyse(model, cfg, data)
            rec["gen"] = gen_number(path)
            rows.append(rec)
            print(f"{run_name} g{rec['gen']} mis {rec['misallocation']:.3f} | count slope "
                  + " ".join(f"{k}={v:+.2f}" for k, v in zip(rec["parts"], rec["count_slope"]))
                  + " | total " + f"{sum(rec['count_slope']):.2f}"
                  + " | slot attn L1 " + " ".join(f"{sum(r[:cfg.n_pairs]):.2f}" for r in rec["attention"])
                  + " | ov " + " ".join(f"{v:.2f}" for v in rec["ov_corr"])
                  + " | mis by slot " + " ".join(f"{v:.3f}" for v in rec["mis_by_slot"]))
        out[run_name] = rows
    path = os.path.join(args.root, "choice_mechanism.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
