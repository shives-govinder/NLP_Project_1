"""Final train accuracy of every generation of a collapse run, without retraining.

    python scripts/train_accuracy.py results/v2/ext_t1 [results/v2/base ...] [--threads 8]

Runs before train accuracy was logged (src/train.py) still saved each
generation's model (gen{g}.pt) and the exact data it trained on
(data_gen{g}.pt), so the end-of-training train accuracy can be recomputed by
scoring each model on its own training data. For each generation it records:

  train_query_acc       argmax at the query matches the query label in the training data
  train_next_label_acc  (extended) same for the next symbol's label
  train_context_acc     same for every repeated context symbol's label (the dense targets)
  train_loss            the training objective averaged over the whole dataset
  data_query_ok         share of training rows whose query label agrees with the context
  data_next_label_ok    (extended) same for the next label (rows whose next symbol is in the context)
  real_query_acc,       accuracy on held-out real data, copied from collapse.json
  real_next_label_acc

From generation 1 on, the training labels were *sampled* by the previous model,
so train accuracy is measured against those samples, errors included. A model
that copies from the context instead of learning the errors scores about
data_query_ok on its training data and about 1 on real data.

Writes train_accuracy.json into each run folder and prints a table.
"""
import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.data import dense_targets, extended_targets
from src.metrics import dense_loss, query_loss
from src.train import auto_device, load_checkpoint, set_threads


def gen_number(path):
    return int(re.search(r"gen(\d+)\.pt$", path).group(1))


def context_label(seq, n_pairs, sym):
    """The label the context gives symbol ``sym`` (-1 if it is not in the context). [B]"""
    syms, labels = seq[:, 0:2 * n_pairs:2], seq[:, 1:2 * n_pairs:2]
    match = syms == sym[:, None]
    first = match.float().argmax(1)
    lab = labels.gather(1, first[:, None])[:, 0]
    return torch.where(match.any(1), lab, torch.full_like(lab, -1))


@torch.no_grad()
def score(model, cfg, data, batch_size=4096):
    """Train metrics of ``model`` on its training ``data`` [N, T] (long)."""
    model.eval()
    q = 2 * cfg.n_pairs                                   # query position
    extended = cfg.variant == "extended"
    tot = {k: 0.0 for k in ("query", "next", "ctx", "ctx_n", "loss", "q_ok", "n_ok", "n_in")}
    for i in range(0, data.shape[0], batch_size):
        seq = data[i:i + batch_size].to(cfg.device)
        B = seq.shape[0]
        q_label = seq[:, q + 1]
        if extended:
            logits = model(seq)
            tgt = extended_targets(seq, cfg.n_pairs, context_targets=cfg.dense_loss, targets=cfg.ext_targets)
            tot["next"] += float((logits[:, q + 2].argmax(-1) == seq[:, q + 3]).sum())
            true_next = context_label(seq, cfg.n_pairs, seq[:, q + 2])
            tot["n_in"] += float((true_next >= 0).sum())
            tot["n_ok"] += float((seq[:, q + 3] == true_next).sum())
        else:
            logits = model(seq[:, :-1])                   # base rows: context, query, query label
            tgt = dense_targets(seq[:, :-1], q_label) if cfg.dense_loss else None
        loss = dense_loss(logits, tgt) if tgt is not None else query_loss(logits, q_label)
        tot["loss"] += float(loss) * B
        tot["query"] += float((logits[:, q].argmax(-1) == q_label).sum())
        tot["q_ok"] += float((q_label == context_label(seq, cfg.n_pairs, seq[:, q])).sum())
        ctx_tgt = dense_targets(seq[:, :q + 1], q_label)[:, :q]   # repeated context symbols only
        mask = ctx_tgt != -100
        tot["ctx"] += float((logits[:, :q].argmax(-1)[mask] == ctx_tgt[mask]).sum())
        tot["ctx_n"] += float(mask.sum())

    N = data.shape[0]
    out = {"train_query_acc": tot["query"] / N, "train_context_acc": tot["ctx"] / max(tot["ctx_n"], 1),
           "train_loss": tot["loss"] / N, "data_query_ok": tot["q_ok"] / N}
    if extended:
        out["train_next_label_acc"] = tot["next"] / N
        out["data_next_label_ok"] = tot["n_ok"] / max(tot["n_in"], 1)
    return out


def analyse(run_dir, device):
    paths = sorted(glob.glob(os.path.join(run_dir, "gen*.pt")), key=gen_number)
    if not paths:
        sys.exit(f"no gen*.pt in {run_dir}: copy the checkpoints from the cluster first")
    real = {}
    if os.path.exists(os.path.join(run_dir, "collapse.json")):
        with open(os.path.join(run_dir, "collapse.json")) as f:
            real = {g["gen"]: g.get("real_eval", {}) for g in json.load(f)["generations"]}

    rows = []
    for p in paths:
        g = gen_number(p)
        data_path = os.path.join(run_dir, f"data_gen{g}.pt")
        if not os.path.exists(data_path):
            print(f"gen {g}: no {data_path}, skipped")
            continue
        model, cfg, _ = load_checkpoint(p, device=device)
        rec = {"gen": g, **score(model, cfg, torch.load(data_path).long())}
        rec["real_query_acc"] = real.get(g, {}).get("query_label_acc")
        rec["real_next_label_acc"] = real.get(g, {}).get("next_label_acc")
        rows.append(rec)

    with open(os.path.join(run_dir, "train_accuracy.json"), "w") as f:
        json.dump(rows, f, indent=1)

    fmt = lambda v: "-" if v is None else f"{v:.4f}"
    print(f"\n{run_dir}")
    print(f"{'gen':>3} {'train q':>8} {'data q ok':>9} {'real q':>7} {'train ctx':>9} "
          f"{'train nl':>8} {'data nl ok':>10} {'real nl':>7} {'train loss':>10}")
    for r in rows:
        print(f"{r['gen']:>3} {fmt(r['train_query_acc']):>8} {fmt(r['data_query_ok']):>9} "
              f"{fmt(r['real_query_acc']):>7} {fmt(r['train_context_acc']):>9} "
              f"{fmt(r.get('train_next_label_acc')):>8} {fmt(r.get('data_next_label_ok')):>10} "
              f"{fmt(r['real_next_label_acc']):>7} {r['train_loss']:>10.4f}")
    print(f"saved {os.path.join(run_dir, 'train_accuracy.json')}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--device", default=None)
    p.add_argument("--threads", type=int, default=None)
    args = p.parse_args()
    set_threads(args.threads)
    device = auto_device(args.device)
    for d in args.run_dirs:
        analyse(d, device)


if __name__ == "__main__":
    main()
