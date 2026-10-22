"""Embedding geometry and attention maps across the generations of a collapse run.

    python scripts/embeddings.py results/v2/ext_t1 --compare results/v2/ext_t07 results/v2/base

For each gen*.pt it records src.interpret.embedding_geometry: symbol and label
norms, how orthogonal symbols / labels are, the symbol-label centroid gap and a
2-D PCA. When the run folder also holds data_gen{g}.pt (the data generation g
trained on), it correlates each symbol's embedding norm with how often that
symbol was the next-symbol choice in that data: do symbols the synthetic data
over-represents become special in the embedding?

Writes embeddings.json into each run folder and figures/fig_embeddings.png:
(a) PCA at gen 0, (b) PCA at the last generation, (c) mean |cos| among symbols
(mean symbol removed) by generation for each run, (d, e) attention of the strongest previous-token
head and induction head on one real sequence (last generation of the first run).
"""
import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from paper_figures import CHAINS, INK_2, MARKERS, SERIES, style
from src.data import decode_tokens, make_icl_batch
from src.interpret import attention_patterns, embedding_geometry, induction_score, previous_token_score
from src.train import load_checkpoint


def gen_number(path):
    return int(re.search(r"gen(\d+)\.pt$", path).group(1))


def choice_counts(path, cfg):
    """How often each symbol is the next-symbol choice in one generation's training data."""
    data = torch.load(path).long()
    if data.shape[1] != 2 * cfg.n_pairs + 4:          # base-variant data has no next symbol
        return None
    return torch.bincount(data[:, 2 * cfg.n_pairs + 2], minlength=cfg.n_symbols)[:cfg.n_symbols].float()


def corr(x, y):
    x, y = x - x.mean(), y - y.mean()
    return float((x * y).sum() / (x.norm() * y.norm()).clamp(min=1e-12))


def analyse(run_dir):
    paths = sorted(glob.glob(f"{run_dir}/gen*.pt"), key=gen_number)
    if not paths:
        sys.exit(f"no gen*.pt in {run_dir}: copy the checkpoints from the cluster first")
    print(f"\n{run_dir}")
    rows = []
    for p in paths:
        model, cfg, _ = load_checkpoint(p)
        g = gen_number(p)
        geo = embedding_geometry(model, cfg)
        geo["gen"] = g
        data_path = os.path.join(run_dir, f"data_gen{g}.pt")
        counts = choice_counts(data_path, cfg) if os.path.exists(data_path) else None
        if counts is not None:
            geo["choice_share"] = (counts / counts.sum()).tolist()
            geo["norm_vs_choice_r"] = corr(torch.tensor(geo["sym_norm"]), counts)
        rows.append(geo)
        print(f"gen {g} | |cos| symbols {geo['sym_abs_cos']:.3f} (centred {geo['sym_abs_cos_centred']:.3f}) labels {geo['label_abs_cos']:.3f} "
              f"(random {geo['random_abs_cos']:.3f}) | centroid gap {geo['centroid_gap']:.2f} "
              f"| PCA var {geo['pca_var'][0]:.2f}, {geo['pca_var'][1]:.2f}"
              + (f" | r(norm, choice share) {geo['norm_vs_choice_r']:+.2f}" if counts is not None else ""))
    with open(os.path.join(run_dir, "embeddings.json"), "w") as f:
        json.dump(rows, f, indent=1)
    return rows, paths[-1]


def pca_panel(ax, geo, n_symbols, title):
    pcs = torch.tensor(geo["pca"])
    ax.scatter(pcs[:n_symbols, 0], pcs[:n_symbols, 1], s=14, color=SERIES[0], label="symbols")
    ax.scatter(pcs[n_symbols:, 0], pcs[n_symbols:, 1], s=26, color=SERIES[1], marker="s", label="labels")
    v = geo["pca_var"]
    style(ax, title, f"PC2 ({100 * v[1]:.0f}% of variance)", f"PC1 ({100 * v[0]:.0f}% of variance)")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("run_dir")
    p.add_argument("--compare", nargs="*", default=[], help="more run folders for panel (c)")
    p.add_argument("--out", default="results/v2/figures/fig_embeddings.png")
    args = p.parse_args()

    import matplotlib.pyplot as plt

    rows, last_path = analyse(args.run_dir)
    runs = [(args.run_dir, rows)] + [(d, analyse(d)[0]) for d in args.compare]
    model, cfg, _ = load_checkpoint(last_path)

    fig, axes = plt.subplots(1, 5, figsize=(15, 3.3), gridspec_kw={"width_ratios": [1, 1, 1, 1.1, 1.1]})
    pca_panel(axes[0], rows[0], cfg.n_symbols, f"(a) Embedding PCA, gen {rows[0]['gen']}")
    pca_panel(axes[1], rows[-1], cfg.n_symbols, f"(b) Embedding PCA, gen {rows[-1]['gen']}")
    axes[0].legend(frameon=False, fontsize=7, labelcolor=INK_2)

    c = axes[2]
    chain_index = {run: i for i, (run, _) in enumerate(CHAINS)}   # same colour per chain as the other figures
    for d, rs in runs:
        run = os.path.basename(os.path.normpath(d))
        i = chain_index.get(run)
        color, marker = (SERIES[i], MARKERS[i]) if i is not None else (INK_2, "^")
        label = dict(CHAINS).get(run, run)
        c.plot([r["gen"] for r in rs], [r["sym_abs_cos_centred"] for r in rs], color=color,
               marker=marker, linewidth=2, label=label)
    c.axhline(rows[0]["random_abs_cos"], color=INK_2, linestyle="--", linewidth=1)
    c.text(len(rows) - 1, rows[0]["random_abs_cos"] - 0.01, "random vectors", color=INK_2, fontsize=7,
           va="top", ha="right")
    style(c, "(c) How distinct symbol identities are", "mean |cos| between centred symbols")
    c.xaxis.get_major_locator().set_params(integer=True)
    c.legend(frameon=False, fontsize=7, labelcolor=INK_2)

    # (d, e) the strongest previous-token and induction heads on one real sequence
    torch.manual_seed(0)
    seq, _ = make_icl_batch(1, cfg.n_pairs, cfg.n_symbols, cfg.n_labels, n_unique=cfg.n_unique)
    att = attention_patterns(model, seq)
    prev_h = int(previous_token_score(model, cfg)[0].argmax())
    ind_h = int(induction_score(model, cfg)[-1].argmax())
    toks = decode_tokens(seq[0], cfg.n_symbols).split(" ")
    heads = ((0, prev_h, "d", "previous-token"), (cfg.n_layers - 1, ind_h, "e", "induction"))
    for ax, (layer, head, letter, name) in zip(axes[3:], heads):
        ax.imshow(att[layer][0, head].cpu().numpy(), cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(toks)))
        ax.set_yticks(range(len(toks)))
        ax.set_xticklabels(toks, rotation=90, fontsize=5.5)
        ax.set_yticklabels(toks, fontsize=5.5)
        ax.set_title(f"({letter}) L{layer}H{head}, {name} head", fontsize=9.5, loc="left")
        ax.set_xlabel("attended-to token", fontsize=8, color=INK_2)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=250)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
