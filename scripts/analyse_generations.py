"""Compare the induction circuit across model-collapse generations.

    python -m src.collapse --variant base ... --save_dir results/collapse_base
    python scripts/analyse_generations.py results/collapse_base mps

For every gen*.pt in the folder it measures accuracy, the previous-token and
induction heads, and how much accuracy drops when they are removed. It prints
a table, writes circuit_summary.json, and saves two figures:

    circuit_across_generations.png  accuracy, entropy and circuit strength per generation
    ablation_by_generation.png      per-head ablation heatmap for each generation

Head indices are arbitrary per training run, so compare the summary numbers
(best head score, accuracy without the key head / the whole last layer), not
specific head ids, across generations.
"""
import glob
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.interpret import circuit_summary
from src.train import TEST_SEED, auto_device, fixed_split, load_checkpoint

# Reference palette (light mode, for print): categorical slots 1-3, text inks, surface.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"


def gen_number(path):
    return int(re.search(r"gen(\d+)\.pt$", path).group(1))


def phase_change_step(history, threshold=0.9):
    """First logged step where validation accuracy reaches the threshold."""
    for rec in history:
        if rec.get("val_acc", 0.0) >= threshold:
            return rec["step"]
    return None


def style_axis(ax, title, ylabel):
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.set_xlabel("generation", color=INK_2, fontsize=9)
    ax.tick_params(colors=INK_2, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def line(ax, x, y, color, label=None):
    ax.plot(x, y, color=color, linewidth=1.8, marker="o", markersize=5,
            markeredgecolor=SURFACE, markeredgewidth=1.2, label=label)


def plot_trends(rows, n_labels, path):
    import matplotlib.pyplot as plt

    gens = [r["gen"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.2), facecolor=SURFACE)
    (a, b), (c, d) = axes

    line(a, gens, [r["test_acc"] for r in rows], BLUE)
    a.set_ylim(0, 1.05)
    style_axis(a, "Test accuracy on real data", "accuracy")

    ent = [r["label_entropy"] for r in rows]
    if all(e is not None for e in ent):
        line(b, gens, ent, BLUE)
        b.axhline(math.log(n_labels), color=INK_2, linewidth=1, linestyle="--")
        b.text(gens[-1], math.log(n_labels), " uniform", color=INK_2, fontsize=8, va="bottom", ha="right")
        b.set_ylim(0, math.log(n_labels) * 1.1)
    else:
        b.text(0.5, 0.5, "collapse.json not found", ha="center", va="center", color=INK_2, transform=b.transAxes)
    style_axis(b, "Entropy of output labels", "entropy (nats)")

    line(c, gens, [r["prev_head_score"] for r in rows], BLUE, "previous-token head (layer 0)")
    line(c, gens, [r["induction_head_score"] for r in rows], ORANGE, "induction head (last layer)")
    c.set_ylim(0, 1.05)
    style_axis(c, "Attention: strongest head of each type", "attention score")
    c.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="lower left")

    line(d, gens, [r["test_acc"] for r in rows], BLUE, "intact model")
    line(d, gens, [r["acc_without_all_prev_heads"] for r in rows], ORANGE, "all previous-token heads removed")
    line(d, gens, [r["acc_without_last_layer"] for r in rows], AQUA, "whole last layer removed")
    d.set_ylim(0, 1.05)
    style_axis(d, "Ablation: accuracy with circuit parts removed", "accuracy")
    d.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="center right")

    for ax in axes.flat:
        ax.set_xticks(gens)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def plot_ablation_grids(rows, path):
    import matplotlib.pyplot as plt

    n = len(rows)
    fig, axes = plt.subplots(1, n, figsize=(2.3 * n + 1.2, 1.9), facecolor=SURFACE, squeeze=False)
    im = None
    for ax, r in zip(axes[0], rows):
        grid = r["ablation_grid"]
        im = ax.imshow(grid, cmap="Blues", vmin=0, vmax=1, aspect="equal")
        for i, row in enumerate(grid):
            for j, v in enumerate(row):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color=SURFACE if v > 0.6 else INK)
        ax.set_title(f"gen {r['gen']}", color=INK, fontsize=9)
        ax.set_xticks(range(len(grid[0])))
        ax.set_xticklabels([f"H{h}" for h in range(len(grid[0]))], fontsize=7, color=INK_2)
        ax.set_yticks(range(len(grid)))
        ax.set_yticklabels([f"L{l}" for l in range(len(grid))], fontsize=7, color=INK_2)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    cbar = fig.colorbar(im, ax=axes[0].tolist(), shrink=0.55, pad=0.015, aspect=12)
    cbar.set_label("accuracy with head removed", color=INK_2, fontsize=8)
    cbar.ax.tick_params(labelsize=7, colors=INK_2)
    fig.suptitle("Per-head ablation (head numbers are not comparable across generations)",
                 color=INK, fontsize=9, x=0.02, ha="left")
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


EXT_PANELS = [
    ("Where the chosen next symbol comes from", "fraction of sequences",
     [("next_symbol_in_context", BLUE, "appears in the context"),
      ("next_symbol_is_query", ORANGE, "is the query symbol")], (0, 1.05)),
    ("Entropy of chosen symbol ids", "entropy (nats)",
     [("symbol_id_entropy", BLUE, None)], None),
    ("Generated labels that match the context", "fraction correct",
     [("query_label_correct", BLUE, "query label"),
      ("next_label_correct", ORANGE, "next-symbol label")], (0, 1.05)),
    ("Entropy of which context symbol is chosen", "entropy (nats)",
     [("choice_rank_entropy", BLUE, None)], None),
]


def plot_extended(gens, stats, ref, path):
    """What each generation wrote into the next generation's training data,
    against the same statistics measured on real data (dashed lines)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(9, 6.2), facecolor=SURFACE)
    for ax, (title, ylabel, series, ylim) in zip(axes.flat, EXT_PANELS):
        top = 0.0
        for key, color, label in series:
            ys = [stats[g][key] for g in gens]
            line(ax, gens, ys, color, label)
            top = max(top, max(ys))
            if ref is not None:
                ax.axhline(ref[key], color=color, linewidth=1, linestyle="--", alpha=0.8)
                top = max(top, ref[key])
        ax.set_ylim(*(ylim or (0, top * 1.15 if top > 0 else 1)))
        style_axis(ax, title, ylabel)
        ax.set_xlabel("generation (data it wrote for the next one)", color=INK_2, fontsize=9)
        ax.set_xticks(gens)
        if any(lbl for _, _, lbl in series):
            ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="lower left")
    fig.text(0.01, 0.005, "dashed lines = the same measure on real data", color=INK_2, fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "results/collapse_base"
    device = auto_device(sys.argv[2] if len(sys.argv) > 2 else None)
    paths = sorted(glob.glob(os.path.join(run_dir, "gen*.pt")), key=gen_number)
    if not paths:
        sys.exit(f"No gen*.pt files in {run_dir}. Run src.collapse with --save_dir {run_dir} first.")

    entropy, ext_stats, ext_ref = {}, {}, None
    cj = os.path.join(run_dir, "collapse.json")
    if os.path.exists(cj):
        with open(cj) as f:
            cjd = json.load(f)
        entropy = {g["gen"]: g["next_label_entropy"] for g in cjd["generations"]}
        ext_stats = {g["gen"]: g["extended_stats"] for g in cjd["generations"] if g.get("extended_stats")}
        ext_ref = cjd.get("real_reference")

    rows, n_labels = [], None
    for path in paths:
        model, cfg, history = load_checkpoint(path, device=device)
        n_labels = cfg.n_labels
        summary = circuit_summary(model, cfg, fixed_split(cfg, TEST_SEED))
        summary.update(gen=gen_number(path), seed=cfg.seed,
                       phase_change_step=phase_change_step(history),
                       label_entropy=entropy.get(gen_number(path)))
        rows.append(summary)
        print(f"analysed {path}")

    print(f"\n{'gen':>3} {'seed':>4} {'phase@':>7} {'acc':>6} {'entropy':>8} "
          f"{'prev(head)':>12} {'induct(head)':>13} {'#prev':>6} {'#ind':>5} "
          f"{'acc-prev':>9} {'acc-allprev':>12} {'acc-allind':>11} {'acc-lastL':>10}")
    for r in rows:
        ent = "-" if r["label_entropy"] is None else f"{r['label_entropy']:.4f}"
        phase = "-" if r["phase_change_step"] is None else str(r["phase_change_step"])
        print(f"{r['gen']:>3} {r['seed']:>4} {phase:>7} {r['test_acc']:>6.3f} {ent:>8} "
              f"{r['prev_head_score']:>7.3f}(H{r['prev_head']}) "
              f"{r['induction_head_score']:>8.3f}(H{r['induction_head']}) "
              f"{len(r['prev_heads']):>6} {len(r['induction_heads']):>5} "
              f"{r['acc_without_prev_head']:>9.3f} {r['acc_without_all_prev_heads']:>12.3f} "
              f"{r['acc_without_all_induction_heads']:>11.3f} {r['acc_without_last_layer']:>10.3f}")

    if ext_stats:
        keys = ["query_label_correct", "next_symbol_in_context", "next_symbol_is_query",
                "next_label_correct", "symbol_id_entropy", "choice_rank_entropy", "label_entropy"]
        short = ["q-lab ok", "in ctx", "=query", "n-lab ok", "H(symid)", "H(choice)", "H(labels)"]
        print("\nwhat each generation wrote for the next one (extended variant):")
        print(f"{'':>6} " + " ".join(f"{h:>9}" for h in short))
        if ext_ref:
            print(f"{'real':>6} " + " ".join(f"{ext_ref[k]:>9.3f}" for k in keys))
        for g in sorted(ext_stats):
            print(f"{'gen ' + str(g):>6} " + " ".join(f"{ext_stats[g][k]:>9.3f}" for k in keys))
        gens_ext = sorted(ext_stats)
        plot_extended(gens_ext, ext_stats, ext_ref, os.path.join(run_dir, "extended_generated_data.png"))
        print(f"saved extended_generated_data.png in {run_dir}")

    out_json = os.path.join(run_dir, "circuit_summary.json")
    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2)
    plot_trends(rows, n_labels, os.path.join(run_dir, "circuit_across_generations.png"))
    plot_ablation_grids(rows, os.path.join(run_dir, "ablation_by_generation.png"))
    print(f"\nsaved {out_json}, circuit_across_generations.png and ablation_by_generation.png in {run_dir}")


if __name__ == "__main__":
    main()
