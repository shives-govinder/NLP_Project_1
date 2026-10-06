"""Compare collapse chains on held-out real data (the `real_eval` records).

    python scripts/collapse_figures.py --out results/v2/collapse_curves.png \
        "T = 1=results/v2/ext_t1" "10% real kept=results/v2/ext_real10" "T = 0.7=results/v2/ext_t07"

Each argument is LABEL=RUN_DIR (a folder with collapse.json from src.collapse).
Extended chains fill the next-symbol panels; every chain fills the query-label
panel. Also prints a table per chain.
"""
import argparse
import json
import os

# Reference palette, light mode (categorical slots 1-3, validated all-pairs), text inks, surface.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
MARKERS = ["o", "s", "^"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"


def load(run_dir):
    with open(os.path.join(run_dir, "collapse.json")) as f:
        d = json.load(f)
    return d["config"], d["generations"]


def style(ax, title, ylabel):
    ax.set_title(title, color=INK, fontsize=10, loc="left")
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.set_xlabel("generation", color=INK_2, fontsize=9)
    ax.tick_params(colors=INK_2, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("runs", nargs="+", help="LABEL=RUN_DIR")
    p.add_argument("--out", default="results/v2/collapse_curves.png")
    args = p.parse_args()
    runs = []
    for spec in args.runs:
        label, run_dir = spec.split("=", 1)
        cfg, gens = load(run_dir)
        runs.append((label, cfg, gens))

    for label, cfg, gens in runs:
        print(f"\n{label} ({cfg['variant']}, T={cfg['collapse_temperature']}, real_frac={cfg.get('real_frac', 0)})")
        keys = list(gens[0]["real_eval"].keys())
        print("gen " + " ".join(f"{k[:16]:>16}" for k in keys) + f" {'gen out-of-ctx':>15}")
        for g in gens:
            ev = g["real_eval"]
            st = g["generated_stats"]
            out = 1 - st["next_symbol_in_context"] if "next_symbol_in_context" in st else float("nan")
            print(f"{g['gen']:>3} " + " ".join(f"{ev[k]:>16.4f}" for k in keys) + f" {out:>15.4f}")

    import matplotlib.pyplot as plt

    ext = [r for r in runs if r[1]["variant"] == "extended"]
    panels = [
        ("Next-symbol choice vs the truth", "KL(true || model), nats", lambda ev: ev["next_symbol_kl"], ext, None),
        ("Probability on symbols not in the context", "probability", lambda ev: ev["next_symbol_mass_out"], ext, None),
        ("Preference for the query symbol", "model / true probability",
         lambda ev: ev["next_symbol_query_prob"] / ev["next_symbol_query_true"], ext, 1.0),
        ("Query label: loss on real data", "NLL, nats", lambda ev: ev["query_label_nll"], runs, None),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.2), facecolor=SURFACE)
    for ax, (title, ylabel, fn, rs, ref) in zip(axes.flat, panels):
        for i, (label, cfg, gens) in enumerate(rs):
            xs = [g["gen"] for g in gens]
            ys = [fn(g["real_eval"]) for g in gens]
            k = runs.index((label, cfg, gens))
            ax.plot(xs, ys, color=SERIES[k], linewidth=2, marker=MARKERS[k], markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, label=label)
            ax.annotate(label, (xs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points",
                        fontsize=7.5, color=INK_2, va="center")
        if ref is not None:
            ax.axhline(ref, color=INK_2, linewidth=1, linestyle="--")
            ax.text(0, ref, "real data", color=INK_2, fontsize=7.5, va="bottom")
        ax.set_ylim(bottom=0)
        ax.margins(x=0.12)
        style(ax, title, ylabel)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor=SURFACE)
    print(f"\nsaved {args.out}")


if __name__ == "__main__":
    main()
