"""Figures for the extended abstract, built from results/v2.

    python scripts/paper_figures.py            # writes results/v2/figures/*.png

fig_collapse.png  how each extended chain's next-symbol choice drifts from the
                  truth on held-out real data: leakage out of the context,
                  misallocation within it, frequency sharpening, and the
                  per-slot-count choice probabilities (lost tails).
fig_interventions.png
                  (a) deleting the erroneous rows of generation 7's data vs
                  deleting random rows (scripts/formation.py), (b) when the
                  induction circuit forms under each training target, and
                  (c) which path carries the previous-token head's effect
                  (scripts/mechanism.py), across all 38 models.

Needs results/v2/choice_profiles.json (computed from every gen*.pt with
src.patching.choice_profile; see compute_profiles below).
"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Reference palette, light mode: categorical slots 1-4 in fixed order (validated:
# adjacent CVD dE >= 9.1; slots 3-4 below 3:1 contrast, so every line is direct-labelled).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MARKERS = ["o", "s", "^", "D"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"

CHAINS = [  # (run folder, label) - order fixes each chain's colour everywhere
    ("ext_t1", "T = 1"),
    ("ext_t1_s100", "T = 1, seed 100"),
    ("ext_real10", "T = 1, 10% real"),
    ("ext_t07", "T = 0.7"),
]
ROOT = "results/v2"


def compute_profiles(path):
    import torch
    from src.collapse import REAL_EVAL_SEED, REAL_EVAL_SIZE, real_dataset
    from src.patching import choice_profile
    from src.train import load_checkpoint

    out, data = {}, None
    for run, _ in CHAINS:
        out[run] = []
        for p in sorted(glob.glob(f"{ROOT}/{run}/gen*.pt"), key=lambda p: int(re.search(r"gen(\d+)\.pt$", p).group(1))):
            model, cfg, _ = load_checkpoint(p)
            if data is None:
                data = real_dataset(cfg, REAL_EVAL_SEED, REAL_EVAL_SIZE)
            cp = choice_profile(model, cfg, data)
            cp["gen"] = int(re.search(r"gen(\d+)\.pt$", p).group(1))
            out[run].append(cp)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    return out


def style(ax, title, ylabel, xlabel="generation"):
    ax.set_title(title, color=INK, fontsize=9.5, loc="left")
    ax.set_ylabel(ylabel, color=INK_2, fontsize=8.5)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=8.5)
    ax.tick_params(colors=INK_2, labelsize=7.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def line(ax, xs, ys, i, label):
    ax.plot(xs, ys, color=SERIES[i], linewidth=2, marker=MARKERS[i], markersize=5.5,
            markeredgecolor=SURFACE, markeredgewidth=1.0, label=label)
    ax._end_labels = getattr(ax, "_end_labels", []) + [(xs[-1], ys[-1], label)]


def place_end_labels(ax, min_gap=0.045):
    """Direct-label each line at its last point, nudging labels apart vertically
    (in axes fraction, so it works on log axes too). Call after limits are final."""
    items = getattr(ax, "_end_labels", [])
    to_frac = ax.transScale + ax.transLimits
    pos = sorted(((to_frac.transform((x, y))[1], x, text) for x, y, text in items))
    placed = []
    for f, x, text in pos:
        if placed and f - placed[-1][0] < min_gap:
            f = placed[-1][0] + min_gap
        placed.append((f, x, text))
    for f, x, text in placed:
        ax.annotate(text, xy=(x, f), xycoords=("data", "axes fraction"), xytext=(6, 0),
                    textcoords="offset points", fontsize=7, color=INK_2, va="center")


def fig_collapse(profiles, out):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(9, 6.4), facecolor=SURFACE)
    (a, b), (c, d) = axes
    for i, (run, label) in enumerate(CHAINS):
        rows = profiles[run]
        g = [r["gen"] for r in rows]
        line(a, g, [r["mass_out"] for r in rows], i, label)
        line(b, g, [r["kl_in"] for r in rows], i, label)
        line(c, g, [r["slope"] for r in rows], i, label)

    style(a, "(a) Probability on symbols not in the context", "probability (real: 0)")
    a.set_ylim(bottom=0)
    style(b, "(b) Probability misallocated among context symbols", "KL within context (nats, log scale)")
    b.set_yscale("log")
    style(c, "(c) Frequency sharpening", "slope of model on true choice probability")
    c.axhline(1, color=INK_2, linewidth=1, linestyle="--")
    c.text(3.5, 0.985, "1 = copies frequencies exactly", color=INK_2, fontsize=7, va="top", ha="center")

    # (d) choice probability by slot count: real data, gen 0, and gen 7 of two chains
    counts = [1, 2, 3, 4, 5]
    d.plot(counts, [k / 8 for k in counts], color=INK_2, linewidth=1.2, linestyle="--", label="real data (c / 8)")
    d._end_labels = [(counts[-1], 5 / 8, "real data")]
    for run, label in (("ext_t1", "T = 1"), ("ext_t07", "T = 0.7")):
        i = [r for r, _ in CHAINS].index(run)
        last = profiles[run][-1]
        line(d, counts, [last["by_count"][str(k)] if str(k) in last["by_count"] else last["by_count"][k]
                         for k in counts], i, f"{label}, gen {last['gen']}")
    style(d, "(d) Lost tails: choice probability by slot count", "mean choice probability", "slots the symbol fills (of 8)")
    d.set_xticks(counts)
    d.set_ylim(0, 1)

    for ax in (a, b, c):
        ax.set_xticks(range(0, 8))
        ax.margins(x=0.22)
    d.margins(x=0.25)
    a.legend(frameon=False, fontsize=7.5, labelcolor=INK_2, loc="upper left")
    c.set_ylim(0.9, None)
    for ax in axes.flat:
        place_end_labels(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=250, facecolor=SURFACE)
    plt.close(fig)


def strip(ax, x, ys, color, marker="o", width=0.22):
    """Each seed / model as a dot (small deterministic jitter) plus a median tick."""
    import statistics
    n = len(ys)
    for k, y in enumerate(ys):
        dx = 0 if n == 1 else (k / (n - 1) - 0.5) * width
        ax.plot(x + dx, y, marker=marker, color=color, markersize=5, markeredgecolor=SURFACE,
                markeredgewidth=0.8, linestyle="none")
    ax.plot([x - 0.3, x + 0.3], [statistics.median(ys)] * 2, color=INK, linewidth=1.5)


def fig_interventions(out):
    import matplotlib.pyplot as plt

    def runs(name):
        with open(f"{ROOT}/formation/{name}.json") as f:
            return json.load(f)["runs"]

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(10.5, 3.4), facecolor=SURFACE,
                                  gridspec_kw={"width_ratios": [1.35, 1, 1.15]})

    # (a) deletion: leakage and within-context misallocation, 3 seeds each
    conds = [("real", "real data"), ("gen7", "gen-7 data"), ("gen7_del_errors", "erroneous\nrows deleted"),
             ("gen7_del_random", "random rows\ndeleted")]
    for i, (name, _) in enumerate(conds):
        rs = runs(name)
        strip(a, i - 0.18, [r["choice"]["leak"] for r in rs], SERIES[0], MARKERS[0], width=0.16)
        strip(a, i + 0.18, [r["choice"]["kl_in"] for r in rs], SERIES[1], MARKERS[1], width=0.16)
    a.plot([], [], marker=MARKERS[0], color=SERIES[0], linestyle="none", label="leakage out of the context")
    a.plot([], [], marker=MARKERS[1], color=SERIES[1], linestyle="none", label="misallocation within it")
    a.set_xticks(range(len(conds)))
    a.set_xticklabels([l for _, l in conds], fontsize=7.5)
    a.set_ylim(0, 0.55)
    style(a, "(a) Deleting the erroneous synthetic rows", "KL on real data (nats)", "")
    a.legend(frameon=False, fontsize=7, labelcolor=INK_2, loc="upper left")

    # (b) phase-change step under each training target (real data)
    targets = [("targets_query", "query label\nonly"), ("targets_qsym", "+ next\nsymbol"), ("real", "+ next label\n(all)")]
    for i, (name, _) in enumerate(targets):
        strip(b, i, [r["phase_step"] for r in runs(name)], SERIES[0], MARKERS[0])
    b.set_xticks(range(len(targets)))
    b.set_xticklabels([l for _, l in targets], fontsize=7.5)
    b.set_ylim(0, 8500)
    style(b, "(b) When induction forms", "phase-change step", "training targets")

    # (c) share of the top layer-0 head's damage carried by each path into layer 1
    shares = {k: [] for k in ("k", "kd", "q", "v")}
    for run in ("base", "ext_t1", "ext_t1_s100", "ext_real10", "ext_t07"):
        with open(f"{ROOT}/{run}/mechanism.json") as f:
            for r in json.load(f):
                acc = r["query_circuit"]["acc"]
                h = min(range(len(acc["resample"][0])), key=lambda i: acc["resample"][0][i])
                full = acc["resample"][0][h]
                if full >= 0.9:          # no single layer-0 head matters in this model
                    continue
                dmg = 1 - full
                shares["k"].append((1 - acc["path_total"][h][1]) / dmg)
                shares["kd"].append((1 - acc["path"][h][1]) / dmg)
                shares["q"].append((1 - acc["path_total"][h][0]) / dmg)
                shares["v"].append((1 - acc["path_total"][h][2]) / dmg)
    paths = [("k", "keys"), ("kd", "keys,\ndirect only"), ("q", "queries"), ("v", "values")]
    for i, (k, _) in enumerate(paths):
        strip(c, i, shares[k], SERIES[0], MARKERS[0], width=0.5)
    c.set_xticks(range(len(paths)))
    c.set_xticklabels([l for _, l in paths], fontsize=7.5)
    c.set_ylim(-0.05, 1.15)
    style(c, "(c) Where the previous-token head acts",
          f"share of the head's effect ({len(shares['k'])} models)", "layer-1 input the head reaches")

    fig.tight_layout()
    fig.savefig(out, dpi=250, facecolor=SURFACE)
    plt.close(fig)


def main():
    os.makedirs(f"{ROOT}/figures", exist_ok=True)
    path = f"{ROOT}/choice_profiles.json"
    if os.path.exists(path):
        with open(path) as f:
            profiles = json.load(f)
    else:
        profiles = compute_profiles(path)
    fig_collapse(profiles, f"{ROOT}/figures/fig_collapse.png")
    print(f"saved {ROOT}/figures/fig_collapse.png")
    fig_interventions(f"{ROOT}/figures/fig_interventions.png")
    print(f"saved {ROOT}/figures/fig_interventions.png")


if __name__ == "__main__":
    main()
