"""
Figure: Effect of graph topology on overall performance and subgroup disparity.

Publication-style (NeurIPS / ICML / Nature MI) two-panel figure.
Left  panel: overall AUROC across Sequential, TGTTemporal, TGTClinical, TGTFull.
Right panel: race AUROC gap (max - min across race subgroups) for the same variants.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
TASK10_DIR = REPO_ROOT / "Task10"
OUT_DIR = REPO_ROOT / "figures" / "main"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-paper")
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

CONDITION_ORDER = ["no_edge", "temporal_only", "clinical_only", "full_graph"]
DISPLAY_NAMES = ["LSTM", "TGTTemporal", "TGTClinical", "TGTFull"]

# Original palette ramp (navy = lowest value, terracotta = highest value).
VALUE_RAMP = ["#1f3b5c", "#64728b", "#be8b7e", "#a05a4a"]


def colors_by_rank(values):
    order = np.argsort(values)
    colors = [None] * len(values)
    for rank, idx in enumerate(order):
        colors[idx] = VALUE_RAMP[rank]
    return colors


def load_data():
    overall = pd.read_csv(TASK10_DIR / "Task10_GSA_perseed_overall.csv")
    gaps = pd.read_csv(TASK10_DIR / "Task10_GSA_perseed_gaps.csv")
    race_gaps = gaps[gaps["attribute"] == "race"].copy()

    auroc_mean, auroc_std, gap_mean, gap_std = [], [], [], []
    for cond in CONDITION_ORDER:
        a = overall.loc[overall["condition"] == cond, "AUROC"].to_numpy()
        g = race_gaps.loc[race_gaps["condition"] == cond, "auroc_gap"].to_numpy()
        auroc_mean.append(a.mean()); auroc_std.append(a.std(ddof=0))
        gap_mean.append(g.mean());   gap_std.append(g.std(ddof=0))
    return (np.array(auroc_mean), np.array(auroc_std),
            np.array(gap_mean),   np.array(gap_std))


def style_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#333333")
    ax.tick_params(colors="#333333")
    ax.yaxis.grid(True, linestyle="-", linewidth=0.5,
                  color="#cccccc", alpha=0.3)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)


def make_panel(ax, means, stds, ylabel, ylim, yticks, show_labels=True):
    x = np.arange(len(DISPLAY_NAMES))
    bars = ax.bar(x, means, yerr=stds, width=0.68,
                  color=colors_by_rank(means),
                  edgecolor="none",
                  error_kw=dict(ecolor="#333333",
                                elinewidth=0.8, capsize=3, capthick=0.8))
    ax.set_xticks(x)
    ax.set_xticklabels(DISPLAY_NAMES)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    style_axis(ax)

    if show_labels:
        span = ylim[1] - ylim[0]
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    m + s + span * 0.02,
                    f"{m:.3f}", ha="center", va="bottom",
                    fontsize=7, color="#333333")
    return bars


def main():
    auroc_m, auroc_s, gap_m, gap_s = load_data()

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))

    make_panel(axes[0], auroc_m, auroc_s,
               ylabel="AUROC",
               ylim=(0.78, 0.86),
               yticks=np.linspace(0.78, 0.86, 5))

    make_panel(axes[1], gap_m, gap_s,
               ylabel="Race AUROC gap",
               ylim=(0.0, 0.24),
               yticks=np.linspace(0.0, 0.24, 5))

    fig.tight_layout(pad=0.4, w_pad=1.2)

    base = OUT_DIR / "Figure5_graph_ablation"
    fig.savefig(base.with_suffix(".pdf"),
                bbox_inches="tight", pad_inches=0.02, dpi=300)
    fig.savefig(base.with_suffix(".svg"),
                bbox_inches="tight", pad_inches=0.02)
    fig.savefig(base.with_suffix(".png"),
                bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)

    print("AUROC   mean:", np.round(auroc_m, 4), "std:", np.round(auroc_s, 4))
    print("RaceGap mean:", np.round(gap_m, 4),   "std:", np.round(gap_s, 4))
    print(f"Wrote: {base}.{{pdf,svg,png}}")


if __name__ == "__main__":
    main()
