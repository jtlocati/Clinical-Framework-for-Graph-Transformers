"""
Figure: Performance-fairness tradeoff for the graph-topology ablation.

x = race AUROC gap (smaller is fairer)
y = overall AUROC (larger is more accurate)
Top-left of the plot is the desirable region.

95% confidence regions are computed via *parametric* bootstrap using
Hanley-McNeil analytic standard errors for AUROC at the patient level
(per overall test set and per race subgroup). The Task 10 ablation does
not save raw predictions, so a fully non-parametric bootstrap is not
possible without re-running inference; the parametric variant captures
patient-level sampling noise using the closed-form AUROC variance and
combines it with seed-to-seed variability.

For each variant we compute, for B=10000 bootstrap iterations:
  - sample a seed s uniformly from {42, 123, 2024}
  - draw AUROC_overall ~ Normal(obs[s], HM_SE_overall[s])
  - for each race subgroup g (with sufficient samples) draw
    AUROC_g ~ Normal(obs[s,g], HM_SE_g[s])
  - record the max-min race-subgroup AUROC gap
Percentile 2.5%, 97.5% give the 95% CI for AUROC and for the race gap.
Layout: 95% CI ellipses (axis-aligned half-axes from the percentile
intervals) + a single marker per variant at the bootstrap median; labels
are placed at fixed offsets that do not overlap any ellipse or marker.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

REPO_ROOT = Path(__file__).resolve().parent.parent
TASK6_RESULTS = REPO_ROOT / "Task6" / "results"
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
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

MODELS = ["LSTM", "TGTTemporal", "TGTClinical", "TGTFull"]
SEEDS = [42, 123, 2024]
COLORS = {
    "LSTM":         "#1f3b5c",
    "TGTTemporal":  "#64728b",
    "TGTClinical":  "#be8b7e",
    "TGTFull":      "#a05a4a",
}
MARKERS = {
    "LSTM": "o", "TGTTemporal": "s",
    "TGTClinical": "^", "TGTFull": "D",
}
RACE_PREFIX = "race_"
N_BOOT = 10000
RNG_SEED = 12345


def hanley_mcneil_se(auc, n_pos, n_neg):
    if auc <= 0 or auc >= 1 or n_pos < 2 or n_neg < 2:
        return np.nan
    q1 = auc / (2 - auc)
    q2 = 2 * auc**2 / (1 + auc)
    var = (auc * (1 - auc)
           + (n_pos - 1) * (q1 - auc**2)
           + (n_neg - 1) * (q2 - auc**2)) / (n_pos * n_neg)
    return float(np.sqrt(max(var, 0.0)))


def load_per_seed_subgroup(model, seed):
    fp = TASK6_RESULTS / f"{model}_seed{seed}_subgroup_metrics.csv"
    df = pd.read_csv(fp)
    df = df[df["note"] == "ok"].copy()
    overall = df[df["group"] == "overall"].iloc[0]
    race = df[df["group"].str.startswith(RACE_PREFIX)].copy()
    return overall, race


def bootstrap_variant(model, rng):
    overall_auc_samples = np.empty(N_BOOT)
    gap_samples = np.empty(N_BOOT)

    seed_data = []
    for s in SEEDS:
        ov, race = load_per_seed_subgroup(model, s)
        ov_se = hanley_mcneil_se(ov["AUROC"], ov["positives"], ov["negatives"])
        race_aucs = race["AUROC"].to_numpy()
        race_ses = np.array([
            hanley_mcneil_se(a, p, n) for a, p, n in zip(
                race["AUROC"], race["positives"], race["negatives"])
        ])
        seed_data.append({
            "ov_auc": float(ov["AUROC"]),
            "ov_se": ov_se,
            "race_aucs": race_aucs,
            "race_ses": race_ses,
        })

    seed_choices = rng.integers(0, len(SEEDS), size=N_BOOT)
    for b in range(N_BOOT):
        sd = seed_data[seed_choices[b]]
        a_overall = rng.normal(sd["ov_auc"], sd["ov_se"])
        a_race = rng.normal(sd["race_aucs"], sd["race_ses"])
        overall_auc_samples[b] = a_overall
        gap_samples[b] = a_race.max() - a_race.min()

    return overall_auc_samples, gap_samples


def main():
    rng = np.random.default_rng(RNG_SEED)

    summary = {}
    for m in MODELS:
        auc_b, gap_b = bootstrap_variant(m, rng)
        summary[m] = {
            "auc_med": np.median(auc_b),
            "auc_lo":  np.percentile(auc_b, 2.5),
            "auc_hi":  np.percentile(auc_b, 97.5),
            "gap_med": np.median(gap_b),
            "gap_lo":  np.percentile(gap_b, 2.5),
            "gap_hi":  np.percentile(gap_b, 97.5),
        }

    fig, ax = plt.subplots(figsize=(4.4, 3.4))

    for m in MODELS:
        s = summary[m]
        cx, cy = s["gap_med"], s["auc_med"]
        w = s["gap_hi"] - s["gap_lo"]
        h = s["auc_hi"] - s["auc_lo"]
        ell = Ellipse((cx, cy), width=w, height=h,
                      facecolor="none", edgecolor=COLORS[m],
                      lw=1.0, alpha=0.55, zorder=1,
                      linestyle="-")
        ax.add_patch(ell)
        ax.scatter([cx], [cy], s=70, marker=MARKERS[m],
                   facecolor=COLORS[m], edgecolor="white",
                   linewidth=0.9, zorder=3)

    label_offsets_pt = {
        "TGTClinical": (-22,  18, "right", "bottom"),
        "TGTFull":     ( 22,  10, "left",  "bottom"),
        "LSTM":        (-22, -16, "right", "top"),
        "TGTTemporal": ( 22, -10, "left",  "top"),
    }
    for m in MODELS:
        s = summary[m]
        dx_pt, dy_pt, ha, va = label_offsets_pt[m]
        ax.annotate(m, xy=(s["gap_med"], s["auc_med"]),
                    xytext=(dx_pt, dy_pt), textcoords="offset points",
                    fontsize=7.8, color="#222222", ha=ha, va=va,
                    zorder=5,
                    arrowprops=dict(arrowstyle="-", color=COLORS[m],
                                    lw=0.5, alpha=0.7,
                                    shrinkA=2, shrinkB=2))

    all_gap_lo = min(s["gap_lo"] for s in summary.values())
    all_gap_hi = max(s["gap_hi"] for s in summary.values())
    all_auc_lo = min(s["auc_lo"] for s in summary.values())
    all_auc_hi = max(s["auc_hi"] for s in summary.values())
    pad_x = (all_gap_hi - all_gap_lo) * 0.10
    pad_y = (all_auc_hi - all_auc_lo) * 0.18
    ax.set_xlim(all_gap_lo - pad_x, all_gap_hi + pad_x)
    ax.set_ylim(all_auc_lo - pad_y, all_auc_hi + pad_y)

    ax.set_xlabel(r"Race AUROC gap   ($\leftarrow$ fairer)")
    ax.set_ylabel(r"AUROC   ($\uparrow$ more accurate)")

    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#333333")
    ax.tick_params(colors="#333333")
    ax.grid(True, linestyle="-", linewidth=0.5,
            color="#cccccc", alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout(pad=0.4)

    base = OUT_DIR / "Figure6_tradeoff"
    fig.savefig(base.with_suffix(".pdf"),
                bbox_inches="tight", pad_inches=0.02, dpi=300)
    fig.savefig(base.with_suffix(".svg"),
                bbox_inches="tight", pad_inches=0.02)
    fig.savefig(base.with_suffix(".png"),
                bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)

    print(f"{'Model':<14} {'AUROC [2.5,97.5]':<28} {'RaceGap [2.5,97.5]'}")
    for m in MODELS:
        s = summary[m]
        print(f"{m:<14} "
              f"{s['auc_med']:.4f} [{s['auc_lo']:.4f}, {s['auc_hi']:.4f}]   "
              f"{s['gap_med']:.4f} [{s['gap_lo']:.4f}, {s['gap_hi']:.4f}]")
    print(f"Wrote: {base}.{{pdf,svg,png}}")


if __name__ == "__main__":
    main()
