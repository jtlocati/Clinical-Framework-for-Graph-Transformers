"""
Radar / Spider chart: AUROC, AUPRC, Race Gap, Sex Gap, Age Gap
across four graph-ablation models (LSTM, TGTTemporal, TGTClinical, TGTFull).

Gaps are inverted so bigger = better. Values labeled on a single spoke going up.

Usage:
    python plot_radar.py

Outputs:
    radar_chart.pdf / .png / .svg
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# ── Data from Tables 2, 4 ───────────────────────────────────────────────
data = {
    "LSTM": {
        "AUROC": 0.8281,
        "AUPRC": 0.4731,
        "Race Gap": 0.1352,
        "Sex Gap": 0.0053,
        "Age Gap": 0.0791,
    },
    "TGTTemporal": {
        "AUROC": 0.8231,
        "AUPRC": 0.4532,
        "Race Gap": 0.1649,
        "Sex Gap": 0.0112,
        "Age Gap": 0.0768,
    },
    "TGTClinical": {
        "AUROC": 0.8324,
        "AUPRC": 0.4818,
        "Race Gap": 0.1302,
        "Sex Gap": 0.0039,
        "Age Gap": 0.0787,
    },
    "TGTFull": {
        "AUROC": 0.8328,
        "AUPRC": 0.4825,
        "Race Gap": 0.1291,
        "Sex Gap": 0.0033,
        "Age Gap": 0.0729,
    },
}

metrics = ["AUROC", "AUPRC", "Race Gap", "Sex Gap", "Age Gap"]
model_names = list(data.keys())
gap_metrics = {"Race Gap", "Sex Gap", "Age Gap"}

# ── Per-axis normalization ───────────────────────────────────────────────
axis_info = {}
for m in metrics:
    raw_vals = [data[mod][m] for mod in model_names]
    if m in gap_metrics:
        plot_vals = [1 - v for v in raw_vals]
    else:
        plot_vals = raw_vals[:]

    lo, hi = min(plot_vals), max(plot_vals)
    spread = hi - lo if hi != lo else 0.01
    pad_lo = lo - spread * 0.3
    pad_hi = hi + spread * 0.3

    axis_info[m] = {"lo": pad_lo, "hi": pad_hi}

normed = {}
for m in metrics:
    info = axis_info[m]
    raw_vals = [data[mod][m] for mod in model_names]
    if m in gap_metrics:
        plot_vals = [1 - v for v in raw_vals]
    else:
        plot_vals = raw_vals[:]
    normed[m] = [(v - info["lo"]) / (info["hi"] - info["lo"]) for v in plot_vals]

# ── Radar setup ──────────────────────────────────────────────────────────
N = len(metrics)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

axis_labels = ["AUROC", "AUPRC", "Race\nFairness", "Sex\nFairness", "Age\nFairness"]

# Grid rings
grid_rings = [0.2, 0.4, 0.6, 0.8, 1.0]

# ── Styling ──────────────────────────────────────────────────────────────
styles = {
    "LSTM":         {"color": "#4a6a8a", "linestyle": "-",  "marker": "o"},
    "TGTTemporal":  {"color": "#7b8fa8", "linestyle": "--", "marker": "s"},
    "TGTClinical":  {"color": "#c4876f", "linestyle": "-.", "marker": "D"},
    "TGTFull":      {"color": "#a05a4a", "linestyle": "-",  "marker": "D"},
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 14,
    "font.weight": "bold",
    "figure.dpi": 150,
})

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

# ── Draw each model ──────────────────────────────────────────────────────
for mod in model_names:
    values = [normed[m][model_names.index(mod)] for m in metrics]
    values += values[:1]
    s = styles[mod]

    ax.plot(angles, values,
            color=s["color"], linewidth=2.5, linestyle=s["linestyle"],
            marker=s["marker"], markersize=10, markeredgecolor="white",
            markeredgewidth=1.5, label=mod, zorder=4)
    ax.fill(angles, values, color=s["color"], alpha=0.06, zorder=2)

# ── Label grid lines on the first spoke only (going straight up) ─────────
# The first spoke is at angle 0 (top of chart = "AUROC")
spoke_idx = 0
m = metrics[spoke_idx]
info = axis_info[m]
theta = angles[spoke_idx]

for ring in grid_rings:
    # Convert ring back to real value
    plot_val = info["lo"] + ring * (info["hi"] - info["lo"])
    real_val = plot_val  # AUROC is not inverted

    ax.text(
        theta - 0.12, ring, f"{real_val:.3f}",
        fontsize=10, color="#555",
        ha="right", va="center",
        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
        zorder=6,
    )

# ── Axis formatting ──────────────────────────────────────────────────────
ax.set_xticks(angles[:-1])
ax.set_xticklabels(axis_labels, fontsize=15, fontweight="bold")

ax.set_ylim(0, 1.05)
ax.set_yticks(grid_rings)
ax.set_yticklabels([""] * len(grid_rings))

ax.grid(True, linewidth=0.8, color="#cccccc", alpha=0.8)
ax.spines["polar"].set_linewidth(1.0)

# Legend
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1),
          fontsize=13, frameon=True, framealpha=0.9,
          edgecolor="#ddd", fancybox=True)


# ── Save ─────────────────────────────────────────────────────────────────
for ext in ["pdf", "png", "svg"]:
    out = SCRIPT_DIR / f"radar_chart.{ext}"
    fig.savefig(out, dpi=600 if ext == "png" else None, bbox_inches="tight")
    print(f"Saved: {out}")

plt.show()