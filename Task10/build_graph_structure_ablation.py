"""
Graph Structure Ablation — paper deliverables (Table 4, appendix table, Figure X, prose).

Scope: four matched variants — no-edge (LSTM), temporal-only (TGTTemporal),
clinical-only (TGTClinical), full graph (TGTFull) — under identical seeds {42,123,2024},
identical splits, identical evaluation pipeline (Task6 subgroup_metrics / disparity_summary).

This script is a strict-superset of build_ablation_table.py:
 - Pulls ALL overall metrics from Task6 subgroup_metrics (not Task5) so that overall
   AUROC/AUPRC and subgroup gaps are computed on the same evaluation pipeline for all
   four variants. This closes the within-row consistency issue the paper table caption
   now references explicitly.
 - Reports mean ± std across the three seeds (matches Table 1 format).
 - Computes BOTH max-min gap AND std-across-groups for sex/age/race (AUROC + AUPRC).
 - Emits deltas vs the no-edge baseline.
 - Writes compact main-paper table, extended appendix table, and two-panel figure.

Outputs written to Task10/:
  Task10_GSA_main.csv / Task10_GSA_main.md         — compact main-paper Table 4
  Task10_GSA_appendix.csv / Task10_GSA_appendix.md — full appendix ablation table
  Task10_GSA_deltas.csv                            — deltas vs no-edge baseline
  Task10_GSA_perseed.csv                           — raw per-seed values for audit
  Task10_GSA_figure.pdf / .png / .svg              — Figure X (two-panel)
  Task10_GSA_subsection.md                         — the paper paragraph + captions
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Run from repo root
REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)

SEEDS = [42, 123, 2024]
# (model_key_in_files, display_label, ablation_condition)
VARIANTS = [
    ("LSTM",        "No-edge baseline",        "no_edge"),
    ("TGTTemporal", "Temporal-only graph",     "temporal_only"),
    ("TGTClinical", "Clinical-only graph",     "clinical_only"),
    ("TGTFull",     "Full graph",              "full_graph"),
]
ATTRIBUTES = ["sex", "age", "race"]

PALETTE = {
    "no_edge":       "#64728b",  # blue_med  — non-graph baseline
    "temporal_only": "#a9afbd",  # blue_light — partial graph
    "clinical_only": "#be8b7e",  # warm_med  — partial graph
    "full_graph":    "#a05a4a",  # warm_dark — full graph
}

# ---------- Part 1: load per-seed overall metrics from the fairness pipeline ----------
def load_overall_perseed():
    rows = []
    for model, label, cond in VARIANTS:
        for seed in SEEDS:
            fp = f"Task6/results/{model}_seed{seed}_subgroup_metrics.csv"
            df = pd.read_csv(fp)
            ov = df[df["group"] == "overall"]
            assert len(ov) == 1, f"{fp}: expected exactly one 'overall' row, got {len(ov)}"
            assert int(ov.iloc[0]["count"]) == 6182, \
                f"{fp}: overall n={ov.iloc[0]['count']} (expected 6182; cohort mismatch)"
            assert int(ov.iloc[0]["positives"]) == 848, \
                f"{fp}: overall pos={ov.iloc[0]['positives']} (expected 848; cohort mismatch)"
            rows.append({
                "condition": cond,
                "model": model,
                "label": label,
                "seed": seed,
                "AUROC": float(ov.iloc[0]["AUROC"]),
                "AUPRC": float(ov.iloc[0]["AUPRC"]),
            })
    return pd.DataFrame(rows)

# ---------- Part 2: subgroup gap measures (max-min AND std-across-groups) ----------
def load_gaps_perseed():
    """Read disparity_summary for max-min gaps AND recompute std-across-groups
    from subgroup_metrics (disparity_summary already has AUROC_std / AUPRC_std but
    we reproduce it to confirm within-row consistency)."""
    rows = []
    for model, label, cond in VARIANTS:
        for seed in SEEDS:
            dsp = pd.read_csv(f"Task6/results/{model}_seed{seed}_disparity_summary.csv")
            sgm = pd.read_csv(f"Task6/results/{model}_seed{seed}_subgroup_metrics.csv")
            for attr in ATTRIBUTES:
                dsp_row = dsp[dsp["attribute"] == attr]
                assert len(dsp_row) == 1, f"{model} seed{seed} {attr}: missing disparity row"
                auroc_gap = float(dsp_row.iloc[0]["AUROC_gap"])
                auprc_gap = float(dsp_row.iloc[0]["AUPRC_gap"])
                auroc_std_groups = float(dsp_row.iloc[0]["AUROC_std"])
                auprc_std_groups = float(dsp_row.iloc[0]["AUPRC_std"])
                # Sanity-check std-across-groups against raw subgroup file
                mask = sgm["group"].str.startswith(attr + "_")
                sub = sgm[mask].dropna(subset=["AUROC"])
                recomputed = float(sub["AUROC"].std(ddof=0))
                assert abs(recomputed - auroc_std_groups) < 1e-6, \
                    f"{model} seed{seed} {attr}: AUROC_std mismatch ({recomputed} vs {auroc_std_groups})"
                rows.append({
                    "condition": cond, "model": model, "label": label,
                    "seed": seed, "attribute": attr,
                    "auroc_gap": auroc_gap, "auprc_gap": auprc_gap,
                    "auroc_std_groups": auroc_std_groups,
                    "auprc_std_groups": auprc_std_groups,
                })
    return pd.DataFrame(rows)

overall_ps = load_overall_perseed()
gaps_ps = load_gaps_perseed()

# Sanity: exactly 4 conditions × 3 seeds = 12 overall rows, and 12 × 3 attrs = 36 gap rows
assert len(overall_ps) == 12, f"expected 12 overall rows, got {len(overall_ps)}"
assert len(gaps_ps) == 36, f"expected 36 gap rows, got {len(gaps_ps)}"

# ---------- Part 3: aggregate to mean ± std (across 3 seeds) ----------
def agg_overall(df):
    g = df.groupby(["condition", "model", "label"], sort=False).agg(
        AUROC_mean=("AUROC", "mean"), AUROC_std=("AUROC", "std"),
        AUPRC_mean=("AUPRC", "mean"), AUPRC_std=("AUPRC", "std"),
    ).reset_index()
    return g

def agg_gaps(df):
    g = df.groupby(["condition", "model", "label", "attribute"], sort=False).agg(
        auroc_gap_mean=("auroc_gap", "mean"), auroc_gap_std=("auroc_gap", "std"),
        auprc_gap_mean=("auprc_gap", "mean"), auprc_gap_std=("auprc_gap", "std"),
        auroc_std_groups_mean=("auroc_std_groups", "mean"),
        auprc_std_groups_mean=("auprc_std_groups", "mean"),
    ).reset_index()
    return g

overall_agg = agg_overall(overall_ps)
gaps_agg = agg_gaps(gaps_ps)

# Pivot gaps wide so one row per condition
def pivot_gaps(gaps_agg, value, prefix):
    p = gaps_agg.pivot(index=["condition", "model", "label"],
                       columns="attribute", values=value).reset_index()
    p.columns.name = None
    p = p.rename(columns={a: f"{prefix}_{a}" for a in ATTRIBUTES})
    return p

gaps_wide = pivot_gaps(gaps_agg, "auroc_gap_mean", "gap_auroc")
for col_src, col_dst in [
    ("auprc_gap_mean",      "gap_auprc"),
    ("auroc_std_groups_mean", "std_auroc"),
    ("auprc_std_groups_mean", "std_auprc"),
]:
    gaps_wide = gaps_wide.merge(
        pivot_gaps(gaps_agg, col_src, col_dst),
        on=["condition", "model", "label"], how="left",
    )

result = overall_agg.merge(gaps_wide, on=["condition", "model", "label"], how="left")

# Preserve ablation order
order = [v[2] for v in VARIANTS]
result = result.set_index("condition").loc[order].reset_index()

# ---------- Part 4: deltas vs no-edge baseline ----------
base = result.iloc[0]  # no_edge row
delta_cols = {
    "dAUROC":           "AUROC_mean",
    "dAUPRC":           "AUPRC_mean",
    "dGap_AUROC_sex":   "gap_auroc_sex",
    "dGap_AUROC_age":   "gap_auroc_age",
    "dGap_AUROC_race":  "gap_auroc_race",
    "dGap_AUPRC_sex":   "gap_auprc_sex",
    "dGap_AUPRC_age":   "gap_auprc_age",
    "dGap_AUPRC_race":  "gap_auprc_race",
    "dStd_AUROC_sex":   "std_auroc_sex",
    "dStd_AUROC_age":   "std_auroc_age",
    "dStd_AUROC_race":  "std_auroc_race",
}
deltas = result[["condition", "label"]].copy()
for new, src in delta_cols.items():
    deltas[new] = result[src] - base[src]

# ---------- Part 5: quality assertions ----------
assert result["AUROC_mean"].notna().all()
assert result["AUPRC_mean"].notna().all()
for c in [c for c in result.columns if c.startswith("gap_") or c.startswith("std_")]:
    assert (result[c] >= 0).all(), f"negative value in {c}"
# Also check the no-edge baseline deltas are exactly zero
for col in delta_cols:
    assert abs(deltas.iloc[0][col]) < 1e-12, f"no_edge baseline delta {col} must be 0"

# ---------- Part 6: write CSVs ----------
os.makedirs("Task10", exist_ok=True)
overall_ps.to_csv("Task10/Task10_GSA_perseed_overall.csv", index=False)
gaps_ps.to_csv("Task10/Task10_GSA_perseed_gaps.csv", index=False)
result.to_csv("Task10/Task10_GSA_appendix.csv", index=False)
deltas.to_csv("Task10/Task10_GSA_deltas.csv", index=False)

main_cols = ["label", "AUROC_mean", "AUROC_std", "AUPRC_mean", "AUPRC_std",
             "gap_auroc_sex", "gap_auroc_age", "gap_auroc_race"]
main_df = result[main_cols].copy()
main_df.to_csv("Task10/Task10_GSA_main.csv", index=False)

# ---------- Part 7: formatted markdown tables ----------
def fmt_mpm(m, s, digits=4):
    return f"{m:.{digits}f} ± {s:.{digits}f}"

def fmt_gap(x, digits=4):
    return f"{x:.{digits}f}"

# Main paper Table 4 — compact
main_md_lines = [
    "# Table 4. Graph Structure Ablation",
    "",
    "| Model Variant | AUROC | AUPRC | Sex gap (AUROC) | Age gap (AUROC) | Race gap (AUROC) |",
    "|---|---|---|---|---|---|",
]
for _, r in result.iterrows():
    main_md_lines.append(
        f"| {r['label']} "
        f"| {fmt_mpm(r['AUROC_mean'], r['AUROC_std'])} "
        f"| {fmt_mpm(r['AUPRC_mean'], r['AUPRC_std'])} "
        f"| {fmt_gap(r['gap_auroc_sex'])} "
        f"| {fmt_gap(r['gap_auroc_age'])} "
        f"| {fmt_gap(r['gap_auroc_race'])} |"
    )
main_md_lines += [
    "",
    "*Values are mean ± std across three training seeds (42, 123, 2024). "
    "Subgroup gaps are max − min across demographic groups per seed, then averaged; "
    "smaller gaps indicate smaller demographic performance disparities. "
    "All four conditions were trained on the same splits and evaluated through the same "
    "fairness evaluation pipeline as the subgroup-gap calculations, preserving "
    "within-row consistency.*",
]
Path("Task10/Task10_GSA_main.md").write_text("\n".join(main_md_lines) + "\n")

# Appendix table — full metric set
app_md_lines = [
    "# Appendix Table: Graph Structure Ablation (full metric set)",
    "",
    "| Model Variant | AUROC | AUPRC | "
    "Sex gap AUROC | Age gap AUROC | Race gap AUROC | "
    "Sex gap AUPRC | Age gap AUPRC | Race gap AUPRC | "
    "Sex std(AUROC) | Age std(AUROC) | Race std(AUROC) |",
    "|---|---|---|---|---|---|---|---|---|---|---|---|",
]
for _, r in result.iterrows():
    app_md_lines.append(
        f"| {r['label']} "
        f"| {fmt_mpm(r['AUROC_mean'], r['AUROC_std'])} "
        f"| {fmt_mpm(r['AUPRC_mean'], r['AUPRC_std'])} "
        f"| {fmt_gap(r['gap_auroc_sex'])} "
        f"| {fmt_gap(r['gap_auroc_age'])} "
        f"| {fmt_gap(r['gap_auroc_race'])} "
        f"| {fmt_gap(r['gap_auprc_sex'])} "
        f"| {fmt_gap(r['gap_auprc_age'])} "
        f"| {fmt_gap(r['gap_auprc_race'])} "
        f"| {fmt_gap(r['std_auroc_sex'])} "
        f"| {fmt_gap(r['std_auroc_age'])} "
        f"| {fmt_gap(r['std_auroc_race'])} |"
    )
app_md_lines += [
    "",
    "*Mean ± std across seeds {42, 123, 2024}. Gap columns report max − min across "
    "groups within each attribute (sex, age, race), averaged across seeds. "
    "std(AUROC) columns report the per-seed standard deviation across groups, averaged "
    "across seeds — an alternative dispersion-based disparity measure. All values come "
    "from the same Task 6 fairness evaluation pipeline applied to the same test cohort "
    "(n=6,182, 848 positives).*",
]
Path("Task10/Task10_GSA_appendix.md").write_text("\n".join(app_md_lines) + "\n")

# Deltas as a readable block for the writer
delta_md_lines = [
    "# Deltas vs No-Edge Baseline",
    "",
    "| Variant | ΔAUROC | ΔAUPRC | ΔSex gap | ΔAge gap | ΔRace gap |",
    "|---|---|---|---|---|---|",
]
for _, r in deltas.iloc[1:].iterrows():  # skip the baseline row
    delta_md_lines.append(
        f"| {r['label']} "
        f"| {r['dAUROC']:+.4f} "
        f"| {r['dAUPRC']:+.4f} "
        f"| {r['dGap_AUROC_sex']:+.4f} "
        f"| {r['dGap_AUROC_age']:+.4f} "
        f"| {r['dGap_AUROC_race']:+.4f} |"
    )
Path("Task10/Task10_GSA_deltas.md").write_text("\n".join(delta_md_lines) + "\n")

# ---------- Part 8: figure — two-panel, overall AUROC + race AUROC gap ----------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
})

labels = result["label"].tolist()
conds = result["condition"].tolist()
colors = [PALETTE[c] for c in conds]

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)

# Panel A: overall AUROC with seed-std error bars
ax = axes[0]
x = np.arange(len(labels))
ax.bar(x, result["AUROC_mean"], yerr=result["AUROC_std"], color=colors,
       edgecolor="#1f3b5c", linewidth=0.8, capsize=3)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20, ha="right")
ax.set_ylabel("Overall AUROC")
ax.set_ylim(0.78, 0.87)
ax.set_title("A. Overall discriminative performance", loc="left", fontsize=10, fontweight="bold")
for xi, m in zip(x, result["AUROC_mean"]):
    ax.text(xi, m + 0.002, f"{m:.3f}", ha="center", va="bottom", fontsize=8)

# Panel B: race AUROC gap with seed-std error bars
ax = axes[1]
ax.bar(x, result["gap_auroc_race"],
       yerr=gaps_agg[gaps_agg["attribute"] == "race"].set_index("condition").loc[conds, "auroc_gap_std"].values,
       color=colors, edgecolor="#1f3b5c", linewidth=0.8, capsize=3)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20, ha="right")
ax.set_ylabel("Race AUROC gap (max − min)")
ax.set_ylim(0, max(result["gap_auroc_race"]) * 1.35)
ax.set_title("B. Race subgroup disparity", loc="left", fontsize=10, fontweight="bold")
for xi, m in zip(x, result["gap_auroc_race"]):
    ax.text(xi, m + 0.005, f"{m:.3f}", ha="center", va="bottom", fontsize=8)

for out in ["Task10_GSA_figure.pdf", "Task10_GSA_figure.png", "Task10_GSA_figure.svg"]:
    fig.savefig(f"Task10/{out}", dpi=600 if out.endswith(".png") else None, bbox_inches="tight")
plt.close(fig)

# ---------- Part 9: paper paragraph (with live numbers) ----------
r_no    = result.iloc[0]
r_temp  = result.iloc[1]
r_clin  = result.iloc[2]
r_full  = result.iloc[3]

# Identify the best variants on each axis
best_auroc_idx = result["AUROC_mean"].idxmax()
best_auprc_idx = result["AUPRC_mean"].idxmax()
best_sex_idx   = result["gap_auroc_sex"].idxmin()
best_age_idx   = result["gap_auroc_age"].idxmin()
best_race_idx  = result["gap_auroc_race"].idxmin()

best_auroc = result.iloc[best_auroc_idx]
best_race  = result.iloc[best_race_idx]

d_temp = deltas.iloc[1]
d_clin = deltas.iloc[2]
d_full = deltas.iloc[3]

para = f"""# Graph Structure Ablation — paper subsection

## Caption — Table 4
**Table 4. Graph structure ablation on ICU mortality prediction.** We compare a
no-edge baseline, a temporal-only graph, a clinical-only graph, and a full graph under
identical data splits, seeds, and training settings. Values are mean ± std over three
training seeds (42, 123, 2024). Smaller subgroup-gap values indicate smaller demographic
performance disparities. Overall AUROC/AUPRC are computed from the same fairness
evaluation pipeline used to derive the subgroup-gap calculations, preserving within-row
consistency.

## Caption — Figure X
**Figure X. Effect of graph topology on overall performance and subgroup disparity.**
Left: overall AUROC across the no-edge, temporal-only, clinical-only, and full-graph
conditions. Right: corresponding subgroup disparity measured as the race AUROC gap
(max − min across race subgroups). Error bars show standard deviation across three
training seeds. This ablation isolates the effect of graph structure by holding data
splits, seeds, and training settings fixed.

## Subsection prose

### Graph Structure Ablation

To isolate the contribution of graph topology, we compared a no-edge baseline with
temporal-only, clinical-only, and full-graph variants under identical training and
evaluation settings. The **{best_auroc['label'].lower()}** achieved the highest overall
predictive performance, with an AUROC of {best_auroc['AUROC_mean']:.4f} and an AUPRC of
{best_auroc['AUPRC_mean']:.4f}, whereas the **{best_race['label'].lower()}** yielded the
smallest race subgroup gap ({best_race['gap_auroc_race']:.4f} AUROC). Moving from the
no-edge baseline to the clinical-only graph changed AUROC by {d_clin['dAUROC']:+.4f} and
the race AUROC gap by {d_clin['dGap_AUROC_race']:+.4f}, while the temporal-only graph
changed AUROC by {d_temp['dAUROC']:+.4f} and the race AUROC gap by
{d_temp['dGap_AUROC_race']:+.4f} — the same dataset, splits, and seeds produced opposite
fairness movements depending on which edges were encoded. Across the four conditions,
the effect of graph structure was therefore not uniform: temporal edges were associated
with the largest race disparity ({r_temp['gap_auroc_race']:.4f}), whereas clinical and
full-graph variants were associated with the smallest race disparities
({r_clin['gap_auroc_race']:.4f} and {r_full['gap_auroc_race']:.4f}, respectively) and
with small AUROC changes within seed-level noise ({d_full['dAUROC']:+.4f} for the full
graph relative to no-edge). This suggests that the fairness impact of graph modelling
depends on which relationships are encoded rather than on the presence of a graph
per se, and that within this design space clinical edges — not temporal ones — are
associated with reduced subgroup disparity under matched training conditions.
"""
Path("Task10/Task10_GSA_subsection.md").write_text(para)

# ---------- Part 10: console report ----------
print("\nGraph Structure Ablation — build report")
print("=" * 72)
print("Matched conditions verified: 4 variants × 3 seeds × n=6182 test cohort")
print("Evaluation pipeline: Task6 subgroup_metrics / disparity_summary (same for all)")
print()
print("Main table (mean ± std across 3 seeds):")
pd.set_option("display.float_format", "{:.4f}".format)
print(main_df.to_string(index=False))
print()
print("Deltas vs no-edge baseline:")
print(deltas.to_string(index=False))
print()
print("Files written to Task10/:")
for f in [
    "Task10_GSA_main.csv", "Task10_GSA_main.md",
    "Task10_GSA_appendix.csv", "Task10_GSA_appendix.md",
    "Task10_GSA_deltas.csv", "Task10_GSA_deltas.md",
    "Task10_GSA_perseed_overall.csv", "Task10_GSA_perseed_gaps.csv",
    "Task10_GSA_figure.pdf", "Task10_GSA_figure.png", "Task10_GSA_figure.svg",
    "Task10_GSA_subsection.md",
]:
    print(f"  {f}")
