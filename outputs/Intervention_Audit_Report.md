# Intervention Audit Report

Date: 2026-04-11

## Scope

This audit checks the repo against the intervention checklist used to frame the fairness story:

- Fairness baseline outputs required for all comparisons
- Data-level and training-time interventions
- Post-hoc calibration and thresholding
- Tradeoff summary objects
- Graph-structure ablation as the mechanism test

The audit distinguishes three states:

- Implemented in the repo
- Surfaced in the current paper-facing `outputs/` package
- Present only in task-local folders and therefore unsafe to promise in Methods unless added to the manuscript or appendix

## Bottom Line

The repo is stronger than the current paper package suggests. The following families are implemented and evidenced in task folders but are not currently integrated into the `outputs/` manuscript package:

- Balanced sampling
- Variance regularization
- Per-group threshold equalization
- TPR-gap / equalized-opportunity reporting
- Fairness heatmap
- Subgroup bar plot
- AUROC-vs-gap tradeoff scatter
- Graph-structure ablation / Task 10 mechanism test

The current paper-facing package is internally coherent if it is framed narrowly around:

- Task 6 fairness baselines
- Task 8 reweighting
- Task 9 Platt and isotonic calibration
- Global-threshold comparisons
- Table 3 as the compact tradeoff table

## Highest-Risk Scope Mismatches

### 1. Balanced sampling exists, but only as task-local evidence

Balanced sampling is present in Task 7 for LSTM and TGTFull, but the repo does not expose a cross-seed paper table for it. The strongest visible artifact is [task7_intervention_comparison.csv](/Users/thomasbezza/Desktop/DDJA/Clinical-Framework-for-Graph-Transformers/Task7/results/final_tables/task7_intervention_comparison.csv), which is seed-42 tuning rather than the same level of paper-ready aggregation used for Task 8 and Task 9.

Recommendation: do not claim balanced sampling as a completed paper result unless you move it into the appendix with the seed-42 limitation stated plainly.

### 2. Variance regularization is implemented and partially selected, but omitted from the paper package

Task 8 clearly includes subgroup-loss variance regularization and even recommends it for TGTFull in [task8_conclusion_summary.txt](/Users/thomasbezza/Desktop/DDJA/Clinical-Framework-for-Graph-Transformers/Task8/results/notes/task8_conclusion_summary.txt). Cross-seed validation tables exist for both models, including [TGTFull_3seed_validation_summary.csv](/Users/thomasbezza/Desktop/DDJA/Clinical-Framework-for-Graph-Transformers/Task8/results/final_tables/TGTFull_3seed_validation_summary.csv).

Recommendation: either add a short appendix/table for variance-regularized runs or narrow Methods so the completed intervention suite no longer implies that variance regularization was reported.

### 3. Threshold equalization and TPR-gap reporting are fully implemented but not surfaced

Task 9 contains the full `Original/Platt/Isotonic x GlobalThresh/PerGroupThresh` grid plus explicit `tpr_gap` reporting in [T9_comparison_table.csv](/Users/thomasbezza/Desktop/DDJA/Clinical-Framework-for-Graph-Transformers/Task9/Task9Results/T9_comparison_table.csv). The current paper-facing outputs intentionally narrow Table 3 to `GlobalThresh`, and [Table3_Complete_Summary.md](/Users/thomasbezza/Desktop/DDJA/Clinical-Framework-for-Graph-Transformers/outputs/Table3_Complete_Summary.md) says `PerGroupThresh` is reserved for appendix if needed.

Recommendation: if the Methods text promises equalized-opportunity or subgroup-threshold evaluation, add appendix material now. Otherwise, narrow the Methods wording to say the reported intervention comparison uses the global-threshold slice of Task 9.

### 4. The figures pipeline is behind the task-local results

The repo already contains Task 6 and Task 9 plot artifacts, but `figures/main/` currently only contains Figures 1 and 2. [figures/INDEX.md](/Users/thomasbezza/Desktop/DDJA/Clinical-Framework-for-Graph-Transformers/figures/INDEX.md) still marks the fairness heatmap, intervention tradeoff plot, and graph ablation figure as "coming next."

Recommendation: either generate the missing paper-facing figures or stop implying that those figures are already part of the publication bundle.

### 5. Task 10 is complete but not integrated into the manuscript package

The mechanism story is real and well documented in [Task10_Findings.md](/Users/thomasbezza/Desktop/DDJA/Clinical-Framework-for-Graph-Transformers/Task10/Task10_Findings.md), with matching tables in [Task10_ablation_table.csv](/Users/thomasbezza/Desktop/DDJA/Clinical-Framework-for-Graph-Transformers/Task10/Task10_ablation_table.csv) and [Task10_ablation_table_extended.csv](/Users/thomasbezza/Desktop/DDJA/Clinical-Framework-for-Graph-Transformers/Task10/Task10_ablation_table_extended.csv). But the current `outputs/Analysis_Index.md` package does not reference Task 10.

Recommendation: either integrate Task 10 into the paper package as the mechanism test or avoid claiming it is already part of the finalized results build.

## What You Can Safely Claim Right Now

The current `outputs/` package safely supports these claims:

- Baseline subgroup fairness evaluation exists and is paper-ready through Table 2
- Reweighting is reported for LSTM and TGTFull through Table 3
- Platt and isotonic calibration are reported through Table 3
- The reported post-hoc comparison uses the GlobalThresh slice of Task 9
- A compact AUROC-versus-fairness tradeoff table exists in the paper package

## What Should Be Narrowed Unless You Add Appendix Material

- Balanced sampling
- Variance regularization
- Per-group threshold equalization
- TPR-gap / equalized-opportunity reporting
- Fairness heatmap and subgroup bar plot as paper figures
- AUROC-vs-gap tradeoff scatter as a paper figure
- Task 10 graph-ablation mechanism test

## Spreadsheet

The row-by-row checklist is in [Intervention_Audit_Checklist.csv](/Users/thomasbezza/Desktop/DDJA/Clinical-Framework-for-Graph-Transformers/outputs/Intervention_Audit_Checklist.csv).
