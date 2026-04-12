# Table 3 Source Map: Fairness Interventions

## Overview
This document maps exactly where each row's metrics originate, ensuring one-to-one correspondence between overall performance and subgroup gap metrics.

## Model Selection
- LSTM (sequential baseline)
- TGT Temporal-only (temporal graph)
- TGT Clinical-only (clinical graph)
- TGT Full (full graph)

## Intervention Methods
Primary methods tested:
1. **Original**: Pre-intervention baseline
2. **Reweighting**: Race-weighted training (Task 8, LSTM and TGT Full only)
3. **Platt Scaling**: Post-hoc calibration (Task 9)
4. **Isotonic Scaling**: Post-hoc calibration (Task 9)

---

## Data Source Locations

### LSTM

**Original**
- Overall metrics: Task9/Task9Results/T9_LSTM_Original_GlobalThresh_metrics.csv (overall row)
- AUPRC: Same file (overall row)
- Subgroup gaps: Task9/Task9Results/T9_LSTM_Original_GlobalThresh_gaps.csv (all attributes)
- Note: Using GlobalThresh variant for clean comparison

**Reweighting**
- Overall metrics: Task8/results/cross_seed_summaries/LSTM_train_race_weighted_vs_original_overall_summary.csv (train_race_weighted row)
- Subgroup gaps: Task8/results/cross_seed_summaries/LSTM_train_race_weighted_vs_original_disparity_summary.csv (train_race_weighted rows)
- Note: Task 8 provides 3-seed aggregates

**Platt Scaling**
- Overall metrics: Task9/Task9Results/T9_LSTM_Platt_GlobalThresh_metrics.csv (overall row)
- Subgroup gaps: Task9/Task9Results/T9_LSTM_Platt_GlobalThresh_gaps.csv (all attributes)
- Note: Same evaluation pipeline as Original for direct comparison

**Isotonic Scaling**
- Overall metrics: Task9/Task9Results/T9_LSTM_Isotonic_GlobalThresh_metrics.csv (overall row)
- Subgroup gaps: Task9/Task9Results/T9_LSTM_Isotonic_GlobalThresh_gaps.csv (all attributes)
- Note: Same evaluation pipeline as Original for direct comparison

---

### TGT Temporal-only

**Original**
- Overall metrics: Task9/Task9Results/T9_tgt_temporal_only_Original_GlobalThresh_metrics.csv (overall row)
- Subgroup gaps: Task9/Task9Results/T9_tgt_temporal_only_Original_GlobalThresh_gaps.csv (all attributes)
- Note: Using GlobalThresh variant

**Platt Scaling**
- Overall metrics: Task9/Task9Results/T9_tgt_temporal_only_Platt_GlobalThresh_metrics.csv (overall row)
- Subgroup gaps: Task9/Task9Results/T9_tgt_temporal_only_Platt_GlobalThresh_gaps.csv (all attributes)

**Isotonic Scaling**
- Overall metrics: Task9/Task9Results/T9_tgt_temporal_only_Isotonic_GlobalThresh_metrics.csv (overall row)
- Subgroup gaps: Task9/Task9Results/T9_tgt_temporal_only_Isotonic_GlobalThresh_gaps.csv (all attributes)

**Reweighting**
- Not available in Task 8 results
- Table will show NA

---

### TGT Clinical-only

**Original**
- Overall metrics: Task9/Task9Results/T9_tgt_clinical_only_Original_GlobalThresh_metrics.csv (overall row)
- Subgroup gaps: Task9/Task9Results/T9_tgt_clinical_only_Original_GlobalThresh_gaps.csv (all attributes)

**Platt Scaling**
- Overall metrics: Task9/Task9Results/T9_tgt_clinical_only_Platt_GlobalThresh_metrics.csv (overall row)
- Subgroup gaps: Task9/Task9Results/T9_tgt_clinical_only_Platt_GlobalThresh_gaps.csv (all attributes)

**Isotonic Scaling**
- Overall metrics: Task9/Task9Results/T9_tgt_clinical_only_Isotonic_GlobalThresh_metrics.csv (overall row)
- Subgroup gaps: Task9/Task9Results/T9_tgt_clinical_only_Isotonic_GlobalThresh_gaps.csv (all attributes)

**Reweighting**
- Not available in Task 8 results
- Table will show NA

---

### TGT Full (tgt_full_graph)

**Original**
- Overall metrics: Task9/Task9Results/T9_tgt_full_graph_Original_GlobalThresh_metrics.csv (overall row)
- Subgroup gaps: Task9/Task9Results/T9_tgt_full_graph_Original_GlobalThresh_gaps.csv (all attributes)

**Reweighting**
- Overall metrics: Task8/results/cross_seed_summaries/TGTFull_train_race_weighted_vs_original_overall_summary.csv (train_race_weighted row)
- Subgroup gaps: Task8/results/cross_seed_summaries/TGTFull_train_race_weighted_vs_original_disparity_summary.csv (train_race_weighted rows)

**Platt Scaling**
- Overall metrics: Task9/Task9Results/T9_tgt_full_graph_Platt_GlobalThresh_metrics.csv (overall row)
- Subgroup gaps: Task9/Task9Results/T9_tgt_full_graph_Platt_GlobalThresh_gaps.csv (all attributes)

**Isotonic Scaling**
- Overall metrics: Task9/Task9Results/T9_tgt_full_graph_Isotonic_GlobalThresh_metrics.csv (overall row)
- Subgroup gaps: Task9/Task9Results/T9_tgt_full_graph_Isotonic_GlobalThresh_gaps.csv (all attributes)

---

## Extraction Notes

### Gap Definitions
All sources define gap as max-min across subgroups within an attribute:
- Sex gap = max(sex_F AUROC, sex_M AUROC) - min(sex_F AUROC, sex_M AUROC)
- Age gap = max(age_<40, age_40-65, age_65+) - min(...)
- Race gap = max(race_Asian, race_Black, race_Hispanic, race_White) - min(...)

AUPRC gaps computed similarly.

### Delta Calculations
For each model, compute deltas relative to that model's Original row:
- ΔAUROC vs Original = [Method] AUROC - [Original] AUROC
- ΔSex AUROC gap vs Original = [Method] Sex gap - [Original] Sex gap
  - Negative delta = improvement (smaller gap is better)

### Cross-Task Integration Notes
- Task 9 (Original, Platt, Isotonic) all evaluated on same test set with consistent methodology
- Task 8 (Reweighting) evaluated on Task 8's test set (same patients as Table 1, different threshold optimization)
- Slight numerical differences expected between Task 8 and Task 9 Original rows due to evaluation pipeline differences
- Will document these differences in Notes column if substantial

---

## Data Validation Checks

Before populating Table 3:
1. Verify all files exist and are readable
2. Confirm metrics extracted from correct rows (overall, not subgroup-specific)
3. Ensure gap columns use same max-min definition across all sources
4. Check that delta signs are consistent (negative = improvement for gaps)
5. Flag any missing intervention-model combinations as NA with reason

---

## File Schedule

Working table will be populated in this order:
1. All Original rows (LSTM, TGT Temporal, TGT Clinical, TGT Full)
2. Reweighting rows (LSTM, TGT Full only)
3. Platt rows (all models)
4. Isotonic rows (all models)

After each method block, compute delta columns relative to Original.
