# COMPREHENSIVE METRIC INVENTORY
## All Results/Values Found in the Repository

---

## TASK 4: BASELINE BENCHMARKS
**File:** `Task4/results/benchmark_results.csv` (Test set, aggregated across 3 seeds: 42, 123, 2024)

| Model | AUROC_mean | AUROC_std | AUPRC_mean | AUPRC_std |
|-------|-----------|-----------|-----------|-----------|
| Logistic Regression | 0.8059619021678893 | 0.009392441825670816 | 0.44124885951716514 | 0.013238745614679555 |
| LightGBM | 0.8369796935760389 | 0.008522633115775044 | 0.49472458077339515 | 0.020491226820355504 |
| LSTM | 0.8283656465111672 | 0.011545387299864611 | 0.4712054587702066 | 0.024071486874358203 |
| Standard Transformer | 0.8316680049427784 | 0.01126954850018405 | 0.4845687731447055 | 0.028661424393605864 |
| ChronoFormer | 0.830628998615739 | 0.0068728993386908835 | 0.47988198012994876 | 0.018563225192124295 |
| PatchTST | 0.8225232164375681 | 0.007648535893749928 | 0.4631206157013546 | 0.020319274563927972 |
| Pretrained TST | 0.8307406444477459 | 0.010164027692019897 | 0.48043859621553 | 0.0262892928332006 |

**File:** `Task4/results/benchmark_results_full.json` (Seed-level results - val and test splits)

**Seed 42:**
- Logistic Regression: Test AUROC=0.8058, Test AUPRC=0.4275
- LightGBM: Test AUROC=0.8312, Test AUPRC=0.4675
- LSTM: Test AUROC=0.8246, Test AUPRC=0.4398
- Standard Transformer: Test AUROC=0.8222, Test AUPRC=0.4583
- ChronoFormer: Test AUROC=0.8308, Test AUPRC=0.4693
- PatchTST: Test AUROC=0.8172, Test AUPRC=0.4381
- Pretrained TST: Test AUROC=0.8295, Test AUPRC=0.4677

**Seed 123:**
- Logistic Regression: Test AUROC=0.8175, Test AUPRC=0.4591
- LightGBM: Test AUROC=0.8490, Test AUPRC=0.5169
- LSTM: Test AUROC=0.8440, Test AUPRC=0.4982
- Standard Transformer: Test AUROC=0.8383, Test AUPRC=0.4950
- ChronoFormer: Test AUROC=0.8375, Test AUPRC=0.4946
- PatchTST: Test AUROC=0.8289, Test AUPRC=0.4689
- Pretrained TST: Test AUROC=0.8397, Test AUPRC=0.4924

**Seed 2024:**
- Logistic Regression: Test AUROC=0.7945, Test AUPRC=0.4372
- LightGBM: Test AUROC=0.8308, Test AUPRC=0.4998
- LSTM: Test AUROC=0.8165, Test AUPRC=0.4757
- Standard Transformer: Test AUROC=0.8300, Test AUPRC=0.4747
- ChronoFormer: Test AUROC=0.8291, Test AUPRC=0.4812
- PatchTST: Test AUROC=0.8206, Test AUPRC=0.4597
- Pretrained TST: Test AUROC=0.8258, Test AUPRC=0.4844

---

## TASK 5: TGT VARIANTS TRAINING STUDY
**File:** `Task5/results/T5_summary.csv` (Aggregated across 3 seeds: 42, 123, 2024)

| Variant | AUROC_mean | AUROC_std | AUPRC_mean | AUPRC_std |
|---------|-----------|-----------|-----------|-----------|
| Temporal-only | 0.8305666666666666 | 0.014693649421887455 | 0.4681666666666667 | 0.033908750099839 |
| Clinical-only | 0.8319333333333333 | 0.011397514348897873 | 0.47359999999999997 | 0.030768652879188566 |
| Full graph | 0.8299666666666666 | 0.013865544826415334 | 0.47079999999999994 | 0.03613986718293247 |

**File:** `Task5/results/T5_all_runs.csv` (Individual seed-level results)
- Contains detailed per-seed results for all three TGT variants
- Seeds: 42, 123, 2024

---

## TASK 6: FAIRNESS ANALYSIS
**File:** `Task6/results_T6/T6_fairness_table.csv` (Overall + Subgroup metrics for all models)

**Overall Performance (aggregated subgroups):**
- LSTM: AUROC=0.8208, AUPRC=0.4523
- Standard Transformer: AUROC=0.8252, AUPRC=0.4547
- ChronoFormer: AUROC=0.8251, AUPRC=0.4615
- PatchTST: AUROC=0.8209, AUPRC=0.4456
- TGT Temporal-only: AUROC=0.8172, AUPRC=0.4328
- TGT Clinical-only: AUROC=0.8235, AUPRC=0.4406
- TGT Full: AUROC=0.8175, AUPRC=0.4308

**Fairness Gap Metrics:**
- Sex gap (largest - smallest): ranges from 0.0023 (TGT Temporal) to 0.0192 (ChronoFormer)
- Age gap: ranges from 0.0554 (Standard Transformer) to 0.0795 (ChronoFormer)
- Race gap: ranges from 0.1761 (Standard Transformer) to 0.2259 (TGT Full)

**File:** `Task6/results_T6/T6_subgroup_metrics.csv` (Detailed per-subgroup metrics)
- Sex subgroups (F, M): AUROC and AUPRC per model
- Age subgroups (<40, 40-65, 65+): AUROC and AUPRC per model
- Race subgroups (Asian, Black, Hispanic, Multiple, Other, Unknown, White): AUROC and AUPRC per model

**File:** `Task6/results_T6/T6_intervention_StandardTransformer_platt_recalibration.csv`
**File:** `Task6/results_T6/T6_intervention_StandardTransformer_threshold_tuning.csv`
**File:** `Task6/results_T6/T6_intervention_tgt_clinical_only_platt_recalibration.csv`
**File:** `Task6/results_T6/T6_intervention_tgt_clinical_only_threshold_tuning.csv`
- Contains calibration and threshold tuning results for selected models

**File:** `Task6/results_T6/T6_summary.json`
- Summary statistics in JSON format

---

## TASK 7: FAIRNESS INTERVENTIONS & CROSS-SEED ANALYSIS
**File:** `Task7/results/final_tables/task7_intervention_comparison.csv` (Test performance with different interventions)

**LSTM (Seed 42):**
- Original: AUROC=0.8248, AUPRC=0.4462
- Race Balanced: AUROC=0.8163, AUPRC=0.4170
- Race Weighted Cap 10: AUROC=0.8171, AUPRC=0.4367
- Race Weighted Cap 5: AUROC=0.8188, AUPRC=0.4453

**TGT Full (Seed 42):**
- Original: AUROC=0.8236, AUPRC=0.4444
- Race Balanced: AUROC=0.8059, AUPRC=0.4165
- Race Weighted Cap 10: AUROC=0.8143, AUPRC=0.4330
- Race Weighted Cap 5: AUROC=0.8108, AUPRC=0.4377
- Race Weighted Cap 15: AUROC=0.8155, AUPRC=0.4332

**Cross-seed summaries available for:**
- LSTM race_weighted_cap5
- TGT Full race_weighted_cap10
- Overall task7 cross-seed summary

---

## TASK 8: VALIDATION METRICS
**File:** `Task8/results/` (Similar structure to Task7)
- Cross-seed summaries
- Final tables with metrics
- Raw predictions
- Per-seed metrics

---

## TASK 9: CALIBRATION & FAIRNESS THRESHOLDING
**Location:** `Task9/Task9Results/`

**Models with Complete Results (3 calibration methods × 2 threshold strategies):**

### Original (Uncalibrated) - Global Threshold
- LSTM: AUROC=0.8232, AUPRC=0.4569
- Standard Transformer: AUROC=0.8270, AUPRC=0.4513
- ChronoFormer: AUROC=0.8210, AUPRC=0.4475
- PatchTST: AUROC=0.8142, AUPRC=0.4330
- TGT Temporal-only: AUROC=0.8172, AUPRC=0.4328
- TGT Clinical-only: AUROC=0.8235, AUPRC=0.4406
- TGT Full: AUROC=0.8175, AUPRC=0.4308

### Original - Per-Group Threshold
(Separate results per demographic subgroup available in individual CSV files)

### Platt Calibration - Global Threshold
(Separate results available)

### Platt Calibration - Per-Group Threshold
(Separate results available)

### Isotonic Calibration - Global Threshold
(Separate results available)

### Isotonic Calibration - Per-Group Threshold
(Separate results available)

**File:** `Task9/Task9Results/T9_comparison_table.csv`
- Summary table comparing all calibration/threshold strategies

**File:** `Task9/Task9Results/T9_summary.json`
- Summary statistics

---

## FIGURES DATA
**File:** `figures/data/main_results.csv` (Per-seed test results)

**TGT Models:**
- TGT-Temporal Seed 42: AUROC=0.8172, AUPRC=0.4328
- TGT-Temporal Seed 123: AUROC=0.8463, AUPRC=0.5004
- TGT-Temporal Seed 2024: AUROC=0.8282, AUPRC=0.4713
- TGT-Clinical Seed 42: AUROC=0.8235, AUPRC=0.4406
- TGT-Clinical Seed 123: AUROC=0.8449, AUPRC=0.5015
- TGT-Clinical Seed 2024: AUROC=0.8274, AUPRC=0.4787
- TGT-Full Seed 42: AUROC=0.8175, AUPRC=0.4308
- TGT-Full Seed 123: AUROC=0.8449, AUPRC=0.5011
- TGT-Full Seed 2024: AUROC=0.8275, AUPRC=0.4805

**LSTM:**
- Seed 42: AUROC=0.8247784, AUPRC=0.4461834
- Seed 123: AUROC=0.8432939, AUPRC=0.4990392
- Seed 2024: AUROC=0.8161735, AUPRC=0.4739736

**File:** `figures/data/subgroup_results.csv`
(Subgroup-level results for different demographic attributes)

**File:** `figures/data/intervention_results.csv`
(Intervention-specific results)

---

## OUTPUTS FOLDER
**File:** `outputs/mimic_demo_subgroup_auc.csv`
**File:** `outputs/mimic_demo_subgroup_distribution.csv`
- Demo/validation results

**File:** `outputs/Table1_Main_Performance_Metrics.csv` (Created summary table)
- Populated with Task9 Original/Global metrics

---

## SUMMARY STATISTICS

### Total Data Points:
- **Baseline models (Task4):** 7 models × 4 metrics = 28 aggregated values + 7 seeds × 7 models × 2 metrics = 98 seed-level values
- **TGT variants (Task5):** 3 variants × 4 metrics = 12 aggregated values + seed-level data
- **Fairness analysis (Task6):** 7 models × (overall + 11 subgroups) × 2 metrics = 182 values
- **Fairness interventions (Task7):** Multiple model/intervention combinations
- **Calibration study (Task9):** 7 models × 3 calibrations × 2 thresholding strategies × multiple subgroups
- **Overall performance (figures/data):** 3 TGT variants × 3 seeds × 2 metrics + LSTM × 3 seeds × 2 metrics

### Models Included:
1. Logistic Regression (Task4 only)
2. LightGBM (Task4 only)
3. LSTM (Tasks 4, 5, 6, 7, 8, 9)
4. Standard Transformer (Tasks 4, 6, 7, 9)
5. ChronoFormer (Tasks 4, 6, 9)
6. PatchTST (Tasks 4, 6, 9)
7. Pretrained TST (Task4 only)
8. TGT Temporal-only (Tasks 5, 6, 7, 8, 9)
9. TGT Clinical-only (Tasks 5, 6, 7, 8, 9)
10. TGT Full (Tasks 5, 6, 7, 8, 9)

### Metrics Reported:
- AUROC (Area Under ROC Curve) - discrimination ability
- AUPRC (Area Under Precision-Recall Curve) - positive class prediction precision
- AUROC_std / AUPRC_std - variability across seeds
- Fairness gaps (sex, age, race)
- Calibration metrics (Platt, Isotonic)
- Threshold-specific metrics
- Subgroup-specific performance
- Intervention effects on fairness

---

## KEY FINDINGS AT A GLANCE

**Best Overall Baseline (Task4 aggregated):**
- LightGBM: AUROC=0.837, AUPRC=0.495

**Best Deep Learning Baseline (Task4 aggregated):**
- Standard Transformer: AUROC=0.832, AUPRC=0.485

**Best TGT Variant (Task5 aggregated):**
- Clinical-only: AUROC=0.832, AUPRC=0.474

**Fairness Performance (Task6):**
- Smallest race gap: Standard Transformer (0.176)
- Smallest sex gap: TGT Temporal-only (0.0023)
- Smallest age gap: Standard Transformer (0.0554)

**Cross-seed Variability:**
- Most stable: ChronoFormer (AUROC_std=0.0069)
- Most variable: Standard Transformer (AUPRC_std=0.0287)
