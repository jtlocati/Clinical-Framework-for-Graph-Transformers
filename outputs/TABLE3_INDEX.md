# Table 3: Contents and Usage Guide

## Overview
Table 3 evaluates fairness interventions across LSTM and TGT variants, measuring effects on both predictive performance (AUROC, AUPRC) and subgroup disparities. The table answers: **Can simple fairness interventions improve the fairness–performance tradeoff?**

**Key Finding**: Post-hoc isotonic scaling consistently improves both performance (+0.011–0.014 AUROC) and fairness, outperforming training-time reweighting which trades off one demographic for another.

---

## Files and Their Purpose

### 🎯 Paper Integration (Use These)

**`Table3_Fairness_Interventions.md` (5.3 KB)**
- **Purpose**: Ready-to-insert main table with results paragraph
- **Contents**: 
  - Narrow table (7 columns) for main paper
  - Full supplementary table (10 columns with AUPRC gaps)
  - Complete results paragraph
- **Action**: Copy table and paragraph directly into Results section

**`Table3_Fairness_Interventions.csv` (1.2 KB)**
- **Purpose**: Data supplement for published paper
- **Contents**: 15 rows (14 data + header), 10 columns
- **Action**: Include in supplementary materials

**`Table3_Fairness_Interventions_Narrow.csv` (869 B)**
- **Purpose**: Narrow version if space constrained
- **Contents**: 7 columns (drops AUPRC gaps)
- **Action**: Use if journal space limits table width

---

### 🔬 Analysis and Documentation (Supporting)

**`Table3_Analysis_Notes.md` (7.0 KB)**
- **Purpose**: Detailed per-model intervention assessment
- **Contents**:
  - Analysis of each intervention for each model
  - Explicit tradeoff quantification
  - Cross-model patterns identified
  - Validation checklist
- **Action**: Reference in paper or appendix for detailed fairness analysis

**`Table3_Validation_Report.md` (9.8 KB)**
- **Purpose**: Formal validation documentation
- **Contents**:
  - Three-point validation (completeness, consistency, claim integrity)
  - Source file verification with explicit paths
  - Spot-checks on gap calculations
  - Evidence-based claim review
- **Action**: Provide to reviewers if they request reproducibility details

**`Table3_Source_Map.md` (6.1 KB)**
- **Purpose**: Data provenance and extraction discipline
- **Contents**:
  - Exact file locations for each metric
  - Explanation of data consistency enforcement
  - Notes on missing interventions (why certain combinations are NA)
- **Action**: Reference if detailed data sourcing explanation needed

**`Table3_Complete_Summary.md` (9.2 KB)**
- **Purpose**: High-level overview of entire Table 3 project
- **Contents**:
  - Goal, scope, and methodology
  - Key findings summary
  - Connection to Tables 1–2
  - File usage guide
- **Action**: Read first for complete context; reference as needed

---

### ⚙️ Working Files (For Analysis)

**`Table3_Fairness_Interventions_Working.csv` (1.8 KB)**
- **Purpose**: Complete working version with delta columns
- **Contents**: 
  - 15 rows (14 data + header)
  - 16 columns including deltas: ΔAUROC vs Original, ΔAUPRC vs Original, ΔGaps vs Original
  - Enables instant visual identification of tradeoffs
- **Action**: Use for further analysis or if additional statistics needed

---

## Key Results at a Glance

### Isotonic Scaling (Consistent Winner)
- **LSTM**: +0.0143 AUROC, improves sex & race gaps
- **TGT Temporal**: +0.0114 AUROC, maintains age parity perfectly
- **TGT Clinical**: +0.0143 AUROC, improves sex & race gaps
- **TGT Full**: +0.0140 AUROC, balanced improvements

### Reweighting (Extreme Tradeoffs)
- **LSTM**: Helps race (gap −0.0416) but worsens age (gap +0.0093)
- **TGT Full**: Dramatically improves race (gap −0.0908, 40% reduction) but worsens age & sex

### Platt Scaling (Conservative)
- Modest AUROC gains across models (~+0.005)
- Minimal fairness impact (gaps typically ±0.002)

---

## Table Structure

### Columns in Main Paper Version
1. **Model**: LSTM, TGT Temporal-only, TGT Clinical-only, TGT Full
2. **Method**: Original, Reweighting (if available), Platt, Isotonic
3. **Overall AUROC**: Test-set AUROC (range 0.817–0.838)
4. **Overall AUPRC**: Test-set AUPRC (range 0.431–0.490)
5. **Sex AUROC gap**: Max-min across sex subgroups (range 0.002–0.013)
6. **Age AUROC gap**: Max-min across age subgroups (range 0.064–0.088)
7. **Race AUROC gap**: Max-min across race subgroups (range 0.135–0.226)

(Plus AUPRC gaps in supplementary version)

### Notes on Missingness
- **LSTM**: 4 rows (Original, Reweighting, Platt, Isotonic) — all present
- **TGT Temporal**: 3 rows (no Reweighting available in Task 7 pipeline)
- **TGT Clinical**: 3 rows (no Reweighting available in Task 7 pipeline)
- **TGT Full**: 4 rows — all present

Total: **14 data rows** plus header

---

## How to Use These Files

### Scenario 1: Preparing Paper Submission
1. Open `Table3_Fairness_Interventions.md`
2. Copy narrow table (7 columns, 14 rows) → paste into Results section
3. Copy results paragraph → insert after table
4. Include CSV files in supplementary materials
5. Done! Ready for submission

### Scenario 2: Responding to Reviewer Questions
1. **"Is fairness truly improved?"** → Reference `Table3_Analysis_Notes.md` for per-model breakdown
2. **"How were the data sourced?"** → Provide `Table3_Source_Map.md` and `Table3_Validation_Report.md`
3. **"Why are some interventions missing?"** → See `Table3_Complete_Summary.md` "Methodological Notes" section

### Scenario 3: Extended Appendix
1. Include full table (10 columns) from `Table3_Fairness_Interventions.md`
2. Add `Table3_Analysis_Notes.md` as detailed fairness analysis section
3. Optional: Add working table with delta columns for methodological transparency

### Scenario 4: Further Analysis
1. Open `Table3_Fairness_Interventions_Working.csv` in spreadsheet program
2. Delta columns show immediate fairness–performance tradeoffs
3. Use for plots, additional statistical tests, or extended discussion

---

## Caption for Paper

**Table 3. Effect of fairness interventions on overall performance and subgroup disparities for sequential and graph-based comparison.**

Table 3 reports the effect of fairness interventions on overall predictive performance (AUROC, AUPRC) and subgroup disparities for sequential (LSTM) and graph-based (TGT Temporal-only, TGT Clinical-only, TGT Full) models. Gap is the max–min difference across subgroups within the corresponding demographic attribute; smaller values indicate smaller disparities. The table measures fairness–performance tradeoffs. Four intervention methods were evaluated: Original (baseline), Reweighting (race-weighted training; Task 8, LSTM and TGT Full only), Platt scaling (post-hoc calibration; Task 9), and Isotonic scaling (post-hoc calibration; Task 9). Each row integrates overall metrics and subgroup gaps from the same evaluation pipeline for consistency.

---

## Citation/Reference

When referencing Table 3 data extraction and validation:
- **Data sources**: Task 8 (reweighting) and Task 9 (post-hoc calibration) outputs
- **Validation**: Three-point check (completeness, consistency, claim integrity); all passed
- **Reproducibility**: Full source map and validation report on file

---

## Document Version and Status

- **Date Completed**: April 11, 2026
- **Status**: ✓ Ready for paper submission
- **Validation**: ✓ All three checks passed (completeness, consistency, claim integrity)
- **Quality**: High (includes working version with deltas, analysis notes, and formal validation)

---

## Quick Access to Key Information

| Need | File |
|------|------|
| Insert table in paper | `Table3_Fairness_Interventions.md` |
| Supplementary data | `Table3_Fairness_Interventions.csv` |
| Detailed fairness analysis | `Table3_Analysis_Notes.md` |
| Data provenance | `Table3_Source_Map.md` |
| Validation evidence | `Table3_Validation_Report.md` |
| Complete overview | `Table3_Complete_Summary.md` |
| Working/delta data | `Table3_Fairness_Interventions_Working.csv` |
| Space-constrained fit | `Table3_Fairness_Interventions_Narrow.csv` |

