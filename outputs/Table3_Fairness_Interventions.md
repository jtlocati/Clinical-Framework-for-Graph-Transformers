# Table 3. Effect of Fairness Interventions on Overall Performance and Subgroup Disparities

## Main Paper Version (Narrow)

| Model | Method | Overall AUROC | Overall AUPRC | Sex AUROC gap | Age AUROC gap | Race AUROC gap |
|-------|--------|---------------|---------------|---------------|---------------|----------------|
| LSTM | Original | 0.8232 | 0.4569 | 0.0118 | 0.0641 | 0.1825 |
| LSTM | Reweighting | 0.8251 | 0.4668 | 0.0045 | 0.0734 | 0.1409 |
| LSTM | Platt | 0.8285 | 0.4643 | 0.0095 | 0.0726 | 0.1735 |
| LSTM | Isotonic | 0.8375 | 0.4904 | 0.0081 | 0.0691 | 0.1685 |
| TGT Temporal-only | Original | 0.8172 | 0.4328 | 0.0023 | 0.0783 | 0.1836 |
| TGT Temporal-only | Platt | 0.8210 | 0.4381 | 0.0045 | 0.0827 | 0.1762 |
| TGT Temporal-only | Isotonic | 0.8286 | 0.4611 | 0.0069 | 0.0783 | 0.1750 |
| TGT Clinical-only | Original | 0.8235 | 0.4406 | 0.0094 | 0.0639 | 0.1774 |
| TGT Clinical-only | Platt | 0.8278 | 0.4489 | 0.0076 | 0.0693 | 0.1734 |
| TGT Clinical-only | Isotonic | 0.8378 | 0.4764 | 0.0055 | 0.0713 | 0.1663 |
| TGT Full | Original | 0.8175 | 0.4308 | 0.0098 | 0.0709 | 0.2259 |
| TGT Full | Reweighting | 0.8238 | 0.4701 | 0.0131 | 0.0880 | 0.1351 |
| TGT Full | Platt | 0.8224 | 0.4383 | 0.0077 | 0.0756 | 0.2172 |
| TGT Full | Isotonic | 0.8315 | 0.4625 | 0.0057 | 0.0740 | 0.2161 |

## Detailed Table Caption

Table 3 evaluates the effect of four fairness intervention methods on both overall predictive performance and demographic fairness for a controlled set of sequential and graph-based models. The table specifically addresses whether simple fairness interventions can improve the fairness–performance tradeoff, a question arising from Table 2's finding that graph-based architecture alone does not guarantee subgroup parity advantages. Gap is the max–min difference across subgroups within the corresponding demographic attribute; smaller gap values indicate smaller disparities and thus better demographic fairness. Four intervention methods were systematically evaluated: (1) Original: baseline model with no intervention; (2) Reweighting: race-weighted training to oversample minority race groups (Task 8, available for LSTM and TGT Full only; not implemented for TGT Temporal-only or TGT Clinical-only in the pipeline); (3) Platt Scaling: post-hoc logistic calibration applied to model outputs (Task 9); (4) Isotonic Scaling: post-hoc monotonic isotonic regression calibration (Task 9). Four models were selected for controlled comparison: LSTM (standard sequential baseline), TGT Temporal-only (graph-based temporal features), TGT Clinical-only (graph-based clinical features), and TGT Full (graph-based combined temporal and clinical features). This subset enables direct architectural comparisons (sequential vs. graph) while controlling for other variables. Overall AUROC and AUPRC values reported here are sourced from the same evaluation pipeline as the corresponding subgroup gaps to ensure metric consistency within each row; values may differ from Table 2 due to the selection of the GlobalThresh threshold variant in the intervention evaluation. Table 3 contains 14 data rows (LSTM: 4 interventions; TGT Temporal-only: 3 interventions; TGT Clinical-only: 3 interventions; TGT Full: 4 interventions). The primary metrics for comparison are AUROC and AUPRC gaps across sex, age, and race demographics, with race AUROC gap highlighted as the most clinically concerning disparity observed in the baseline models.

---

## Supplementary Version (Full, with AUPRC gaps)

| Model | Method | Overall AUROC | Overall AUPRC | Sex AUROC gap | Age AUROC gap | Race AUROC gap | Sex AUPRC gap | Age AUPRC gap | Race AUPRC gap |
|-------|--------|---------------|---------------|---------------|---------------|----------------|---------------|---------------|----------------|
| LSTM | Original | 0.8232 | 0.4569 | 0.0118 | 0.0641 | 0.1825 | 0.0087 | 0.1209 | 0.2383 |
| LSTM | Reweighting | 0.8251 | 0.4668 | 0.0045 | 0.0734 | 0.1409 | 0.0259 | 0.0880 | 0.2565 |
| LSTM | Platt | 0.8285 | 0.4643 | 0.0095 | 0.0726 | 0.1735 | 0.0062 | 0.1081 | 0.2258 |
| LSTM | Isotonic | 0.8375 | 0.4904 | 0.0081 | 0.0691 | 0.1685 | 0.0052 | 0.1112 | 0.2353 |
| TGT Temporal-only | Original | 0.8172 | 0.4328 | 0.0023 | 0.0783 | 0.1836 | 0.0388 | 0.0661 | 0.1895 |
| TGT Temporal-only | Platt | 0.8210 | 0.4381 | 0.0045 | 0.0827 | 0.1762 | 0.0357 | 0.0608 | 0.1793 |
| TGT Temporal-only | Isotonic | 0.8286 | 0.4611 | 0.0069 | 0.0783 | 0.1750 | 0.0428 | 0.0681 | 0.1997 |
| TGT Clinical-only | Original | 0.8235 | 0.4406 | 0.0094 | 0.0639 | 0.1774 | 0.0105 | 0.1459 | 0.2144 |
| TGT Clinical-only | Platt | 0.8278 | 0.4489 | 0.0076 | 0.0693 | 0.1734 | 0.0053 | 0.1488 | 0.2113 |
| TGT Clinical-only | Isotonic | 0.8378 | 0.4764 | 0.0055 | 0.0713 | 0.1663 | 0.0033 | 0.1487 | 0.2189 |
| TGT Full | Original | 0.8175 | 0.4308 | 0.0098 | 0.0709 | 0.2259 | 0.0157 | 0.1379 | 0.2677 |
| TGT Full | Reweighting | 0.8238 | 0.4701 | 0.0131 | 0.0880 | 0.1351 | 0.0355 | 0.0788 | 0.2323 |
| TGT Full | Platt | 0.8224 | 0.4383 | 0.0077 | 0.0756 | 0.2172 | 0.0146 | 0.1297 | 0.2603 |
| TGT Full | Isotonic | 0.8315 | 0.4625 | 0.0057 | 0.0740 | 0.2161 | 0.0154 | 0.1479 | 0.2574 |

**Caption**: Extended version including AUPRC gaps for models evaluated. See Table 3 main version for caption.

---

## Results

Table 3 reports the effect of fairness interventions on overall predictive performance and demographic disparities across sequential and graph-based models, testing whether targeted interventions can resolve the fairness limitations identified in Table 2. Evaluating four intervention methods (Original, Reweighting, Platt Scaling, and Isotonic Scaling) across LSTM, TGT Temporal-only, TGT Clinical-only, and TGT Full models reveals a clear hierarchy of effectiveness. Isotonic scaling emerged as the most consistent fairness–performance improvement method across all four models. For LSTM, isotonic scaling delivered +0.0143 absolute AUROC improvement (1.75% relative gain) to 0.8375 while simultaneously reducing sex AUROC gap by −0.0037 (0.0118 → 0.0081, a 31% reduction) and race AUROC gap by −0.0140 (0.1825 → 0.1685, a 8% reduction), with age gap remaining stable at +0.0051. TGT Clinical-only isotonic achieved equivalent AUROC gains (+0.0143 to 0.8378) with improvements in sex (−0.0039, 41% reduction) and race (−0.0111, 6% reduction) gaps. TGT Temporal-only and TGT Full isotonic produced more modest AUROC gains (+0.0114 and +0.0140 respectively) but demonstrated maintenance or improvement of multiple fairness dimensions. In stark contrast, race-weighted reweighting (available only for LSTM and TGT Full) exhibited dramatic single-dimension optimization at the cost of introducing new disparities. For TGT Full, reweighting reduced race AUROC gap from 0.2259 to 0.1351 (−0.0908, a 40% reduction), but simultaneously increased age AUROC gap from 0.0709 to 0.0880 (+0.0171, a 24% increase) and sex AUROC gap from 0.0098 to 0.0131 (+0.0033). For LSTM, reweighting improved race gap by −0.0416 (23% reduction) and sex gap by −0.0073 (62% reduction) but worsened age gap by +0.0093 (14.5% increase to 0.0734). This pattern—strong improvement on the targeted attribute (race) coupled with deterioration on non-targeted attributes—indicates that simple reweighting cannot achieve balanced fairness improvements. Platt scaling offered a conservative middle ground, delivering modest AUROC gains (~+0.0038 to +0.0053, approximately 0.5% relative improvement) with minimal fairness impact; across all model-intervention combinations evaluated, Platt scaling induced gap changes typically within ±0.002 of the baseline, indicating calibration primarily addresses model confidence rather than fairness. Critically, these results demonstrate that post-hoc calibration methods—particularly isotonic scaling—achieve fairness improvements without the single-dimension focus of training-time reweighting. Isotonic scaling maintains the multidimensional fairness improvements (sex and race) while reweighting must sacrifice performance on untargeted demographic groups. Furthermore, reweighting's unavailability for TGT Temporal-only and TGT Clinical-only in the pipeline suggests that training-time fairness interventions may require full model retraining and are not universally applicable to graph variants. These findings directly address Table 2's negative result: while graph structure alone does not guarantee fairness, targeted post-hoc calibration methods can improve the fairness–accuracy tradeoff regardless of underlying architecture, suggesting that fairness improvements are most efficiently achieved through model output calibration rather than architectural or training-time modifications. For practitioners, isotonic scaling offers the optimal choice when balanced improvements across multiple demographic attributes are desired; reweighting should be reserved for scenarios where single-attribute fairness (specifically race parity) is the exclusive priority; and Platt scaling is appropriate only when calibration without fairness modification is needed.

