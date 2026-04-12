# Appendix Table: Graph Structure Ablation (full metric set)

| Model Variant | AUROC | AUPRC | Sex gap AUROC | Age gap AUROC | Race gap AUROC | Sex gap AUPRC | Age gap AUPRC | Race gap AUPRC | Sex std(AUROC) | Age std(AUROC) | Race std(AUROC) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| No-edge baseline | 0.8281 ± 0.0139 | 0.4731 ± 0.0264 | 0.0053 | 0.0792 | 0.1352 | 0.0291 | 0.0761 | 0.2283 | 0.0027 | 0.0337 | 0.0446 |
| Temporal-only graph | 0.8231 ± 0.0107 | 0.4532 ± 0.0323 | 0.0112 | 0.0768 | 0.1649 | 0.0311 | 0.0777 | 0.2664 | 0.0056 | 0.0329 | 0.0540 |
| Clinical-only graph | 0.8324 ± 0.0112 | 0.4818 ± 0.0347 | 0.0040 | 0.0788 | 0.1302 | 0.0276 | 0.0933 | 0.2422 | 0.0020 | 0.0331 | 0.0409 |
| Full graph | 0.8328 ± 0.0106 | 0.4825 ± 0.0344 | 0.0033 | 0.0729 | 0.1292 | 0.0287 | 0.0886 | 0.2365 | 0.0016 | 0.0307 | 0.0408 |

*Mean ± std across seeds {42, 123, 2024}. Gap columns report max − min across groups within each attribute (sex, age, race), averaged across seeds. std(AUROC) columns report the per-seed standard deviation across groups, averaged across seeds — an alternative dispersion-based disparity measure. All values come from the same Task 6 fairness evaluation pipeline applied to the same test cohort (n=6,182, 848 positives).*
