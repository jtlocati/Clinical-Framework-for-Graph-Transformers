# Table 4. Graph Structure Ablation

| Model Variant | AUROC | AUPRC | Sex gap (AUROC) | Age gap (AUROC) | Race gap (AUROC) |
|---|---|---|---|---|---|
| No-edge baseline | 0.8281 ± 0.0139 | 0.4731 ± 0.0264 | 0.0053 | 0.0792 | 0.1352 |
| Temporal-only graph | 0.8231 ± 0.0107 | 0.4532 ± 0.0323 | 0.0112 | 0.0768 | 0.1649 |
| Clinical-only graph | 0.8324 ± 0.0112 | 0.4818 ± 0.0347 | 0.0040 | 0.0788 | 0.1302 |
| Full graph | 0.8328 ± 0.0106 | 0.4825 ± 0.0344 | 0.0033 | 0.0729 | 0.1292 |

*Values are mean ± std across three training seeds (42, 123, 2024). Subgroup gaps are max − min across demographic groups per seed, then averaged; smaller gaps indicate smaller demographic performance disparities. All four conditions were trained on the same splits and evaluated through the same fairness evaluation pipeline as the subgroup-gap calculations, preserving within-row consistency.*
