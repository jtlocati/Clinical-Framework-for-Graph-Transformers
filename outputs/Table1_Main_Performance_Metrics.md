# Table 1. Main predictive performance on ICU mortality

| Model | AUROC mean | AUROC std | AUPRC mean | AUPRC std | Best seed | Notes |
|-------|-----------|-----------|-----------|-----------|-----------|-------|
| Logistic Regression | 0.8060 | 0.0094 | 0.4412 | 0.0132 | | Baseline shallow model |
| LightGBM | 0.8370 | 0.0085 | 0.4947 | 0.0205 | | Best overall baseline performance |
| LSTM | 0.8284 | 0.0115 | 0.4712 | 0.0241 | 123 | Standard seq2seq baseline |
| Standard Transformer | 0.8317 | 0.0113 | 0.4846 | 0.0287 | 123 | Best transformer baseline |
| ChronoFormer | 0.8306 | 0.0069 | 0.4799 | 0.0186 | 123 | Time-series specialized design |
| PatchTST | 0.8225 | 0.0076 | 0.4631 | 0.0203 | 123 | Patch-based time-series |
| Pretrained TST | 0.8307 | 0.0102 | 0.4804 | 0.0263 | 123 | Pre-trained transformer |
| TGT Temporal-only | 0.8306 | 0.0147 | 0.4682 | 0.0339 | 123 | Temporal graph only |
| TGT Clinical-only | 0.8319 | 0.0114 | 0.4736 | 0.0308 | 123 | Clinical feature relationships |
| TGT Full | 0.8300 | 0.0139 | 0.4708 | 0.0361 | 123 | Temporal + clinical graphs |
| TGT Full | 0.8300 | 0.0139 | 0.4708 | 0.0361 | 123 | Temporal + clinical graphs |