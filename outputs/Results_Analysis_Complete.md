# Results Analysis: Complete Findings and Statistical Methodology

## Main Performance Findings

### Baseline and Transformer Model Performance

We evaluated seven baseline models on ICU mortality prediction, comprising classical machine learning approaches (Logistic Regression, LightGBM, LSTM) and modern transformer-based architectures (Standard Transformer, ChronoFormer, PatchTST, Pretrained TST). All models were trained across three seeds (42, 123, 2024) and evaluated on a test set of 6,182 patients.

The LightGBM model achieved the best overall baseline performance with AUROC of 0.8370 ± 0.0085 and AUPRC of 0.4947 ± 0.0205, demonstrating that gradient-boosting approaches remain highly competitive for clinical prediction tasks. Among deep learning models, the Standard Transformer achieved the best performance with AUROC of 0.8317 ± 0.0113 and AUPRC of 0.4846 ± 0.0287, followed closely by Pretrained TST (AUROC: 0.8307 ± 0.0102) and ChronoFormer (AUROC: 0.8306 ± 0.0069).

The LSTM baseline achieved AUROC of 0.8284 ± 0.0115 and AUPRC of 0.4712 ± 0.0241, performing competitively with transformer models. PatchTST, a specialized time-series transformer design, showed lower performance (AUROC: 0.8225 ± 0.0076), suggesting that domain-specific time-series architectures do not necessarily improve over general transformer designs in clinical prediction. Logistic Regression, as the simplest baseline, achieved AUROC of 0.8060 ± 0.0094.

### Proposed TGT Variants Performance

The three proposed Temporal Graph Transformer (TGT) variants demonstrated comparable performance to the best transformer baselines:

TGT Clinical-only achieved the highest performance among TGT variants with AUROC of 0.8319 ± 0.0114 and AUPRC of 0.4736 ± 0.0308. This variant models clinical feature relationships, achieving performance that exceeds the standard transformer baseline and matches the best-in-class Pretrained TST model.

TGT Temporal-only (AUROC: 0.8306 ± 0.0147, AUPRC: 0.4682 ± 0.0339) models temporal progressions of features and achieves competitive AUROC performance while showing higher variance across seeds (std: 0.0147).

TGT Full (AUROC: 0.8300 ± 0.0139, AUPRC: 0.4708 ± 0.0361) combines both temporal and clinical graphs. Integrating both information sources does not improve performance over the clinical-only variant, suggesting potential information redundancy when jointly modeling both relationship types.

### Key Observations

**Competitive but Essentially Equivalent Performance**

A paired seed-wise analysis across all three seeds reveals small differences between TGT Clinical-only and transformer baselines:

- Standard Transformer: ΔAUROC = +0.0003 ± 0.0024
- Pretrained TST: ΔAUROC = +0.0012 ± 0.0011
- ChronoFormer: ΔAUROC = +0.0013 ± 0.0030
- LSTM: ΔAUROC = +0.0036 ± 0.0064

These differences are less than 0.4% and well within the noise of model training variation. The differences are so small that they are not practically or statistically significant. The wins by TGT Clinical-only (2/3 seeds against most baselines) reflect stochastic variation rather than meaningful superiority.

On AUPRC, Standard Transformer maintains an advantage (ΔAUPRC = -0.0110 ± 0.0043), suggesting better precision-recall balance. While TGT Clinical-only matches transformers on discrimination (AUROC), it sacrifices some confidence on the positive class.

**Clear Dominance Over Weaker Transformers**

TGT Clinical-only shows more convincing advantages over PatchTST with ΔAUROC = +0.0094 ± 0.0031 (wins 3/3 seeds), suggesting graph-based approaches outperform patch-based time-series designs.

**Shallow Learning vs. Deep Learning Trade-off**

LightGBM's superior performance (AUROC: 0.8370) compared to all deep learning methods demonstrates that explicit feature interactions captured through gradient boosting remain more effective. The ~0.5% gap between LightGBM and TGT Clinical-only is consistent and larger than intra-baseline variation, suggesting a fundamental advantage to gradient-boosted approaches.

**Information Redundancy in Graph Fusion**

The TGT variants reveal an important pattern. TGT Clinical-only achieves AUROC 0.8319, while TGT Temporal-only achieves 0.8306 and TGT Full achieves 0.8300. Adding temporal information to clinical relationships decreases performance. This suggests temporal and clinical relationships encode overlapping information, joint optimization becomes more difficult, or temporal information introduces noise.

All models show reasonable stability across seeds, with standard deviations ranging from 0.0068 (ChronoFormer) to 0.0287 (Standard Transformer AUPRC). TGT variants show higher variability, particularly TGT Full (AUROC std: 0.0139).

## Statistical Methodology: Paired Seed-wise Analysis

### Step 1: Per-Seed Paired Differences

We computed ΔAUROC and ΔAUPRC for TGT Clinical-only minus each baseline across seeds 42, 123, and 2024:

| Baseline | ΔAUROC (mean ± std) | ΔAUPRC (mean ± std) | TGT Wins |
|----------|-------------------|-------------------|----------|
| Standard Transformer | +0.0003 ± 0.0024 | -0.0110 ± 0.0043 | 2/3 |
| ChronoFormer | +0.0013 ± 0.0030 | -0.0063 ± 0.0082 | 2/3 |
| LSTM | +0.0036 ± 0.0064 | +0.0024 ± 0.0014 | 2/3 |
| Pretrained TST | +0.0012 ± 0.0011 | -0.0068 ± 0.0017 | 2/3 |
| PatchTST | +0.0094 ± 0.0031 | +0.0105 ± 0.0075 | 3/3 |

AUROC deltas are small (0.0003 - 0.0094), all within 1%. AUPRC deltas are more varied, with TGT Clinical-only underperforming Standard Transformer on precision-recall. Only PatchTST shows consistent improvement across all seeds.

### Step 2: Summary Statistics

Magnitude and consistency analysis:
- Standard Transformer: ΔAUROC = +0.0003 ± 0.0024 means the 95% CI crosses zero (±0.0048)
- PatchTST comparison shows most consistent advantage (3/3 seeds)
- LSTM comparison shows very small differences with inconsistent direction

All differences are within training-induced stochasticity and do not support a claim of meaningful superiority for TGT over transformer baselines.

### Step 3: Significance Testing (Optional)

Paired prediction-level hypothesis testing would require extracting raw test-set probability predictions from Standard Transformer (seed 123) and TGT Clinical-only (seed 123). DeLong's test or bootstrap paired resampling (1000 iterations) could be applied to the 6,182-sample test set. Current stage: Analysis complete; seed-wise descriptive statistics provide sufficient evidence for negating claims of superiority.

## Interpretation for the Paper

TGT Clinical-only achieves numerically competitive performance with transformer baselines, with seed-wise differences too small to be practically significant. While the graph-based approach matches attention mechanisms on discrimination ability (AUROC), it does not surpass them nor the best shallow baseline (LightGBM). Both explicit gradient-boosted interactions and implicit transformer attention capture similar clinical patterns effectively. TGT is most clearly distinguished from PatchTST but represents an alternative approach to transformers rather than an improvement.

## References

See Table 1. Main predictive performance on ICU mortality for detailed metrics across all models. Main claim: Does the proposed TGT perform competitively against strong baselines on ICU mortality?

Answer: TGT Clinical-only achieves essentially equivalent performance to transformer baselines when analyzed at the seed level (vs Standard Transformer: ΔAUROC = +0.0003 ± 0.0024). The aggregate statistics mask the reality that seed-wise differences are less than 0.3%, all within training-induced stochasticity. The TGT approach is numerically competitive but shows no meaningful performance advantage except when compared to PatchTST (ΔAUROC = +0.0094, wins 3/3 seeds).
