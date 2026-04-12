# Graph Structure Ablation — paper subsection

## Caption — Table 4
**Table 4. Graph structure ablation on ICU mortality prediction.** We compare a
no-edge baseline, a temporal-only graph, a clinical-only graph, and a full graph under
identical data splits, seeds, and training settings. Values are mean ± standard
deviation over three training seeds (42, 123, 2024). Smaller subgroup-gap values
indicate smaller demographic performance disparities. Overall AUROC and AUPRC are
computed from the same fairness evaluation pipeline used for the subgroup-gap
calculations, preserving within-row consistency.

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
evaluation settings, with the same three seeds and the same test cohort. The full
graph achieved the highest overall predictive performance (AUROC 0.8328 ± 0.0106,
AUPRC 0.4825 ± 0.0344) and also the smallest race, sex, and age AUROC gaps
(0.1292, 0.0033, and 0.0729, respectively), with the clinical-only variant a close
second on every metric. The temporal-only variant, in contrast, was both the weakest
overall predictor (AUROC 0.8231 ± 0.0107) and the most disparate across race
subgroups (race gap 0.1649). Moving from the no-edge baseline to the clinical-only
graph changed AUROC by +0.0043 and the race AUROC gap by −0.0050, whereas moving
from the no-edge baseline to the temporal-only graph changed AUROC by −0.0050 and
the race AUROC gap by +0.0296 — the same dataset, splits, and seeds produced
opposite movements on both axes depending on which edges were encoded. Across the
four conditions, the effect of graph structure was therefore not uniform:
temporal-only topology was associated with the largest race disparity, whereas
clinical and full-graph variants were associated with the smallest race disparities
and with small positive AUROC changes that remain within seed-level noise. Taken
together, these results suggest that the fairness impact of graph modelling depends
on which relationships are encoded rather than on the presence of a graph per se,
and that within this design space clinical edges — not temporal ones — are
associated with reduced subgroup disparity under matched training conditions.

## Supporting numbers (for reference while writing)

| Variant | AUROC | AUPRC | Sex gap | Age gap | Race gap |
|---|---|---|---|---|---|
| No-edge baseline    | 0.8281 ± 0.0139 | 0.4731 ± 0.0264 | 0.0053 | 0.0792 | 0.1352 |
| Temporal-only graph | 0.8231 ± 0.0107 | 0.4532 ± 0.0323 | 0.0112 | 0.0768 | 0.1649 |
| Clinical-only graph | 0.8324 ± 0.0112 | 0.4818 ± 0.0347 | 0.0040 | 0.0788 | 0.1302 |
| Full graph          | 0.8328 ± 0.0106 | 0.4825 ± 0.0344 | 0.0033 | 0.0729 | 0.1292 |

### Deltas vs no-edge baseline

| Variant | ΔAUROC | ΔAUPRC | ΔSex gap | ΔAge gap | ΔRace gap |
|---|---|---|---|---|---|
| Temporal-only graph | −0.0050 | −0.0199 | +0.0059 | −0.0023 | +0.0296 |
| Clinical-only graph | +0.0043 | +0.0088 | −0.0014 | −0.0004 | −0.0050 |
| Full graph          | +0.0047 | +0.0094 | −0.0020 | −0.0063 | −0.0061 |

### Matched-conditions audit

- **Test cohort**: n = 6,182 with 848 positives for every (variant × seed) evaluation.
- **Seeds**: exactly {42, 123, 2024} for every variant.
- **Evaluation pipeline**: every overall AUROC/AUPRC in the table above, and every
  subgroup gap, is derived from `Task6/results/<variant>_seed<seed>_subgroup_metrics.csv`
  and `…_disparity_summary.csv`. Nothing is pulled from Task 5's training-time summary,
  so every row in Table 4 shares a single evaluation pipeline.
