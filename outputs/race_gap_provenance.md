# Provenance of the Table 2 race-gap numbers (camera-ready scrub)

**Question:** Where do the paper's race AUROC gaps (TGTFull 0.1291, LSTM 0.1352, "TGTFull
achieves the smallest race AUROC gap") come from, and do they reproduce?

**Short answer:** They reproduce *only* from a stale, per-seed-**re-split** evaluation
(Task6, 2026-04-05). The team's own later fixed-split retraining (Task10 `task10_pred.py`,
2026-06-09) — which actually implements the paper's §3.8 "identical splits" protocol —
**reverses** the ordering, but those predictions were never evaluated into gaps.

---

## 1. Exact source of the published numbers

- `Task6/results/task6_final_summary.csv`, `fairness_gap_summary_by_model.csv`,
  and the per-seed `Task6/results/{model}_seed{seed}_disparity_summary.csv`.
- Re-surfaced unchanged by `Task10/build_graph_structure_ablation.py` (lines 61, 87–88),
  which reads those Task6 files and writes `Task10/Task10_GSA_main.csv`.
- Confirmed authoritative: Task6 3-seed mean overall AUROC == paper Table 5 exactly
  (LSTM 0.8281, TGTFull 0.8328).

## 2. The mechanism — split varies per seed (stale path)

Race subgroup counts change across seeds in the saved Task6 results, proving each seed
used its own train/test split (split seed = model seed in the old code path):

| seed | Asian | Black | Hispanic | Other | Unknown | White |
|------|-------|-------|----------|-------|---------|-------|
| 42   | 182   | 682   | 239      | 447   | 580     | 4050  |
| 123  | 170   | 665   | 234      | 445   | 611     | 4052  |
| 2024 | 179   | 625   | 214      | 463   | 625     | 4071  |

Per-seed race AUROC gap (max−min over 6 race groups; Multiple excluded, n<30):

| seed | LSTM | TGTFull | LSTM − TGTFull |
|------|------|---------|----------------|
| 42   | 0.1838 | 0.1795 | +0.0043 |
| 123  | 0.0863 | 0.1231 | **−0.0368** (TGTFull worse) |
| 2024 | 0.1356 | 0.0849 | +0.0507 |
| **mean** | **0.1352** | **0.1292** | **+0.0061 ± 0.0438** |

The headline "TGTFull smallest" is a **+0.006 mean paired difference with ±0.044
seed-to-seed std** — ~7× smaller than its own noise, and it flips sign in seed 123.
The entire positive mean is carried by seed 2024.

## 3. The contradiction — fixed-split retraining reverses it (correct §3.8 path)

`Task10/task10_pred.py` retrains all 4 models on a **single fixed split** (`SPLIT_SEED=42`,
line 73; only `torch.manual_seed(seed)` varies). All 12 prediction CSVs share the same
6,182 test stay_ids. Evaluating race gaps from these (the protocol the paper *describes*):

| seed | LSTM | TGTFull | LSTM − TGTFull |
|------|------|---------|----------------|
| 42   | 0.1984 | 0.2259 | −0.0275 |
| 123  | 0.1706 | 0.2091 | −0.0384 |
| 2024 | 0.1850 | 0.1919 | −0.0070 |
| **mean** | **0.1847** | **0.2090** | **−0.0243 ± 0.0160** |

Under the matched fixed split, **TGTFull's race gap is larger than LSTM's in 3/3 seeds**
(opposite of the claim). The same fixed-split 3-seed averages also fail "TGTFull dominates
on all attributes": TGTFull loses on sex (0.0075 vs 0.0043) and race, wins only on age.

These fixed-split predictions exist on disk but were never turned into subgroup gaps;
`build_graph_structure_ablation.py` ignored them and re-read the stale Task6 summaries.

## 4. Why the magnitudes look different elsewhere

The gap is driven by the Asian subgroup (n≈180, AUROC ≈ 0.69–0.72). Any race grouping that
includes Asian gives ~0.13–0.22; restricting to White-vs-Black gives ~0.01–0.03. No
grouping reproduces 0.129/0.135 *and* the TGTFull<LSTM ordering except the noisy
per-seed-resplit average above.

## 5. Implication / recommendation

- Claim (i) "TGTFull achieves the smallest race AUROC gap under matched training" is not
  supported once you (a) use the matched fixed split the paper describes, or (b) account
  for the ±0.044 seed variance on a 0.006 difference.
- Recommended camera-ready fix: re-evaluate the Task10 fixed-split predictions into
  subgroup gaps (one evaluation pass, no retraining), report the paired seed difference
  and bootstrap CI, and soften/withdraw the "smallest race gap" claim accordingly.
- The bootstrap (Task 1.1) should run on the **fixed-split Task10 predictions**, not the
  Task9 cache (whose TGTFull entry is seed-42-only) nor the stale Task6 summaries.

*Data integrity note:* the `Task9/Task9Results/T9_predictions_cache.pkl` `tgt_full_graph`
entry is byte-identical to Task10 TGTFull seed-42 (not a 3-seed average), despite the
runfile labeling it seed-averaged.
