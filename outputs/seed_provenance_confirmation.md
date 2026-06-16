# Seed-provenance confirmation (Task 1.1b)

Read-only forensics on the race-gap reversal. No code was modified, nothing retrained or
re-run. All evidence is from source files, git history, and on-disk artifacts.

---

## Check 1 — Two seed roles collapsed into one variable in the post-hoc generator

**Generator of the stale per-seed disparity summaries:** `Task8/run_validation.py`. It
trains one (model, seed, intervention) run, saves predictions, then shells out to
`Task6/compute_fairness_metrics.py`, which writes
`Task6/results/{model}_seed{seed}_{subgroup_metrics,disparity_summary}.csv`
(`compute_fairness_metrics.py:11` `RESULTS_DIR = .../Task6/results`; invocation at
`run_validation.py:95-107` via `compute_fairness_outputs`).

**Seed-like variables and their roles:**

| variable | role | where |
|---|---|---|
| `args.seed` | **model initialization** | `run_validation.py:139` `set_seed(args.seed)` → `np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all` (lines 28-30) |
| `args.seed` | **train/test split** | `run_validation.py:141` (quoted below) → `prepare_for_seed` → `split_data(random_state=seed)` (`Task4/DataPipeline.py:345,350`) |

There is **only one** seed variable. It is **not** held to a constant for the split — the
split seed is the same per-run `args.seed` that also seeds the model. Quoted assignment:

```python
# Task8/run_validation.py:141
splits, _, _ = prepare_for_seed(X_raw, M, y, ids, args.seed)
```

Sweeping `args.seed ∈ {42,123,2024}` therefore draws a **different stratified split each
time** (`train_test_split(random_state=seed)`), so each seed scores a different test
cohort. The fix (Task9/Task10) introduces a *constant* split seed distinct from the model
seed: `Task6/config.py:16 SPLIT_SEED = 42`; `Task10/task10_pred.py:73`
`prepare_for_seed(..., SPLIT_SEED)` with the model seed varied separately in the loop
(`task10_pred.py:94,114`).

**Verdict:** CONFIRMED — split seed is taken from the per-run/model seed, not a constant;
no separate split-seed variable existed in the generator.

---

## Check 2 — Resplit fingerprint (drifting vs stable subgroup counts)

Per-seed **race** subgroup counts:

```
Task6 stale path (Task6/results/TGTFull_seed{S}_subgroup_metrics.csv):
  seed42    Asian=182 Black=682 Hispanic=239 Other=447 Unknown=580 White=4050  (Multiple=2)
  seed123   Asian=170 Black=665 Hispanic=234 Other=445 Unknown=611 White=4052  (Multiple=5)
  seed2024  Asian=179 Black=625 Hispanic=214 Other=463 Unknown=625 White=4071  (Multiple=5)

Task10 fixed split (Task10/results/TGTFull_seed{S}_test_predictions.csv):
  seed42    Asian=182 Black=682 Hispanic=239 Other=447 Unknown=580 White=4050
  seed123   Asian=182 Black=682 Hispanic=239 Other=447 Unknown=580 White=4050
  seed2024  Asian=182 Black=682 Hispanic=239 Other=447 Unknown=580 White=4050
```

The known fingerprint (Asian 182/170/179) reproduces exactly in the stale path and is flat
(182/182/182) in the fixed split. Note `compute_fairness_metrics.py` uses `min_count=20`
(not 30); only Multiple (n=2–5) is dropped either way, so the threshold choice does not
affect the race gap.

**Verdict:** CONFIRMED — drifting counts in Task6 + identical counts in Task10 establish
that the Task6 cohort changed per seed while Task10's did not.

---

## Check 3 — The warning conditional

The only warning in the fairness-eval code is `Task6/eval.py:19`:

```python
print(f"!!!!WARNING: col: {col_name} is not in dempgraphics, Will skip")
```

It lives inside `eval.BuildMasks`, and it guards a **different** condition — a demographic
*column* missing from the demographics table — not a single-seed/locked-seed configuration.
Decisively, the stale generator never reaches it: `Task6/compute_fairness_metrics.py` does
its own demographic merge + `groupby` and **does not call `BuildMasks` or
`load_demographics`** (grep count = 0). So this warning is in a code path the
disparity-summary pipeline never executes, and in any case it cannot detect
split-equals-model-seed coupling.

**Did it ever fire?** No evidence. The string appears nowhere except `eval.py` itself; the
repo contains no captured stdout/`.log`/`.out` files, and no results artifact records it.

**Verdict:** No seed-configuration warning exists. The single extant warning checks a
missing demographic column, sits in an unused-by-this-path function, and would not catch
the bug. No evidence it fired.

---

## Check 4 — What the GSA actually reads

`Task10/build_graph_structure_ablation.py` loads from the **stale Task6 summaries**, not
the corrected Task10 fixed-split predictions:

```python
# line 61
fp = f"Task6/results/{model}_seed{seed}_subgroup_metrics.csv"
# line 87
dsp = pd.read_csv(f"Task6/results/{model}_seed{seed}_disparity_summary.csv")
# line 88
sgm = pd.read_csv(f"Task6/results/{model}_seed{seed}_subgroup_metrics.csv")
```

No path in this script references `Task10/results/*_test_predictions.csv`. Its integrity
asserts (lines 64-69) only check `overall count == 6182` and `positives == 848` — both
preserved by a stratified resplit — so they pass on the stale cohorts and give false
confidence that membership is matched.

**Verdict:** CONFIRMED — the GSA re-aggregates the stale per-seed-resplit Task6 source; it
never reads the corrected fixed-split predictions sitting in the same task folder.

---

## Check 5 — The Task9 cache seed artifact

The on-disk cache `Task9/Task9Results/T9_predictions_cache.pkl` was committed at
`b81d34f` ("Task9 initial results", 2026-04-20; file mtime 2026-04-20) — **before Task10
existed.** The code at that commit:

```python
# Task6/config.py:14 @ b81d34f
seed =  42 #same see for all cuz we are comparing directly
# Task9/runfile.py:21 @ b81d34f
splits, means, stddev = DataPipeline.prepare_for_seed(Rawx, M, y, ids, config6.seed)  # = 42
# Task9/runfile.py:32 @ b81d34f
predictions = modelmetrics.CollectMetrics(splits, config6.seed)                         # = 42
```

At build time `config6.seed` was the **scalar 42** ("same seed for all"), there was **no
seed loop**, and the single 42 fed both the split and model init. The cache is therefore a
**seed-42-only artifact for both models** — not a 3-seed average — even though the current
`Task9/runfile.py` (lines 33-51) prints "averaging across seeds" and loops
`config6.seed = [42,123,2024]`. The current runfile only rebuilds the cache when the file
is absent, so the stale seed-42 pickle was never regenerated.

This corrects my earlier (Task 1.1) inference that cached LSTM was "a genuine multi-seed
blend": the build-time code cannot produce a blend. Cached `tgt_full_graph` matches the
seed-42 TGTFull run to <1e-5 (reproduced by Task10 seed42); cached `LSTM` differs from
Task10's seed-42 LSTM only by training nondeterminism between the two runs
(cache trained `device="cpu"`, Task10 `device="cuda"`), correlation 0.94 — consistent with
two seed-42 runs, not with averaging.

**Was the lock reverted?** Partially. `Task6/config.py` now has `SPLIT_SEED=42`,
`ModelSeed=[42,123,2024]`, `seed=[42,123,2024]` (lines 16-18) and `runfile.py` now loops,
but the **cached pickle on disk is still the seed-42 lock** and is reused as-is by any
consumer that loads it.

**Verdict:** This is a **distinct second artifact**. Checks 1–4 are an *over-variation* bug
(split seed coupled to the model seed → a different cohort per seed). Check 5 is the
opposite *under-variation* bug (split and model seed both locked to a constant 42 → no seed
variation at all). Same root deficiency — no clean separation between a fixed split seed and
a varying model seed — but a different failure mode and a different file.

---

## Synthesis

- **(a) Task6 split-seed coupling — CONFIRMED.** `Task8/run_validation.py:141` passes the
  per-run `args.seed` (also the model-init seed via `set_seed`, line 139) straight into
  `prepare_for_seed`; `split_data` uses it as `random_state`. No constant split seed
  existed, so each seed scored a different cohort (Asian 182/170/179). The GSA
  (`build_graph_structure_ablation.py`, lines 61/87/88) re-published those stale summaries
  rather than reading the corrected Task10 fixed-split predictions.

- **(b) The warning never fired — and could not have.** The only warning in the eval code
  (`Task6/eval.py:19`) checks for a missing demographic column inside `BuildMasks`, a
  function the disparity-summary generator never calls. No log or artifact records it.

- **(c) The Task9 cache is a separate, second artifact.** It was built under a hardcoded
  scalar `seed = 42` lock (`config.py:14`, `Task9/runfile.py:21,32` @ `b81d34f`), making it
  seed-42-only for both models — an under-variation lock, opposite to the Checks 1–4
  over-variation coupling. The lock was reverted in config/runfile but the on-disk pickle
  was never regenerated.
