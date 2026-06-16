# Repo cleanup — Stage B manifest

Branch **`repo-cleanup`** (NOT merged — waits for co-author + Kiran scope decision).
Everything is recoverable via git. Quarantine scope: **Option 2 (full contaminated chain)**.

## Reference hashes

| ref | hash | meaning |
|---|---|---|
| Original pre-cleanup `main` | `1fb0ab7ce512080625786080e5d23ef67576cbff` | repo state before any cleanup |
| Pre-guard reference (HEAD before group c) | `f2fda4977326725f8daf40ea9767fd4385f5ee19` | quarantine done, scripts still unmodified |
| Canonical unmodified `Task8/run_validation.py` blob | `8d0b9f84bcd77d8337625229bea239ae52a8a77f` | the original coupled-split generator |

The blob above is the canonical version with the bug:
`Task8/run_validation.py:141` `prepare_for_seed(X_raw, M, y, ids, args.seed)` →
`Task4/DataPipeline.py:345` `train_test_split(..., random_state=seed, ...)`.
Retrieve: `git cat-file -p 8d0b9f84` or `git show f2fda49:Task8/run_validation.py`.

## Commit groups (each its own commit on `repo-cleanup`)

| group | commit | summary |
|---|---|---|
| (a) | `7c7d08c12fca432ca1e05269d78303d238a526bd` | create `deprecated_stale_eval/` + quarantine `README.md` |
| (b) | `f2fda4977326725f8daf40ea9767fd4385f5ee19` | `git mv` quarantine set (83 renames) |
| (c) | `1828801d7f1db344bf23f251824e7a8278a85bdb` | add `ALLOW_DEPRECATED_EVAL` guards (4 scripts) + pin canonical original in README |
| (d) | `45e29e2d23d673d7ccff001c7879e1c81dedd73f` | remove untracked `__pycache__` dirs (empty marker commit — see note) |
| (e) | `da8ca4eb04d71a944226f4ab23d8089059173239` | extend `.gitignore` |
| standalone | `ae2d8d425606ef36e9bb41fc29bf705c334f0f0d` | commit `Task9/__init__.py` (0-byte package marker; unrelated to cleanup, kept as its own commit) |
| (f) | _this commit_ | track the correction evidence under `outputs/` (provenance, corrected gaps, plan, this manifest) |

## MOVED (quarantined via `git mv` → `deprecated_stale_eval/`, history preserved)

- `Task6/results/` entire directory → `deprecated_stale_eval/Task6/results/` (82 files):
  38 `*_disparity_summary.csv`, 38 `*_subgroup_metrics.csv`,
  `fairness_gap_summary_by_model.csv`, `all_model_disparity_summaries.csv`,
  `task6_final_summary.csv`, and 3 `task6_*.png`.
- `Task9/Task9Results/T9_predictions_cache.pkl` →
  `deprecated_stale_eval/Task9/Task9Results/T9_predictions_cache.pkl` (seed-42-locked).
- Total: **83 files**, ~1.1 MB. `deprecated_stale_eval/` now holds 84 tracked files
  (83 moved + `README.md`). `Task6/results/` no longer exists in the working tree.

## DELETED (safe — genuine cruft)

- **9** untracked `__pycache__/` dirs removed from the working tree
  (`.`, `Task4–10`, `Tests`). The original 8 plus `Task10/__pycache__`, which was
  regenerated when I executed the guard scripts to verify they raise. All untracked and
  gitignored, so **there is no tracked git change** for this group — commit (d) is an
  intentional empty marker (restricted to an unchanged pathspec so it does not sweep in the
  staged `Task9/__init__.py`). Nothing else qualified (no `.DS_Store`, `.ipynb_checkpoints`,
  `.pytest_cache`, editor temp, or scratch files inside the repo).

## GUARDED (additive only — original logic intact below each guard, insertions-only diff)

Each gains a top-of-file `RuntimeError` guard gated by `ALLOW_DEPRECATED_EVAL=1` (forensic
escape hatch), pointing to `deprecated_stale_eval/README.md` and the Task10 fixed-split
source. Verified: all 4 compile, raise (exit 1) without the env var, and bypass with it.

- `Task10/build_graph_structure_ablation.py` (stale re-aggregator)
- `Task10/build_ablation_table.py` (stale re-aggregator)
- `Task8/run_validation.py` (coupled-split generator)
- `Task6/aggregate_fairness_results.py` (globs the now-moved summaries; top-level script,
  no `__main__`, imported nowhere — guarded per spec since it is run-as-script, not
  import-only)

**Not guarded:** `Task6/compute_fairness_metrics.py` — neutral metric tool whose staleness
was purely upstream; guarding it would block legitimate use.

## IGNORED (.gitignore additions)

`*.pyo`, `*.pyd`, `.pytest_cache/`, `.mypy_cache/`, `.ipynb_checkpoints/`, `.DS_Store`,
`*~`, `*.swp`, `*.swo` (kept existing `__pycache__/`, `*.pyc`, `.venv/`, `.env`).
`outputs/` and `deprecated_stale_eval/` are deliberately **not** ignored.

## KEEP — verified untouched

- `Task10/results/*_test_predictions.csv` — 12 present (source of record).
- `analysis/corrected_eval.py` — present.
- `outputs/` evidence — 5 files present (`race_gap_provenance.md`,
  `seed_provenance_confirmation.md`, `corrected_gaps.csv`, `corrected_gaps_summary.csv`,
  `corrected_eval_notes.txt`); plus `cleanup_plan.md` and this manifest.
- `Task6/results_T6/` — preserved (separate SimpleModelEval output, never fed paper tables).
- `Task9/__init__.py` — 0-byte package marker; was excluded from cleanup groups (a)–(e),
  then committed on its own (`ae2d8d4`) so it is no longer a staged-but-uncommitted trap.

## TRACKED in group (f) — correction evidence (patient-level check passed)

These were untracked working-tree files; group (f) puts them under version control so the
evidentiary trail travels with the branch and is reproducible:
`outputs/race_gap_provenance.md`, `outputs/seed_provenance_confirmation.md`,
`outputs/corrected_gaps.csv`, `outputs/corrected_gaps_summary.csv`,
`outputs/corrected_eval_notes.txt`, `outputs/cleanup_plan.md`, `outputs/cleanup_manifest.md`.

**Patient-level data check (per the commit caveat): PASSED.** The two CSVs are subgroup-level
aggregates — `corrected_gaps.csv` is (model, attribute, seed) rows (36 data rows, no patient
column); `corrected_gaps_summary.csv` is 3-seed means/std + paired diffs. The `.md`/`.txt`
files are prose; the only `stay_id` token is the phrase "6,182 test stay_ids" (a count), and
the 8-digit strings in this manifest are commit/blob hashes, not IDs. No per-patient rows are
tracked. The raw per-patient `Task10/results/*_test_predictions.csv` stay where they are
under `Task10/` and were not moved.

## Still untracked (not in scope of these instructions) — flagged for a decision

- `analysis/corrected_eval.py` — the script that produces the corrected gaps. It is the one
  remaining reproducibility gap: a fresh clone of this branch can regenerate `corrected_gaps*`
  only if this is also tracked. Recommend a follow-up commit to track it.
- `figures/` (untracked `06_*.py`, `07_*.py`, `main/Figure5_*`, `main/Figure6_*`) — paper
  figure assets; left untracked pending the figures/paper-assets decision.

## Not done (by design)

Branch **not merged** — same hold as the rest, pending the scope + chair-notification
decisions.
