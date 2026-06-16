# Repo cleanup plan — Stage A (inventory only, no changes made)

Scope: git repo `Clinical-Framework-for-Graph-Transformers` (the parent `DDJA/` is not a
git repo; `YAIB/`, `yaib_logs/`, `DDJA-Thomas.pem`, parent `.DS_Store` are out of scope).
Current branch: `main`. **No files have been moved, deleted, or ignored. This document is
the only thing written.** Stage B (branch `repo-cleanup`, git mv / delete / gitignore /
commit) awaits approval.

Sizes are content bytes; for the 38-file CSV sets I also note on-disk (`du`, 4 KB-block
rounded) because the per-file content is tiny.

---

## 1. KEEP — do not touch (authoritative + evidence)

| path | reason |
|---|---|
| `Task10/results/*_test_predictions.csv` (12) | source of record (fixed-split predictions) |
| `analysis/corrected_eval.py` | corrected re-evaluation script |
| `outputs/race_gap_provenance.md`, `outputs/seed_provenance_confirmation.md` | provenance evidence |
| `outputs/corrected_gaps.csv`, `outputs/corrected_gaps_summary.csv`, `outputs/corrected_eval_notes.txt` | corrected results |
| `outputs/cleanup_plan.md` (this file) | the plan |
| `Task10/build_graph_structure_ablation.py`, `Task10/build_ablation_table.py` | stale re-aggregators — KEEP in place as part of the trail (do **not** delete) |
| `figures/` (incl. untracked `06_*.py`, `07_*.py`, `main/Figure5_*`, `main/Figure6_*`) | paper figure assets |
| root PDFs `ablation_comparison.pdf`, `fig1_model_comparison.pdf`, `model_comparison.pdf` | paper figure assets |
| all `Task*/` source code, `config.py`, `requirements.txt`, `README.md`, instruction sheets | source/config |

No `main.tex` exists in-repo (the paper source lives elsewhere); nothing to do there.

**KEEP total:** not sized (authoritative; the bulk of the repo).

---

## 2. QUARANTINE — `git mv` into `deprecated_stale_eval/`, do not delete (known-bad but evidence)

Preserve relative subpaths, e.g. `deprecated_stale_eval/Task6/results/...`,
`deprecated_stale_eval/Task9/Task9Results/...`. All are git-tracked → use `git mv`.

| path(s) | size | reason |
|---|---|---|
| `Task6/results/*_disparity_summary.csv` (38 files) | ~16 KB (152 KB on disk) | stale per-seed-resplit gaps (split seed coupled to model seed) |
| `Task6/results/task6_final_summary.csv` | 2,015 B | stale aggregate carrying paper's 0.1352 / 0.1291 race gaps |
| `Task9/Task9Results/T9_predictions_cache.pkl` | 198,225 B (~194 KB) | seed-42-locked cache (not the 3-seed average it claims) |

**QUARANTINE total:** ≈ 216 KB content (≈ 352 KB on disk).
Plus: write `deprecated_stale_eval/README.md` explaining each item and pointing to
`outputs/race_gap_provenance.md`.

**Breakage this intentionally causes (the point — "impossible to reuse by accident"), to be noted in the manifest:**
- `Task6/aggregate_fairness_results.py` globs `*_disparity_summary.csv` (line 7) → will read 0 files after the move.
- `Task10/build_graph_structure_ablation.py` (line 87) and `Task10/build_ablation_table.py` read the disparity summaries → will `FileNotFoundError`. These are the stale re-aggregators we want disabled.
- `Task8/run_validation.py` / `Task6/compute_fairness_metrics.py` would *re-create* files in `Task6/results/` if re-run (they write there) — quarantine does not block regeneration, only removes the existing stale copies from the live path.
- `Task9/runfile.py` reads the cache only `if os.path.exists(...)`; after the move it will rebuild the cache (now via the multi-seed loop) on next run. `analysis/corrected_eval.py` does **not** read the cache, so the corrected results are unaffected.

---

## 3. SAFE-DELETE — genuine cruft only

| path(s) | size | reason |
|---|---|---|
| `__pycache__/` × 8 (`.`, `Task4–9`, `Tests`) | 428 KB | Python bytecode cache, untracked, regenerable, already gitignored |

No `*.pyc/*.pyo` outside `__pycache__`, no `.ipynb_checkpoints/`, `.pytest_cache/`, mpl/font
caches, `.DS_Store` (none inside the repo), `*~/*.swp`, `tmp_*/scratch_*/*.bak`, or zero-byte
scratch files were found. (Zero-byte `__init__.py` package markers exist but are KEEP.)

**SAFE-DELETE total:** ≈ 428 KB (all untracked working-tree only).

---

## 4. UNCERTAIN — needs human decision (not acted on)

These are stale-but-not-in-the-named-quarantine-scope, or derived artifacts. Flagged rather
than moved/deleted.

| path(s) | size | why uncertain |
|---|---|---|
| `Task6/results/*_subgroup_metrics.csv` (38) | ~16 KB (152 KB on disk) | **same stale per-seed-resplit data**; the disparity summaries are derived from these. Not in the task's quarantine glob, but equally contaminated, and `build_graph_structure_ablation.py:61/88` + `make_task6_*.py` read them. Recommend quarantining alongside the disparity summaries for consistency. |
| `Task6/results/fairness_gap_summary_by_model.csv` | 1,748 B | stale aggregate that literally holds the paper's race gaps; `make_task6_final_summary.py:42` reads it to regenerate `task6_final_summary.csv`. Leaving it lets the quarantined final-summary be re-created. Strong evidence — recommend quarantining too. |
| `Task6/results/all_model_disparity_summaries.csv` | 3,499 B | stale aggregate (suffix `summaries` not matched by the `*_disparity_summary.csv` glob); output of `aggregate_fairness_results.py`. |
| `Task6/results/task6_*.png` (3: heatmap, dotplot, barplot) | ~890 KB | plots rendered from the stale subgroup metrics; regenerable, harmless, but visually misleading. Keep or quarantine? |
| `Task6/results_T6/` contents | not enumerated | older Task6 fairness outputs (`T6_*.csv/png`); status unclear — leave for review. |
| staged `Task9/__init__.py` (already `git add`-ed by a prior session) | 0 B | legitimate package marker, but currently staged on `main`; flag so the human decides whether that staging should ride along. |

**UNCERTAIN total:** ≈ 911 KB (dominated by the 3 PNGs).

---

## Category byte totals

| category | content bytes (approx) |
|---|---|
| KEEP | not sized (authoritative bulk) |
| QUARANTINE | ≈ 216 KB |
| SAFE-DELETE | ≈ 428 KB |
| UNCERTAIN | ≈ 911 KB |

## Key decision for the human before Stage B

The task scopes quarantine to `*_disparity_summary.csv` + `task6_final_summary.csv`, but the
**upstream `*_subgroup_metrics.csv` and `fairness_gap_summary_by_model.csv` are the same
contaminated data** and can regenerate the quarantined files. Options:
1. **Quarantine exactly as scoped** (disparity summaries + final summary + cache) — literal, but leaves the stale chain partly in place.
2. **Quarantine the whole stale chain** (also the 38 subgroup_metrics + the 2 aggregates) — fully prevents accidental reuse; recommended for the stated goal.

I will not proceed to Stage B (branch, moves, deletes, gitignore) until you approve and tell
me which quarantine scope (1 or 2) to use.
