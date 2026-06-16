# === DEPRECATED EVALUATION PATH (added by repo-cleanup; do not remove) ===
# This top-level script globs the now-quarantined Task6/results/*_disparity_summary.csv
# (stale per-seed-resplit). Authoritative source of record:
# Task10/results/*_test_predictions.csv via analysis/corrected_eval.py.
# See deprecated_stale_eval/README.md and outputs/race_gap_provenance.md.
import os as _dep_os
if _dep_os.environ.get("ALLOW_DEPRECATED_EVAL") != "1":
    raise RuntimeError(
        "DEPRECATED stale evaluation path; its inputs were quarantined to "
        "deprecated_stale_eval/. Use Task10/results/*_test_predictions.csv via "
        "analysis/corrected_eval.py. "
        "Set ALLOW_DEPRECATED_EVAL=1 to run anyway (forensic re-runs only)."
    )
# === end deprecation guard ===

from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "Task6" / "results"

files = sorted(RESULTS_DIR.glob("*_disparity_summary.csv"))

dfs = []
for f in files:
    df = pd.read_csv(f)
    dfs.append(df)

all_df = pd.concat(dfs, ignore_index=True)

# Save the full seed-level table
all_out = RESULTS_DIR / "all_model_disparity_summaries.csv"
all_df.to_csv(all_out, index=False)

# Aggregate across seeds by model + attribute
agg_df = (
    all_df.groupby(["model", "attribute"], as_index=False)
    .agg(
        AUROC_gap_mean=("AUROC_gap", "mean"),
        AUROC_gap_std=("AUROC_gap", "std"),
        AUROC_std_mean=("AUROC_std", "mean"),
        AUPRC_gap_mean=("AUPRC_gap", "mean"),
        AUPRC_gap_std=("AUPRC_gap", "std"),
        AUPRC_std_mean=("AUPRC_std", "mean"),
        n_seeds=("seed", "count"),
    )
)

agg_out = RESULTS_DIR / "fairness_gap_summary_by_model.csv"
agg_df.to_csv(agg_out, index=False)

print("Saved:", all_out)
print("Saved:", agg_out)
print("\nSeed-level disparity summaries:")
print(all_df.to_string(index=False))
print("\nAggregated fairness gap summary:")
print(agg_df.to_string(index=False))