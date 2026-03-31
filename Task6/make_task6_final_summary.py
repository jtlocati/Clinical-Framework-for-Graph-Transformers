from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK4_RESULTS = REPO_ROOT / "Task4" / "results"
TASK6_RESULTS = REPO_ROOT / "Task6" / "results"

# Task 5 performance summaries
task5_perf = {
    "LSTM": {
        "AUROC_mean": 0.8283656465111672,
        "AUROC_std": 0.011545387299864611,
        "AUPRC_mean": 0.4712054587702066,
        "AUPRC_std": 0.024071486874358203,
    },
    "TGTTemporal": {
        "AUROC_mean": 0.8231,
        "AUROC_std": 0.0087,
        "AUPRC_mean": 0.4532,
        "AUPRC_std": 0.0264,
    },
    "TGTClinical": {
        "AUROC_mean": 0.8324,
        "AUROC_std": 0.0091,
        "AUPRC_mean": 0.4818,
        "AUPRC_std": 0.0283,
    },
    "TGTFull": {
        "AUROC_mean": 0.8328,
        "AUROC_std": 0.0087,
        "AUPRC_mean": 0.4825,
        "AUPRC_std": 0.0281,
    },
}

perf_df = (
    pd.DataFrame.from_dict(task5_perf, orient="index")
    .reset_index()
    .rename(columns={"index": "model"})
)

fair_df = pd.read_csv(TASK6_RESULTS / "fairness_gap_summary_by_model.csv")

pivot = fair_df.pivot(index="model", columns="attribute")
pivot.columns = [f"{attr}_{metric}" for metric, attr in pivot.columns]
pivot = pivot.reset_index()

final_df = perf_df.merge(pivot, on="model", how="inner")

out_path = TASK6_RESULTS / "task6_final_summary.csv"
final_df.to_csv(out_path, index=False)

print(final_df.to_string(index=False))
print(f"\nSaved: {out_path}")