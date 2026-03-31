import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[1]
DEMOGRAPHICS_PATH = REPO_ROOT / "Tests" / "outputs" / "Task2Demographics.csv"
RESULTS_DIR = REPO_ROOT / "Task6" / "results"


def safe_metrics(df: pd.DataFrame, group_name: str, min_count: int = 20, min_pos: int = 2, min_neg: int = 2) -> dict:
    y = df["label"].values
    s = df["pred_score"].values
    count = len(df)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())

    if count < min_count or pos < min_pos or neg < min_neg:
        return {
            "group": group_name,
            "count": count,
            "positives": pos,
            "negatives": neg,
            "AUROC": np.nan,
            "AUPRC": np.nan,
            "note": "Too few samples",
        }

    return {
        "group": group_name,
        "count": count,
        "positives": pos,
        "negatives": neg,
        "AUROC": roc_auc_score(y, s),
        "AUPRC": average_precision_score(y, s),
        "note": "ok",
    }


def summarize_attribute(res_df: pd.DataFrame, prefix: str) -> dict:
    sub = res_df[res_df["group"].str.startswith(prefix)].copy()
    sub_ok = sub[sub["note"] == "ok"].copy()

    if len(sub_ok) == 0:
        return {
            "attribute": prefix.rstrip("_"),
            "AUROC_gap": np.nan,
            "AUROC_std": np.nan,
            "AUPRC_gap": np.nan,
            "AUPRC_std": np.nan,
            "n_groups_used": 0,
        }

    return {
        "attribute": prefix.rstrip("_"),
        "AUROC_gap": sub_ok["AUROC"].max() - sub_ok["AUROC"].min(),
        "AUROC_std": sub_ok["AUROC"].std(ddof=0),
        "AUPRC_gap": sub_ok["AUPRC"].max() - sub_ok["AUPRC"].min(),
        "AUPRC_std": sub_ok["AUPRC"].std(ddof=0),
        "n_groups_used": len(sub_ok),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-file", required=True, help="Path to prediction CSV")
    parser.add_argument("--model-name", required=True, help="Model label to save in outputs")
    parser.add_argument("--seed", required=True, help="Seed label to save in outputs")
    args = parser.parse_args()

    pred_path = Path(args.pred_file)
    pred = pd.read_csv(pred_path)

    demo = pd.read_csv(DEMOGRAPHICS_PATH)

    pred.columns = pred.columns.str.strip()
    demo.columns = demo.columns.str.strip()

    merged = pred.merge(demo, on="stay_id", how="inner").copy()

    if len(merged) == 0:
        raise ValueError("Merged dataframe is empty. Check stay_id alignment.")

    if not (merged["label_x"] == merged["label_y"]).all():
        mismatches = int((merged["label_x"] != merged["label_y"]).sum())
        raise ValueError(f"Found {mismatches} label mismatches after merge.")

    merged["label"] = merged["label_x"]

    overall = safe_metrics(merged, "overall")

    results = []
    results.append(
        {
            "model": args.model_name,
            "seed": args.seed,
            **overall,
        }
    )

    for sex in sorted(merged["sex"].dropna().unique()):
        results.append(
            {
                "model": args.model_name,
                "seed": args.seed,
                **safe_metrics(merged[merged["sex"] == sex], f"sex_{sex}"),
            }
        )

    age_order = ["<40", "40-65", "65+"]
    for age_group in age_order:
        results.append(
            {
                "model": args.model_name,
                "seed": args.seed,
                **safe_metrics(merged[merged["age_group"] == age_group], f"age_{age_group}"),
            }
        )

    for race in sorted(merged["race"].dropna().unique()):
        results.append(
            {
                "model": args.model_name,
                "seed": args.seed,
                **safe_metrics(merged[merged["race"] == race], f"race_{race}"),
            }
        )

    subgroup_df = pd.DataFrame(results)

    disparity_df = pd.DataFrame(
        [
            {
                "model": args.model_name,
                "seed": args.seed,
                **summarize_attribute(subgroup_df, "sex_"),
            },
            {
                "model": args.model_name,
                "seed": args.seed,
                **summarize_attribute(subgroup_df, "age_"),
            },
            {
                "model": args.model_name,
                "seed": args.seed,
                **summarize_attribute(subgroup_df, "race_"),
            },
        ]
    )

    subgroup_out = RESULTS_DIR / f"{args.model_name}_seed{args.seed}_subgroup_metrics.csv"
    disparity_out = RESULTS_DIR / f"{args.model_name}_seed{args.seed}_disparity_summary.csv"

    subgroup_df.to_csv(subgroup_out, index=False)
    disparity_df.to_csv(disparity_out, index=False)

    print("OVERALL")
    print(subgroup_df[subgroup_df["group"] == "overall"].to_string(index=False))
    print("\nDISPARITY SUMMARY")
    print(disparity_df.to_string(index=False))
    print(f"\nSaved subgroup metrics: {subgroup_out}")
    print(f"Saved disparity summary: {disparity_out}")


if __name__ == "__main__":
    main()