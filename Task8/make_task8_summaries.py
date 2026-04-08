from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK6_RESULTS_DIR = REPO_ROOT / "Task6" / "results"
TASK8_RESULTS_DIR = REPO_ROOT / "Task8" / "results"
TASK8_FINAL_TABLES_DIR = TASK8_RESULTS_DIR / "final_tables"
TASK8_CROSS_SEED_DIR = TASK8_RESULTS_DIR / "cross_seed_summaries"

SEEDS = [42, 123, 2024]
MODEL_SETTINGS = {
    "LSTM": ["original", "train_race_weighted", "train_race_weighted_varreg001"],
    "TGTFull": ["original", "train_race_weighted", "train_race_weighted_varreg005"],
}


def ensure_output_dirs() -> None:
    TASK8_FINAL_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    TASK8_CROSS_SEED_DIR.mkdir(parents=True, exist_ok=True)


def fairness_model_name(model: str, intervention: str) -> str:
    return model if intervention == "original" else f"{model}_{intervention}"


def load_overall_test_row(model: str, seed: int, intervention: str) -> dict:
    subgroup_path = TASK6_RESULTS_DIR / f"{fairness_model_name(model, intervention)}_seed{seed}_subgroup_metrics.csv"
    if not subgroup_path.exists():
        raise FileNotFoundError(f"Missing subgroup metrics file: {subgroup_path}")

    subgroup_df = pd.read_csv(subgroup_path)
    overall_df = subgroup_df[subgroup_df["group"] == "overall"]
    if overall_df.empty:
        raise ValueError(f"Missing overall row in subgroup metrics: {subgroup_path}")

    return overall_df.iloc[0].to_dict()


def load_disparity_row_map(model: str, seed: int, intervention: str) -> dict[str, dict]:
    disparity_path = TASK6_RESULTS_DIR / f"{fairness_model_name(model, intervention)}_seed{seed}_disparity_summary.csv"
    if not disparity_path.exists():
        raise FileNotFoundError(f"Missing disparity summary: {disparity_path}")

    disparity_df = pd.read_csv(disparity_path)
    return {
        row["attribute"]: row
        for _, row in disparity_df.iterrows()
    }


def build_seed_level_row(
    model: str,
    seed: int,
    intervention: str,
) -> dict:
    overall_row = load_overall_test_row(model, seed, intervention)
    disparity_map = load_disparity_row_map(model, seed, intervention)
    return {
        "model": model,
        "seed": seed,
        "setting": intervention,
        "test_auroc": overall_row["AUROC"],
        "test_auprc": overall_row["AUPRC"],
        "sex_auroc_gap": disparity_map["sex"]["AUROC_gap"],
        "sex_auprc_gap": disparity_map["sex"]["AUPRC_gap"],
        "age_auroc_gap": disparity_map["age"]["AUROC_gap"],
        "age_auprc_gap": disparity_map["age"]["AUPRC_gap"],
        "race_auroc_gap": disparity_map["race"]["AUROC_gap"],
        "race_auprc_gap": disparity_map["race"]["AUPRC_gap"],
    }


def aggregate_seed_rows(seed_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        seed_df.groupby(["model", "setting"], as_index=False)
        .agg(
            test_auroc_mean=("test_auroc", "mean"),
            test_auroc_std=("test_auroc", "std"),
            test_auprc_mean=("test_auprc", "mean"),
            test_auprc_std=("test_auprc", "std"),
            sex_auroc_gap_mean=("sex_auroc_gap", "mean"),
            sex_auroc_gap_std=("sex_auroc_gap", "std"),
            sex_auprc_gap_mean=("sex_auprc_gap", "mean"),
            sex_auprc_gap_std=("sex_auprc_gap", "std"),
            age_auroc_gap_mean=("age_auroc_gap", "mean"),
            age_auroc_gap_std=("age_auroc_gap", "std"),
            age_auprc_gap_mean=("age_auprc_gap", "mean"),
            age_auprc_gap_std=("age_auprc_gap", "std"),
            race_auroc_gap_mean=("race_auroc_gap", "mean"),
            race_auroc_gap_std=("race_auroc_gap", "std"),
            race_auprc_gap_mean=("race_auprc_gap", "mean"),
            race_auprc_gap_std=("race_auprc_gap", "std"),
            n_seeds=("seed", "count"),
        )
    )
    return grouped


def save_model_validation_tables(model: str, seed_df: pd.DataFrame) -> None:
    model_seed_df = seed_df[seed_df["model"] == model].sort_values(["setting", "seed"]).copy()
    model_summary_df = aggregate_seed_rows(model_seed_df).sort_values("setting").copy()

    detail_path = TASK8_FINAL_TABLES_DIR / f"{model}_3seed_validation_detail.csv"
    summary_path = TASK8_FINAL_TABLES_DIR / f"{model}_3seed_validation_summary.csv"
    model_seed_df.to_csv(detail_path, index=False)
    model_summary_df.to_csv(summary_path, index=False)


def save_cross_seed_comparison(seed_df: pd.DataFrame, model: str, settings: list[str], output_stub: str) -> None:
    subset = seed_df[(seed_df["model"] == model) & (seed_df["setting"].isin(settings))].copy()

    overall_df = (
        subset.groupby("setting", as_index=False)
        .agg(
            test_auroc_mean=("test_auroc", "mean"),
            test_auroc_std=("test_auroc", "std"),
            test_auprc_mean=("test_auprc", "mean"),
            test_auprc_std=("test_auprc", "std"),
            n_seeds=("seed", "count"),
        )
    )
    overall_df.to_csv(TASK8_CROSS_SEED_DIR / f"{output_stub}_overall_summary.csv", index=False)

    disparity_rows = []
    for setting in settings:
        setting_df = subset[subset["setting"] == setting]
        for attribute in ("sex", "age", "race"):
            disparity_rows.append(
                {
                    "setting": setting,
                    "attribute": attribute,
                    "AUROC_gap_mean": setting_df[f"{attribute}_auroc_gap"].mean(),
                    "AUROC_gap_std": setting_df[f"{attribute}_auroc_gap"].std(),
                    "AUPRC_gap_mean": setting_df[f"{attribute}_auprc_gap"].mean(),
                    "AUPRC_gap_std": setting_df[f"{attribute}_auprc_gap"].std(),
                    "n_seeds": len(setting_df),
                }
            )

    disparity_df = pd.DataFrame(disparity_rows)
    disparity_df.to_csv(TASK8_CROSS_SEED_DIR / f"{output_stub}_disparity_summary.csv", index=False)


def save_combined_cross_seed_summary(seed_df: pd.DataFrame) -> None:
    combined_summary = (
        seed_df.groupby(["model", "setting"], as_index=False)
        .agg(
            test_auroc_mean=("test_auroc", "mean"),
            test_auprc_mean=("test_auprc", "mean"),
            sex_auroc_gap_mean=("sex_auroc_gap", "mean"),
            sex_auprc_gap_mean=("sex_auprc_gap", "mean"),
            age_auroc_gap_mean=("age_auroc_gap", "mean"),
            age_auprc_gap_mean=("age_auprc_gap", "mean"),
            race_auroc_gap_mean=("race_auroc_gap", "mean"),
            race_auprc_gap_mean=("race_auprc_gap", "mean"),
            n_seeds=("seed", "count"),
        )
    )
    combined_summary.to_csv(TASK8_CROSS_SEED_DIR / "task8_cross_seed_summary.csv", index=False)


def main() -> None:
    ensure_output_dirs()

    seed_rows = []
    for model, settings in MODEL_SETTINGS.items():
        for setting in settings:
            for seed in SEEDS:
                seed_rows.append(build_seed_level_row(model, seed, setting))

    seed_df = pd.DataFrame(seed_rows)

    for model in MODEL_SETTINGS:
        save_model_validation_tables(model, seed_df)

    save_cross_seed_comparison(
        seed_df,
        model="LSTM",
        settings=["original", "train_race_weighted"],
        output_stub="LSTM_train_race_weighted_vs_original",
    )
    save_cross_seed_comparison(
        seed_df,
        model="LSTM",
        settings=["train_race_weighted", "train_race_weighted_varreg001"],
        output_stub="LSTM_train_race_weighted_varreg001_vs_train_race_weighted",
    )
    save_cross_seed_comparison(
        seed_df,
        model="TGTFull",
        settings=["original", "train_race_weighted"],
        output_stub="TGTFull_train_race_weighted_vs_original",
    )
    save_cross_seed_comparison(
        seed_df,
        model="TGTFull",
        settings=["train_race_weighted", "train_race_weighted_varreg005"],
        output_stub="TGTFull_train_race_weighted_varreg005_vs_train_race_weighted",
    )
    save_combined_cross_seed_summary(seed_df)

    print("Saved Task8 final tables and cross-seed summaries.")


if __name__ == "__main__":
    main()
