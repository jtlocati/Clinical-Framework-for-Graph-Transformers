import os
import sys
import time
import numpy
import pandas
import torch
from pathlib import Path

from Task4 import DataPipeline
from Task4.experiment import RunSingleWithPred
import Task6.modelmetrics
import Task6.config

REPOROOT = Path(__file__).resolve().parents[1]

SEEDS = Task6.config.ModelSeed
SPLIT_SEED = Task6.config.SPLIT_SEED
#Model, Name, TGT?, tgt_varient
MODELS = {("LSTM", "LSTM", False, None), ("TGTTemporal", "temporal_only", True, "temporal_only"), ("TGTClinical", "clinical_only", True, "clinical_only"), ("TGTFull", "full_graph", True, "full_graph")}

OUTPUT_DIRECTORY = os.path.join(REPOROOT, "Task10", "results")
EXSPECT = 6182

def load_demographics():
    demo = pandas.read_csv(Task6.config.DemographicsPath)
    demo.columns = demo.columns.str.strip()
    demo = demo[["stay_id", "sex", "age_group", "race"]].copy()
    demo = demo.drop_duplicates(subset="stay_id")
    return demo.set_index("stay_id")


def save_predictions(model_name, seed, test_ids, y_true, y_prob, demo_df):
    pred_df = pandas.DataFrame({ "stay_id": test_ids.astype(int),"y_true": y_true.astype(int), "y_score": y_prob})

    pred_df = pred_df.merge(demo_df, left_on="stay_id", right_index=True, how="left")

    missing_demo = pred_df["sex"].isna().sum()
    if missing_demo > 0:
        print(f"WARNING: {missing_demo} patients missing demographics")

    filename = f"{model_name}_seed{seed}_test_predictions.csv"
    filepath = os.path.join(OUTPUT_DIRECTORY, filename)
    pred_df.to_csv(filepath, index=False)
    print(f"Saved: {filepath} ({len(pred_df)} rows)")
    return filepath


def train_and_predict_lstm(splits, seed):
    _, y_true, y_prob = RunSingleWithPred("LSTM", splits, seed, device="cuda")
    return y_true, y_prob


def train_and_predict_tgt(variant, splits, seed):
    y_true, y_prob = Task6.modelmetrics.TAPTGT(variant, splits, seed, device="cuda")
    return y_true, y_prob



def main():
    print("=" * 60)
    print("Task 10 — Generate Per-Patient Prediction CSVs")
    print("=" * 60)
    total_start = time.time()

    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    print("\n[1/4] Loading cached data...")
    X_raw, M, y, ids = DataPipeline.run_pipeline()
    print(f"Total cohort: {len(y):,} patients")
    print(f"Positive rate: {y.mean():.3f}")

    print(f"\n[2/4] Preparing data splits (split_seed={SPLIT_SEED})...")
    splits, means, stds = DataPipeline.prepare_for_seed(X_raw, M, y, ids, SPLIT_SEED)
    test_ids = splits["test"]["ids"]
    test_n = len(test_ids)
    print(f"Test set: {test_n:,} patients")
    assert test_n == EXSPECT, \
        f"Expected {EXSPECT} test patients, got {test_n}"

    print("\n[3/4] Loading demographics...")
    demo_df = load_demographics()
    print(f"Demographics loaded: {len(demo_df):,} patients")

    test_in_demo = numpy.isin(test_ids, demo_df.index.values)
    coverage = test_in_demo.mean()
    print(f"Test set coverage: {coverage:.1%}")

    print(f"\n[4/4] Training {len(MODELS)} models × {len(SEEDS)} seeds = "
          f"{len(MODELS) * len(SEEDS)} runs...")

    results_log = []

    for model_name, model_type, is_tgt, tgt_variant in MODELS:
        for seed in SEEDS:
            run_label = f"{model_name}_seed{seed}"

            out_path = os.path.join(OUTPUT_DIRECTORY, f"{run_label}_test_predictions.csv")
            if os.path.exists(out_path):
                existing = pandas.read_csv(out_path)
                if len(existing) == EXSPECT:
                    print(f"\nSKIP {run_label} — already exists with {len(existing)} rows")
                    results_log.append({
                        "model": model_name, "seed": seed,
                        "status": "skipped", "n_test": len(existing)
                    })
                    continue

            print(f"\n{'─' * 60}")
            print(f"Training: {run_label}")
            print(f"{'─' * 60}")
            run_start = time.time()

            try:
                torch.manual_seed(seed)
                numpy.random.seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

                if is_tgt:
                    y_true, y_prob = train_and_predict_tgt(tgt_variant, splits, seed)
                else:
                    y_true, y_prob = train_and_predict_lstm(splits, seed)

                assert len(y_true) == EXSPECT, \
                    f"{run_label}: got {len(y_true)} predictions, expected {EXSPECT}"

                # Save
                filepath = save_predictions(
                    model_name, seed, test_ids, y_true, y_prob, demo_df
                )

                elapsed = time.time() - run_start
                print(f"Completed in {elapsed:.1f}s")

                results_log.append({
                    "model": model_name, "seed": seed,
                    "status": "success", "n_test": len(y_true),
                    "elapsed_s": round(elapsed, 1)
                })

            except Exception as e:
                elapsed = time.time() - run_start
                print(f"FAILED after {elapsed:.1f}s: {e}")
                results_log.append({
                    "model": model_name, "seed": seed,
                    "status": f"FAILED: {e}", "n_test": 0
                })

    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    log_df = pandas.DataFrame(results_log)
    print(log_df.to_string(index=False))

    total_elapsed = time.time() - total_start
    print(f"\nTotal time: {total_elapsed / 60:.1f} minutes")

    print(f"\nVerifying outputs in {OUTPUT_DIRECTORY}:")
    all_ok = True
    for model_name, _, _, _ in MODELS:
        for seed in SEEDS:
            fp = os.path.join(OUTPUT_DIRECTORY, f"{model_name}_seed{seed}_test_predictions.csv")
            if os.path.exists(fp):
                df = pandas.read_csv(fp)
                n = len(df)
                status = "OK" if n == EXSPECT else f"WRONG SIZE ({n})"
                print(f"{model_name}_seed{seed}: {n} rows — {status}")
                if n != EXSPECT:
                    all_ok = False
            else:
                print(f"{model_name}_seed{seed}: MISSING")
                all_ok = False

    if all_ok:
        print("\nAll 12 prediction files verified successfully.")
    else:
        print("\nWARNING: Some files are missing or have wrong sizes.")

    log_path = os.path.join(OUTPUT_DIRECTORY, "generation_log.csv")
    log_df.to_csv(log_path, index=False)
    print(f"Run log saved to: {log_path}")


if __name__ == "__main__":
    main()
