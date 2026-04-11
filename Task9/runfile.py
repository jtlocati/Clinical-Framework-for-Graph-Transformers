import os
import sys
import json
import time
import pandas
import numpy
import Task9.config as config9
import Task6.config as config6
import Task6.intervention as intervention
import Task9.calibration as calibration
import Task9.eval as eval9
import Task6.eval as eval6
from Task4 import DataPipeline
from Task6 import modelmetrics
import Task9.plots as plots
import pickle

def main():
    total_time = time.time()
    print(f"{config6.BAG} LOADING DATA + TRAINING MODELS {config6.BAG}")
    Rawx, M, y, ids = DataPipeline.run_pipeline()
    all_predictions = {}
    for seed in config6.seed:
        print(f"\n{'='*60} SEED {seed} {'='*60}")
        splits, means, stddev = DataPipeline.prepare_for_seed(Rawx, M, y, ids, seed)
        test_ids = splits["test"]["ids"]
        preds = modelmetrics.CollectMetrics(splits, seed)
        for model, (yt, yp) in preds.items():
            if model not in all_predictions:
                all_predictions[model] = []
            all_predictions[model].append((yt, yp))

    # Average predictions across seeds
    predictions = {}
    for model, pred_list in all_predictions.items():
        avg_prob = numpy.mean([yp for _, yp in pred_list], axis=0)
        y_true = pred_list[0][0]  # labels are the same across seeds
        predictions[model] = (y_true, avg_prob)
    print(f"TEST SET: {len(test_ids):,}")

    CACHE_PATH = os.path.join(config9.OUTPUT_DIR, "T9_predictions_cache.pkl")

    if os.path.exists(CACHE_PATH):
        print("Loading cached predictions...")
        with open(CACHE_PATH, "rb") as f:
            predictions = pickle.load(f)
    else:
        predictions = modelmetrics.CollectMetrics(splits, config6.seed)
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(predictions, f)
        print(f"Models Trained: {list(predictions.keys())}")

    print(f"{config9.BAG} Loading Deomgraphics + Creating Masks {config9.BAG}")
    demographics = eval6.load_demographics()
    masks = eval6.BuildMasks(test_ids, demographics)

    print(f"{config6.BAG} Calebrating subgroups {config6.BAG}")
    calebrated_preds = calibration.CalAllModels(predictions, masks)

    print(f"{config6.BAG} Calebrating All models {config6.BAG}")
    all_results = eval9.EvalAll(calebrated_preds, masks)

    print(f"{config9.BAG} Building Comparison Table {config9.BAG}")
    comparison = eval9.BuildComparTale(all_results)
    print(f"{config9.BAG} TABLE {config9.BAG}")
    print(comparison.to_string(index=False))

    print(f"{config9.BAG} Generating Plots {config9.BAG}")
    plots.generate_all_plots(comparison, all_results, config9.OUTPUT_DIR)

    print(f"\n{config9.BAG} Saving results {config9.BAG}")

    # comparison table
    path = os.path.join(config9.OUTPUT_DIR, "T9_comparison_table.csv")
    comparison.to_csv(path, index=False)
    print(f"  {config9.AG} Comparison table -> {path}")

    # per-model detailed results
    for model_name, method_results in all_results.items():
        safe_model = model_name.replace(" ", "_")
        for method_label, res in method_results.items():
            safe_method = method_label.replace(" ", "_")
            prefix = f"T9_{safe_model}_{safe_method}"

            # subgroup metrics
            mpath = os.path.join(config9.OUTPUT_DIR, f"{prefix}_metrics.csv")
            res["metrics"].to_csv(mpath, index=False)

            # gaps
            gpath = os.path.join(config9.OUTPUT_DIR, f"{prefix}_gaps.csv")
            res["gaps"].to_csv(gpath, index=False)

            # threshold detail
            tpath = os.path.join(config9.OUTPUT_DIR, f"{prefix}_thresholds.csv")
            res["threshold_detail"].to_csv(tpath, index=False)

    print(f"  {config9.AG} Detailed CSVs saved to {config9.OUTPUT_DIR}")

    # JSON summary
    summary = {
        "seed": config6.seed,
        "models": list(predictions.keys()),
        "methods": config9.METHOD_LABELS,
        "comparison": comparison.to_dict(orient="records"),
    }

    # find best post-hoc method per model
    for model_name in predictions.keys():
        model_rows = comparison[comparison["model"] == model_name]
        original_gap = model_rows[
            model_rows["method"] == "Original_GlobalThresh"
        ]["max_auroc_gap"].values
        if len(original_gap):
            original_gap = original_gap[0]
        else:
            original_gap = 0.0

        best_row = model_rows.sort_values("max_auroc_gap").iloc[0]
        summary[f"best_method_{model_name}"] = {
            "method": best_row["method"],
            "original_gap": round(float(original_gap), 4),
            "best_gap": round(float(best_row["max_auroc_gap"]), 4),
            "reduction": round(float(original_gap - best_row["max_auroc_gap"]), 4),
            "auroc": round(float(best_row["overall_auroc"]), 4),
        }

    jpath = os.path.join(config9.OUTPUT_DIR, "T9_summary.json")
    with open(jpath, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  {config9.AG} JSON summary -> {jpath}")

    elapsed = time.time() - total_time
    print(f"\n{config9.BAG}\n  DONE. Total time: {elapsed:.1f}s"
          f"\n  Results saved to: {config9.OUTPUT_DIR}\n{config9.BAG}")


if __name__ == "__main__":
    main()