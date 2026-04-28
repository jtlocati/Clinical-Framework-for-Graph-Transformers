import os
import numpy
import pandas
from Task4 import DataPipeline
from Task6 import modelmetrics
from Task6 import eval as eval6
import Task6.config as config6
import Task9.config as config9
import Task9.calibration as calibration
import Task6.intervention as intervention


def compute_tpr_gap(y_true, y_prob, masks, global_thresh, group_thresh, thresh_type):
    tdf = intervention.ApplyThresh(y_true, y_prob, masks, global_thresh, group_thresh)
    if len(tdf) < 2:
        return 0.0
    if thresh_type == "GlobalThresh":
        return float(tdf["tpr_global"].max() - tdf["tpr_global"].min())
    else:
        return float(tdf["tpr_tuned"].max() - tdf["tpr_tuned"].min())


def run_seed(seed, Rawx, M, y, ids):
    print(f"\n{'='*60} SEED {seed} {'='*60}")
    splits, means, stddev = DataPipeline.prepare_for_seed(Rawx, M, y, ids, seed)
    test_ids = splits["test"]["ids"]

    preds = modelmetrics.CollectMetrics(splits, seed)

    demographics = eval6.load_demographics()
    masks = eval6.BuildMasks(test_ids, demographics)

    predictions = {}
    for model, (yt, yp) in preds.items():
        predictions[model] = (yt, yp)

    calibrated = calibration.CalAllModels(predictions, masks)

    seed_results = {}
    for model in predictions.keys():
        for cal_method in ["original"] + config9.CAL_METHODS:
            y_true, y_prob = calibrated[cal_method][model]
            global_thresh = intervention.BestF1Thresh(y_true, y_prob)
            group_thresh = intervention.PerSGThresh(y_true, y_prob, masks, global_thresh)

            for thresh_type in ["GlobalThresh", "PerGroupThresh"]:
                if cal_method == "original":
                    label = f"Original_{thresh_type}"
                else:
                    label = f"{cal_method.capitalize()}_{thresh_type}"

                tpr_gap = compute_tpr_gap(
                    y_true, y_prob, masks,
                    global_thresh, group_thresh, thresh_type
                )

                key = (model, label)
                seed_results[key] = tpr_gap
                print(f"  {model} | {label} | TPR gap = {tpr_gap:.4f}")

    return seed_results


def main():
    print(f"{config9.BAG} LOADING DATA {config9.BAG}")
    Rawx, M, y, ids = DataPipeline.run_pipeline()

    all_seed_results = []
    for seed in config6.seed:
        seed_results = run_seed(seed, Rawx, M, y, ids)
        all_seed_results.append(seed_results)

    all_keys = sorted(all_seed_results[0].keys())
    rows = []
    for key in all_keys:
        model, method = key
        values = [sr[key] for sr in all_seed_results]
        rows.append({
            "model": model,
            "method": method,
            "tpr_gap_mean": round(numpy.mean(values), 4),
            "tpr_gap_std": round(numpy.std(values, ddof=1), 4),
            "seed_values": [round(v, 4) for v in values],
        })

    df = pandas.DataFrame(rows)

    print(f"\n{'='*60}")
    print("TPR GAP: MEAN ± STD ACROSS SEEDS")
    print(f"{'='*60}")
    for _, row in df.iterrows():
        print(f"  {row['model']:20s} | {row['method']:28s} | "
              f"{row['tpr_gap_mean']:.4f} ± {row['tpr_gap_std']:.4f}  "
              f"({row['seed_values']})")

    # Save
    out_path = os.path.join(config9.OUTPUT_DIR, "T9_tpr_gap_per_seed.csv")
    df.drop(columns=["seed_values"]).to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    detail_rows = []
    for i, seed in enumerate(config6.seed):
        for key in all_keys:
            model, method = key
            detail_rows.append({
                "seed": seed,
                "model": model,
                "method": method,
                "tpr_gap": round(all_seed_results[i][key], 4),
            })
    detail_df = pandas.DataFrame(detail_rows)
    detail_path = os.path.join(config9.OUTPUT_DIR, "T9_tpr_gap_per_seed_detail.csv")
    detail_df.to_csv(detail_path, index=False)
    print(f"Saved per-seed detail to {detail_path}")


if __name__ == "__main__":
    main()