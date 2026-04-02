"""
run_all.py — Master script: run all models × all seeds, report results
========================================================================
Usage:
    python run_all.py                    # run everything
    python run_all.py --models LSTM PatchTST   # run subset
    python run_all.py --device cpu       # force CPU
    python run_all.py --refresh-data     # requery BigQuery
"""

import argparse, json, os, time
import numpy as np
import pandas as pd
import torch

import config
from DataPipeline import run_pipeline, prepare_for_seed
from experiment import run_single, ALL_MODELS


def set_seed(seed: int):
    #Set random seed for reproducibility across all libraries.
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="ICU Mortality Benchmark")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS,
                        help="Which models to run")
    parser.add_argument("--seeds", nargs="+", type=int, default=config.Seeds,
                        help="Random seeds")
    parser.add_argument("--device", default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--refresh-data", action="store_true",
                        help="Force re‑query from BigQuery")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    # 1. Load / extract data 
    print("=" * 70)
    print("  ICU MORTALITY BENCHMARK — MIMIC‑IV (48 h, in‑hospital mortality)")
    print("=" * 70)
    X_raw, M, y, ids = run_pipeline(force_refresh=args.refresh_data)
    print(f"\nDataset: {X_raw.shape[0]:,} patients  |  "
          f"mortality rate: {y.mean():.3f}  |  "
          f"features: {config.NumberOfFeatures}")

    # 2. Run experiments 
    # results[model_name][seed] = {"val": {...}, "test": {...}}
    results = {m: {} for m in args.models}

    for seed in args.seeds:
        print(f"\n{'─' * 70}")
        print(f"  SEED {seed}")
        print(f"{'─' * 70}")
        set_seed(seed)

        # prepare data for this seed (split + normalise)
        splits, means, stds = prepare_for_seed(X_raw, M, y, ids, seed)

        for model_name in args.models:
            set_seed(seed)    # reset seed before each model
            result = run_single(model_name, splits, seed, device)
            results[model_name][seed] = result
            if "test_outputs" in result:
                pred_df = pd.DataFrame({
                    "stay_id": result["test_outputs"]["ids"],
                    "label": result["test_outputs"]["y_true"],
                    "pred_score": result["test_outputs"]["y_prob"],
                })
                pred_path = os.path.join(
                    config.RESULTS_DIECTORY,
                    f"{model_name}_seed{seed}_test_predictions.csv"
                )
                pred_df.to_csv(pred_path, index=False)
                print(f"  Saved test predictions to {pred_path}")

                del result["test_outputs"]
    #3. Aggregate and report 
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)

    rows = []
    for model_name in args.models:
        aurocs = [results[model_name][s]["test"]["auroc"] for s in args.seeds]
        auprcs = [results[model_name][s]["test"]["auprc"] for s in args.seeds]
        row = {
            "Model": model_name,
            "AUROC_mean": np.mean(aurocs),
            "AUROC_std":  np.std(aurocs),
            "AUPRC_mean": np.mean(auprcs),
            "AUPRC_std":  np.std(auprcs),
        }
        rows.append(row)
        print(f"  {model_name:<22s}  "
              f"AUROC = {row['AUROC_mean']:.4f} ± {row['AUROC_std']:.4f}  |  "
              f"AUPRC = {row['AUPRC_mean']:.4f} ± {row['AUPRC_std']:.4f}")

    # 4. Save results
    df = pd.DataFrame(rows)
    csv_path = os.path.join(config.RESULTS_DIECTORY, "benchmark_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSummary saved to {csv_path}")

    # also save full per‑seed results as JSON under Task4\results
    json_path = os.path.join(config.RESULTS_DIECTORY, "benchmark_results_full.json")
    # convert numpy types for JSON interpritatin
    serializable = {}
    for model_name, seed_dict in results.items():
        serializable[model_name] = {}
        for seed, metrics in seed_dict.items():
            serializable[model_name][str(seed)] = {
                split: {k: float(v) for k, v in m.items()}
                for split, m in metrics.items()
            }
    with open(json_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f" Full results saved to {json_path}")

    # 5 Print LaTeX table because why not
    print("\n  LaTeX table:\n")
    print(r"  \begin{tabular}{lcc}")
    print(r"  \toprule")
    print(r"  Model & AUROC & AUPRC \\")
    print(r"  \midrule")
    for row in rows:
        print(f"  {row['Model']} & "
              f"${row['AUROC_mean']:.4f} \\pm {row['AUROC_std']:.4f}$ & "
              f"${row['AUPRC_mean']:.4f} \\pm {row['AUPRC_std']:.4f}$ \\\\")
    print(r"  \bottomrule")
    print(r"  \end{tabular}")

    return results


if __name__ == "__main__":
    main()
