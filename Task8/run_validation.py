from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK4_DIR = REPO_ROOT / "Task4"
TASK6_SCRIPT = REPO_ROOT / "Task6" / "compute_fairness_metrics.py"
TASK8_RESULTS_DIR = REPO_ROOT / "Task8" / "results"
RAW_PREDICTIONS_DIR = TASK8_RESULTS_DIR / "raw_predictions"
METRICS_DIR = TASK8_RESULTS_DIR / "metrics"

sys.path.insert(0, str(TASK4_DIR))

import config  # noqa: E402
from DataPipeline import prepare_for_seed, run_pipeline  # noqa: E402
from experiment import run_single  # noqa: E402


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def format_intervention_name(
    loss_weighting_strategy: str | None,
    subgroup_var_reg_strength: float,
) -> str:
    if not loss_weighting_strategy:
        return "original"

    intervention = loss_weighting_strategy
    if subgroup_var_reg_strength > 0.0:
        intervention = f"{intervention}_varreg{str(subgroup_var_reg_strength).replace('.', '')}"
    return intervention


def ensure_output_dirs() -> None:
    for output_dir in (TASK8_RESULTS_DIR, RAW_PREDICTIONS_DIR, METRICS_DIR):
        output_dir.mkdir(parents=True, exist_ok=True)


def save_predictions(
    model: str,
    seed: int,
    intervention: str,
    result: dict,
) -> Path:
    pred_df = pd.DataFrame(
        {
            "stay_id": result["test_outputs"]["ids"],
            "label": result["test_outputs"]["y_true"],
            "pred_score": result["test_outputs"]["y_prob"],
        }
    )
    pred_path = RAW_PREDICTIONS_DIR / f"{model}_seed{seed}_{intervention}_test_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    return pred_path


def save_metrics(
    model: str,
    seed: int,
    intervention: str,
    result: dict,
) -> Path:
    metrics_df = pd.DataFrame(
        [
            {
                "model": model,
                "seed": seed,
                "intervention": intervention,
                "val_auroc": result["val"]["auroc"],
                "val_auprc": result["val"]["auprc"],
                "test_auroc": result["test"]["auroc"],
                "test_auprc": result["test"]["auprc"],
            }
        ]
    )
    metrics_path = METRICS_DIR / f"{model}_seed{seed}_{intervention}_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    return metrics_path


def compute_fairness_outputs(pred_path: Path, model: str, seed: int, intervention: str) -> None:
    fairness_model_name = model if intervention == "original" else f"{model}_{intervention}"
    cmd = [
        sys.executable,
        str(TASK6_SCRIPT),
        "--pred-file",
        str(pred_path),
        "--model-name",
        fairness_model_name,
        "--seed",
        str(seed),
    ]
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single Task8 validation experiment.")
    parser.add_argument("--model", required=True, choices=["LSTM", "TGTFull"])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--sampling-strategy", default=None)
    parser.add_argument("--max-sampling-weight", type=float, default=10.0)
    parser.add_argument("--loss-weighting-strategy", default="train_race_weighted")
    parser.add_argument("--max-loss-weight", type=float, default=10.0)
    parser.add_argument("--subgroup-var-reg-strength", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dirs()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    intervention = format_intervention_name(
        args.loss_weighting_strategy,
        args.subgroup_var_reg_strength,
    )
    print(f"Running Task8 validation: model={args.model} seed={args.seed} intervention={intervention}")

    set_seed(args.seed)
    X_raw, M, y, ids = run_pipeline(force_refresh=args.refresh_data)
    splits, _, _ = prepare_for_seed(X_raw, M, y, ids, args.seed)

    result = run_single(
        args.model,
        splits,
        args.seed,
        device=device,
        sampling_strategy=args.sampling_strategy,
        max_sampling_weight=args.max_sampling_weight,
        loss_weighting_strategy=args.loss_weighting_strategy,
        max_loss_weight=args.max_loss_weight,
        subgroup_var_reg_strength=args.subgroup_var_reg_strength,
    )

    pred_path = save_predictions(args.model, args.seed, intervention, result)
    metrics_path = save_metrics(args.model, args.seed, intervention, result)
    compute_fairness_outputs(pred_path, args.model, args.seed, intervention)

    print(f"Saved predictions: {pred_path}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
