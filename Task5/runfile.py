#Run task 5, built from params and connfigs set in ClinGroups, Train, Models, and Graph

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import time
import numpy
import pandas
import matplotlib
import matplotlib.pyplot as pyplot 
from Task4 import DataPipeline
import ClinGroups
import Train

matplotlib.use("Agg")
#Grab attention for important results diuring training
AG = "------"
BAG = "==================================="

RESULTS_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIRECTORY, exist_ok=True)

#Format of two rows and three cols: row0=loss, row1= valAUROC
def Convergence(Alogs: dict):
    figure, axi = pyplot.subplot(2,3, figsize=(16,9))
    figure.suptitle(f"{AG}Tempporal Graphical Transformer convergence{AG}", fontsize=14)

    for col, var in enumerate(ClinGroups.GRAPH_TYPES):
        for s in ClinGroups.SEEDS:
            key = f"{var}_seed{s}"
            if key not in Alogs:
                continue
            log = Alogs[key]
            label = f"seed{s}"
            axi[0, col].plot(log["epoch"], log["train_loss"], label=label, alpha=0.8)
            axi[1, col].plot(log["epoch"], log["val_auroc"], label=label, alpha=0.8)

        # formatting
        pretty = var.replace("_", " ").title()
        axi[0, col].set_title(pretty)
        axi[0, col].set_xlabel("Epoch")
        axi[0, col].set_ylabel("Train Loss")
        axi[0, col].legend(fontsize=8)
        axi[0, col].grid(True, alpha=0.3)

        axi[1, col].set_xlabel("Epoch")
        axi[1, col].set_ylabel("Val AUROC")
        axi[1, col].legend(fontsize=8)
        axi[1, col].grid(True, alpha=0.3)

    pyplot.tight_layout()
    path = os.path.join(RESULTS_DIRECTORY, "T5_convergence.png")
    pyplot.savefig(path, dpi=150)
    pyplot.close()
    print(f"\n{AG}Convergence plot saved TO ROOT: {path}{AG}")

#important outputs & whatnot
def main():
    print(f"{AG}\nTEMPORAL GRAPH TRANSFORMER BENCHMARK\n{AG}")
    print(f"{AG}Loading Cached Data...{AG}")
    RawX, M, y, ids = DataPipeline.run_pipeline()
    print(f"{BAG}\nCohort: {RawX.shape[0]:,} patients  | {RawX.shape[1]}h  |  {RawX.shape[2]} features\n{BAG}")

    Results = []
    Logs = {}
    total_t0 = time.time()

    for s in ClinGroups.SEEDS:
        print(f"{AG}SEED: {s}{AG}")
        splits, means, stds = DataPipeline.prepare_for_seed(RawX, M, y, ids, s)

        for var in ClinGroups.GRAPH_TYPES:
            print(f"\n %%% {var} -> SEED: {s}")
            t0 = time.time()

            result, log = Train.FullModel(var, splits, s)

            TotTime = time.time - t0
            print(f"{BAG} {var} DONE.{BAG}\n test AUROC:    {result["test"]["auroc"]:.4f}\nAUPRC:   {result["test"]["auprc"]:.4f}")

            Results.append({
                "variant":    var,
                "seed":       s,
                "test_auroc": round(result["test"]["auroc"], 4),
                "test_auprc": round(result["test"]["auprc"], 4),
                "val_auroc":  round(result["val"]["auroc"], 4),
                "val_auprc":  round(result["val"]["auprc"], 4),
                "time_s":     round(TotTime, 1),
            })
            Logs[f"{var}_seed{s}"] = log

    #Final Test results and data routeing.
    total_elapsed = time.time() - total_t0
    print(f"\n  Total wall time: {total_elapsed / 60:.1f} min")

    df = pandas.DataFrame(Results)


    print(f"{BAG}\nRESULTS SUMMARY  (mean PM std over 3 seeds)\n{BAG}")

    summary_rows = []
    for variant in ClinGroups.GRAPH_TYPES:
        vdf = df[df["variant"] == variant]
        summary_rows.append({
            "Variant":    variant,
            "AUROC_mean": vdf["test_auroc"].mean(),
            "AUROC_std":  vdf["test_auroc"].std(),
            "AUPRC_mean": vdf["test_auprc"].mean(),
            "AUPRC_std":  vdf["test_auprc"].std(),
        })

    summary = pandas.DataFrame(summary_rows)
    summary["AUROC"] = summary.apply(lambda r: f"{r['AUROC_mean']:.4f} PM {r['AUROC_std']:.4f}", axis=1)
    summary["AUPRC"] = summary.apply(lambda r: f"{r['AUPRC_mean']:.4f} PM {r['AUPRC_std']:.4f}", axis=1)

    print(summary[["Variant", "AUROC", "AUPRC"]].to_string(index=False))


    #ALL OF THE FOLLOWING WILL SAVE TO 'Task5\{specifyed path}'
    # Saves per-run 
    csv_path = os.path.join(RESULTS_DIRECTORY, "T5_all_runs.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Per-run CSV -> {csv_path}")

    # Summary
    sum_path = os.path.join(RESULTS_DIRECTORY, "T5_summary.csv")
    summary.to_csv(sum_path, index=False)
    print(f"  Summary CSV -> {sum_path}")

    # JSON (machine-readable)
    json_path = os.path.join(RESULTS_DIRECTORY, "T5_all_runs.json")
    with open(json_path, "w") as f:
        json.dump(Results, f, indent=2)
    print(f"  JSON -> {json_path}")

    #CANCEL out to save on runtime and space 
    """
    # Training logs (for later analysis)
    logs_path = os.path.join(RESULTS_DIRECTORY, "T5_training_logs.json")
    with open(logs_path, "w") as f:
        json.dump(Logs, f, indent=2)
    print(f"  Training logs -> {logs_path}")
    """

    
    Convergence(Logs)

 
    print(F"{BAG}TASK FIVE DONE.{BAG}")


if __name__ == "__main__":
    main()