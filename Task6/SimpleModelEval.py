import os
import os
import json
import time
from Task4 import DataPipeline
from Task6 import modelmetrics, eval, plotting, intervention, config

BASELINE_MODELS = ["Logistic Regression", "LightGBM", "Pretrained TST"]

def main():

    print(f"{config.BAG}\nFAIRNESS EVALUATION\n{config.BAG}")
    TotalTime = time.time()

    print(f"{config.AG} Loading Data {config.AG}")
    Rawx, M, y, ids = DataPipeline.run_pipeline()
    splits, means, studiodev = DataPipeline.prepare_for_seed(Rawx, M, y, ids, config.seed)
    test_ids = splits["test"]["ids"]
    print(f"{config.AG}TEST SET: {len(test_ids):,} patients{config.AG}")

    print(f"{config.BAG}TRAINING MODELS {config.AG}")
    predictions = modelmetrics.CollectMetricsMultiSeed(Rawx, M, y, ids, config.seed)

    print(f"{config.AG}COLLECTING DEMOGRAPHICS{config.AG}")
    demographics = eval.load_demographics()
    masks = eval.BuildMasks(test_ids, demographics)

    print(f"{config.AG} GETTING SUBGROUP METRICS{config.AG}")
    metrics = eval.SubgroupMetrics(predictions, masks)
    gap = eval.FairnessDif(metrics)
    fairnessTable = eval.BuildFairTable(metrics)

    print(f"{config.BAG} AUROC permodel{config.BAG}")
    OverallMetrics = metrics[metrics["subgroup"] == "overall"][["model", "auroc", "auprc"]]
    print(OverallMetrics.to_string(index=False))

    print(f"{config.BAG}FAIRNESS GAPS :{config.BAG}\n{gap[["model", "attribute", "auroc_gap", "auprc_gap", "auroc_std"]].to_string(index=False)}")

    print(f"{config.BAG}GENERATING PLOTS{config.BAG}")
    #plotting.AllPlot(metrics, gap)

    #best = intervention.FindBestModel(metrics)

    #Find best TGT
    """interventionResults = {}
    for key in ["best_baseline", "best_tgt"]:
        modelName = best[key]
        YTrue, YProb = predictions[modelName]
        result = intervention.ExecuteInterns(modelName, YTrue,YProb, masks)
        interventionResults[modelName] = result"""

    #Save outputs:
    print(f"{config.BAG}SAVING RESULTS{config.BAG}")
        # full subgroup metrics
    path = os.path.join(config.RESULTS_DIRECTORY, "T6_subgroup_metrics.csv")
    metrics.to_csv(path, index=False)
    print(f"{config.AG}Subgroup metrics -> {path}{config.AG}")

    # fairness gaps
    path = os.path.join(config.RESULTS_DIRECTORY, "T6_fairness_gaps.csv")
    gap.to_csv(path, index=False)
    print(f"{config.AG}Fairness gaps -> {path}{config.AG}")

    # wide fairness table
    path = os.path.join(config.RESULTS_DIRECTORY, "T6_fairness_table.csv")
    fairnessTable.to_csv(path, index=False)
    print(f"{config.AG}Fairness table {path}{config.AG}")

    # intervention results
    """for model_name, res in interventionResults.items():
        save = model_name.replace(" ", "_")
        for int_name, df in res.items():
            path = os.path.join(config.RESULTS_DIRECTORY, f"T6_intervention_{save}_{int_name}.csv")
            df.to_csv(path, index=False)
            print(f"{config.AG}{int_name:25s} ({model_name}) -> {path}{config.AG}")"""

    # JSON summary
    json_data = {
        "seed": config.seed,
        "models": config.BASELINE_MODELS,
        "overall": OverallMetrics.to_dict(orient="records"),
        "gaps": gap.to_dict(orient="records"),
        #"best_baseline": best["best_baseline"],
        #"best_tgt": best["best_tgt"],
    }
    path = os.path.join(config.RESULTS_DIRECTORY, "T6_summary_SimpleModel.json")
    with open(path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"{config.AG}JSON summary -> {path} {config.AG}")

    elap = time.time() - TotalTime
    print(f"{config.BAG}\nDONE.  |   TIME ELAPSED: {elap}\nRESULTS SAVED TO: {config.RESULTS_DIRECTORY}")

if __name__ == "__main__":
    main()
