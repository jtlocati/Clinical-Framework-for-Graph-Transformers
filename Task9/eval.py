from Task6 import config as config6
from Task6 import eval
import Task6.intervention as intervention
import os
import numpy
import pandas
from Task9 import config as config9
from sklearn.metrics import roc_auc_score, average_precision_score


def SubgroupMetrics(y_true, y_prob, masks, model="model") -> pandas.DataFrame:
    rows = []
    rows.append({
        "model": model,
        "subgroup": "overall",
        "attribute": "overall",
        "n": len(y_true),
        "auroc": roc_auc_score(y_true, y_prob),
        "auprc": average_precision_score(y_true, y_prob)
    })

    for subg, mask in masks.items():
        num = mask.sum()
        attribute = subg.split("_")[0]
        if num < config6.MinGroup:
            rows.append({"model": model, "subgroup": subg, "attribute": attribute, "n": int(num), "auroc": numpy.nan, "auprc": numpy.nan })
            continue
        sub_y = y_true[mask]
        sub_p = y_prob[mask]
        if len(set(sub_y)) < 2:
            rows.append({"model": model, "subgroup": subg, "attribute": attribute, "n": int(num), "auroc": numpy.nan, "auprc": numpy.nan})
            continue
        rows.append({"model": model, "subgroup": subg, "attribute": attribute, "n": int(num), "auroc": roc_auc_score(sub_y, sub_p), "auprc": average_precision_score(sub_y, sub_p)})

    return pandas.DataFrame(rows)

#fairness dif -> Task6.eval.FairnessDif + argument param => not None


def SingleEval(y_true, y_prob, mask, method):
    mdf = SubgroupMetrics (y_true, y_prob, mask, model=method)
    gdf = eval.FairnessDif(mdf, "T9")
    return mdf, gdf

def EvalAll(calibrated: dict, masks: dict) -> dict:
    all_results = {}
    model_names = list(calibrated["original"].keys())

    for model in model_names:
        print(f"{config9.BAG}Evaluating post-hoc methods for {model} {config9.BAG}")
        model_results = {}

        for cal_method in ["original"] + config9.CAL_METHODS:
            y_true, y_prob = calibrated[cal_method][model]
            GlobalThresh = intervention.BestF1Thresh(y_true, y_prob)
            GroupThresh = intervention.PerSGThresh(y_true, y_prob, masks, GlobalThresh)

            for thresh_type in ["GlobalThresh", "PerGroupThresh"]:
                label = f"{cal_method.capitalize()}_{thresh_type}"
                if cal_method == "original":
                    label = f"Original_{thresh_type}"
                print(f"{config9.AG} {label} {config9.AG}")

                mdf, gdf = SingleEval(y_true, y_prob, masks, label)

                tdf = intervention.ApplyThresh(y_true, y_prob, masks, GlobalThresh, GroupThresh)

                overall_row = mdf[mdf["subgroup"] == "overall"].iloc[0]
                max_auroc_gap = gdf["auroc_gap"].max() if len(gdf) else 0.0
                max_auprc_gap = gdf["auprc_gap"].max() if len(gdf) else 0.0

                if thresh_type == "GlobalThresh":
                    tpr_gap = (tdf["tpr_global"].max() - tdf["tpr_global"].min()
                               if len(tdf) >= 2 else 0.0)
                else:
                    tpr_gap = (tdf["tpr_tuned"].max() - tdf["tpr_tuned"].min()
                               if len(tdf) >= 2 else 0.0)

                model_results[label] = {
                    "metrics": mdf,
                    "gaps": gdf,
                    "threshold_detail": tdf,
                    "overall_auroc": round(float(overall_row["auroc"]), 4),
                    "overall_auprc": round(float(overall_row["auprc"]), 4),
                    "max_auroc_gap": round(float(max_auroc_gap), 4),
                    "max_auprc_gap": round(float(max_auprc_gap), 4),
                    "tpr_gap": round(float(tpr_gap), 4),
                }

        all_results[model] = model_results

    return all_results

def BuildComparTale(all_results: dict) -> pandas.DataFrame:
    rows = []
    for model_name, method_results in all_results.items():
        for method_label, res in method_results.items():
            rows.append({
                "model": model_name,
                "method": method_label,
                "overall_auroc": res["overall_auroc"],
                "overall_auprc": res["overall_auprc"],
                "max_auroc_gap": res["max_auroc_gap"],
                "max_auprc_gap": res["max_auprc_gap"],
                "tpr_gap": res["tpr_gap"],
            })
    return pandas.DataFrame(rows)
