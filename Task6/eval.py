import numpy
import pandas
from google.cloud import bigquery
from sklearn.metrics import roc_auc_score, average_precision_score
import Task6.config as config
import os

def load_demographics() -> pandas.DataFrame:
    print("loading saved demographics...")
    Demographic = pandas.read_csv(config.DemographicsPath)
    print(f"Loaded {len(Demographic):,} patients from csv")
    return Demographic

#Bool masks for set
def BuildMasks(ids: numpy.ndarray, demographics: pandas.DataFrame) -> dict:
    test_demo = demographics.set_index("stay_id").loc[ids].reset_index()

    masks = {}
    for attrobute, col_name in config.SUBGROUP_COLS.items():
        if col_name not in test_demo.columns:
            print(f"!!!!WARNING: col: {col_name} is not in dempgraphics, Will skip")
            continue
        for val in sorted(test_demo[col_name].dropna().unique()):
            key = (f"{attrobute}_{val}")
            masks[key] = (test_demo[col_name]==val).values
    print(f"{config.AG} subgroup sizes {config.AG}")
    for name, m in masks.items():
        print(f"{name:20s} n={m.sum():,}")
    return masks

#compute metrics per subgroup per model (for evry (model, subgroup) compute AUROC and AUPRC)
def SubgroupMetrics(prediction: dict, masks: dict) -> pandas.DataFrame:
    rows = []
    #overall test
    for model, (y_true, y_prob) in prediction.items():
        rows.append({"model": model, "subgroup": "overall", "attribute": "overall", "n": len(y_true), "auroc": roc_auc_score(y_true, y_prob), "auprc": average_precision_score(y_true, y_prob)})

    #per subgroup
    for subg, mask in masks.items():
        nm = mask.sum()

        #find attrobute in current test, also filter results given posible missing values or too few data in a subgroup
        attrobute = subg.split("_")[0]
        if nm < config.MinGroup:
            rows.append({"model": model, "subgroup": subg, "attribute": attrobute, "n": int(nm),"auroc": numpy.nan, "auprc": numpy.nan})
            continue
        sub_y = y_true
        sub_p = y_prob
        if len(set(sub_y)) <2:
            rows.append({"model": model, "subgroup": subg, "attribute": attrobute, "n": int(nm),"auroc": numpy.nan, "auprc": numpy.nan})
            continue

        rows.append({
            "model": model,
            "subgroup": subg,
            "attribute": attrobute,
            "n": int(nm),
            "auroc": roc_auc_score(sub_y, sub_p),
            "auprc": average_precision_score(sub_y, sub_p),
        })
        return pandas.DataFrame(rows)
    
def FairnessDif(metrics: pandas.DataFrame) -> pandas.DataFrame:
    # drop rows with too few values
    substi = metrics[(metrics["attrobute"] != "overall") & (metrics["auroc"].notna())].copy()
    rows = []

    for (model, attrobute), grp in substi.groupby(["model", "attrobute"]):
        rows.append({ 
            "model": model,
            "attribute": attrobute,
            "auroc_max": grp["auroc"].max(),
            "auroc_min": grp["auroc"].min(),
            "auroc_gap": grp["auroc"].max() - grp["auroc"].min(),
            "auroc_std": grp["auroc"].std(),
            "auprc_max": grp["auprc"].max(),
            "auprc_min": grp["auprc"].min(),
            "auprc_gap": grp["auprc"].max() - grp["auprc"].min(),
            "auprc_std": grp["auprc"].std(),
            })
    return pandas.DataFrame(rows)

def BuildFairTable(metrics: pandas.DataFrame) -> pandas.DataFrame:
    rows = []
    models = metrics["model"].unique()
    for model in models:
        mdf = metrics[metrics["model"] == model]
        entry = {"model": model}

        overall = mdf[mdf["subgroup"] == "overall"]
        if len(overall):
            entry["overall_auroc"] = round(overall["auroc"].values[0], 4)
            entry["overall_auprc"] = round(overall["auprc"].values[0], 4)
        
        #per subgroup
        for _, row in mdf[mdf["subgroup"] != "overall"].iterrows():
            key = f"{row['subgroup']}_auroc"
            entry[key] = round(row["auroc"],4) if pandas.notna(row["auroc"]) else None

        #gaps per attrobute
        for attrobute in config.SUBGROUP_COLS.keys():
            attrobute_rows = [(mdf[mdf["attrobute"] == attrobute]) & (mdf["auroc"].notna())]
            if len(attrobute_rows) >= 2:
                entry[f"{attrobute}_gap"] = round(attrobute_rows["auroc"].max() - attrobute_rows["auroc"].min(), 4)
                entry[f"{attrobute}_std"] = round(attrobute_rows["auroc"].std(), 4)
            else:
                entry[f"{attrobute}_gap"] = None
                entry[f"{attrobute}_std"] = None
        
        rows.append(entry)
        return pandas.DataFrame(rows)
    

