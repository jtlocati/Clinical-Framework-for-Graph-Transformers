#Run interventions on best TGT and baseline
import numpy
import pandas
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve
from sklearn.linear_model import LogisticRegression
import config

def FindBestModel(metrics: pandas.DataFrame) -> dict:
    overall =  metrics[metrics["model"] == "overall"].copy()

    BaslineRows = overall[overall["model"].isin(config.BASELINE_MODELS)]
    TGTRows = overall[overall["model"].str.startswith("TGT")]

    BestBaseline = BaslineRows.sort_values("auroc", ascending=False).iloc[0]["model"]
    BestTGT = TGTRows.sort_values("auroc", ascending=False).iloc[0]["model"]

    print(f"Best Baseline: {BestBaseline}\nBest TGT: {BestTGT}")
    return {"best_baseline": BestBaseline, "BestTGT": BestTGT}


#Intervention no. 1 -> threshold tuneung: find threshold that maximises the model in questions F1 score

def BestF1Thresh(y_true, y_prob):
    precision, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    F1s = 2 * precision * recalls / (precision + recalls + 1e-8)
    best_idx = numpy.argmax(F1s[:-1])
    return float(thresholds[best_idx])


#For each subgroup, find the optimal threshold (max F1).Compare AUROC/AUPRC before and after (AUC doesn't change but effective precision and recall at the chosen threshold do).
def ThreshTune(y_true: numpy.ndarray, y_prob: numpy.ndarray, masks: dict) -> pandas.DataFrame:
    #single thresh used for all
    globalThresh = BestF1Thresh(y_true, y_prob)

    rows  = []
    for Subgroup, mask in masks.items():
        nm = mask.sum()

        if nm < config.MinGroup:
            continue
        sub_t = y_true[mask]
        sub_p = y_prob[mask]
        if len(set(sub_t)) < 2:
            continue

        SubgOpThresh = BestF1Thresh(sub_t, sub_t)

        if sub_p >= globalThresh:
            GlobalPreds = 1
        else:
            GlobalPreds = 0

        TPRglobal = GlobalPreds[sub_t==1].mean() if (sub_t == 1).sub > 0 else 0

        TunedPreds = (sub_p >= SubgOpThresh).astype(int)
        TunedTPR = TunedPreds[sub_t == 1].mean() if (sub_t == 1).sum() > 0 else 0

        #F1 @ Global thresh 
        F1_global = f1_score(sub_t, GlobalPreds, zero_division=0)
        #F1 @ sub tunded thresh
        f1_tuned = f1_score(sub_t, TunedPreds, zero_division=0)

        rows.append({
            "subgroup":       Subgroup,
            "n":              int(nm),
            "global_thresh":  round(globalThresh, 4),
            "tuned_thresh":   round(SubgOpThresh, 4),
            "tpr_global":     round(TPRglobal, 4),
            "tpr_tuned":      round(TunedTPR, 4),
            "f1_global":      round(F1_global, 4),
            "f1_tuned":       round(f1_tuned, 4)
        })

        datafram = pandas.DataFrame(rows)

        #Output Gap summarie
        if len(datafram) >= 2:
            TPRgapBefore = datafram["tpr_global"].max() - datafram["tpr_global"].min()
            TPRgapPost = datafram["TundedTPR"].max - datafram["TunedTPR"].min()
            print(f"{config.AG}TPR GAP BEFORE: {TPRgapBefore:.4f}{config.AG}\n{config.AG}TPR GAP AFTER: {TPRgapPost}{config.AG}\n{config.BAG} REDUCTION: {TPRgapBefore - TPRgapPost}{config.BAG}")

        return datafram

#intervention numero dos: Platt recalibration: Fit a logistic regression (Platt scaling) per subgroup to recalibrate probabilities, then re-measure AUROC / AUPRC.
def PlattRecal(y_true: numpy.ndarray, y_prob: numpy.ndarray, masks:dict) -> pandas.DataFrame:
    rows = []
    for subgroup, mask in masks.items():
        ms = mask.sum()
        if ms < config.MinGroup:
            continue
        sub_t = y_true[mask]
        sub_p = y_prob[mask]
        if len(set(sub_t)) < 2:
            continue


        OGAUROC = roc_auc_score(sub_t, sub_p)
        OGAUPRC = average_precision_score(sub_t, sub_p)

        #Scale Platt baised off of og values
        TrackOdds = numpy.log(sub_p / (1-sub_p+1e-8)).reshape(-1, 1)
        Platt = LogisticRegression(solver="lbfgs", max_iter=200)
        Platt.fit(TrackOdds, sub_t)
        RecallPlatt = Platt.predict_proba(TrackOdds)[:, 1]

        rocRecall = roc_auc_score(sub_t, RecallPlatt)
        prcRecall = average_precision_score(sub_t, RecallPlatt)

        rows.append({
            "subgroup":     subgroup,
            "n":            int(ms),
            "orig_auroc":   round(OGAUROC, 4),
            "recal_auroc":  round(rocRecall, 4),
            "orig_auprc":   round(OGAUPRC, 4),
            "recal_auprc":  round(prcRecall, 4),
        })
        return pandas.DataFrame(rows)

def ExecuteInterns(model: str, y_true: numpy.ndarray, y_prob:numpy.ndarray, masks: dict) -> dict:
    print(f"{config.BAG}\nIntervention on: {model}\n{config.BAG}")
    print(f"{config.AG}Running Per subgroup threshold tuneing..{config.AG}")
    ThreshCal = ThreshTune(y_true, y_prob, masks)
    print(f"{config.AG}Running Per subgroup Platt recalibratio..{config.AG}")
    Platt = PlattRecal(y_true, y_prob, masks)
    return{"threshold_tuning": ThreshCal, "platt_recalibration": Platt}