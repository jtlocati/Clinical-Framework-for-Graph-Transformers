#Run interventions on best TGT and baseline
import numpy
import pandas
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_recall_curve
from sklearn.linear_model import LogisticRegression
import Task6.config as config6
import Task9.config as config9

def FindBestModel(metrics: pandas.DataFrame) -> dict:
    overall = metrics[metrics["subgroup"] == "overall"].copy()

    BaslineRows = overall[overall["model"].isin(config6.BASELINE_MODELS)]
    TGTRows = overall[overall["model"].str.startswith("tgt_")]

    BestBaseline = BaslineRows.sort_values("auroc", ascending=False).iloc[0]["model"]
    BestTGT = TGTRows.sort_values("auroc", ascending=False).iloc[0]["model"]

    print(f"Best Baseline: {BestBaseline}\nBest TGT: {BestTGT}")
    return {"best_baseline": BestBaseline, "best_tgt": BestTGT}


#Intervention no. 1 -> threshold tuneung: find threshold that maximises the model in questions F1 score

def BestF1Thresh(y_true, y_prob):
    precision, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    F1s = 2 * precision * recalls / (precision + recalls + 1e-8)
    best_idx = numpy.argmax(F1s[:-1])
    return float(thresholds[best_idx])

def SweepThresh(y_true, y_prob, metric="f1"):
    thresholds = numpy.arange(config9.THRECH_SWEEP["start"], config9.THRECH_SWEEP["stop"], config9.THRECH_SWEEP["step"])
    best_thresh = 0.5
    best_score = -1

    for t in thresholds:
        predictions = (y_prob >= t).astype(int)

        if metric == "f1":
            score = f1_score(y_true, predictions, zero_division=0)
        elif metric == "balanced":
            tp = ((predictions == 1) & (y_true == 1)).sum()
            tn = ((predictions == 0) & (y_true == 0)).sum()
            pos = (y_true == 1).sum()
            neg = (y_true == 0).sum()
            tpr = tp / max(pos, 1)
            tnr = tn / max(neg, 1)
            score = min(tpr, tnr)
        else:
            raise ValueError(f"NO KNOWN METRIC: {metric} CHECK SPELLING")
        if score > best_score:
            best_score = score
            best_thresh = float(t)
    return best_thresh

def TargetTPRThresh(y_true, y_prob, target_tpr):
    thresholds = numpy.arange(config9.THRECH_SWEEP["start"], config9.THRECH_SWEEP["stop"], config9.THRECH_SWEEP["step"])
    best_thresh = 0.5
    best_diff = float("inf")
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        pos = (y_true == 1).sum()
        if pos == 0:
            continue
        tpr = preds[y_true == 1].mean()
        diff = abs(tpr - target_tpr)
        if diff < best_diff:
            best_diff = diff
            best_thresh = float(t)
    return best_thresh

def PerSGThresh(y_true: numpy.ndarray, y_prob: numpy.ndarray, masks: dict, globalthresh: float = None) -> dict:
    if globalthresh is None:
        globalthresh = BestF1Thresh(y_true, y_prob)

    # First pass: compute each subgroup's TPR at the global threshold
    group_tprs = {}
    for subgroup, mask in masks.items():
        num = mask.sum()
        if num < config6.MinGroup:
            continue
        sub_t = y_true[mask]
        sub_p = y_prob[mask]
        if len(set(sub_t)) < 2:
            continue
        global_preds = (sub_p >= globalthresh).astype(int)
        pos = (sub_t == 1).sum()
        if pos > 0:
            group_tprs[subgroup] = global_preds[sub_t == 1].mean()

    # Use median TPR across groups as the equalization target
    if len(group_tprs) == 0:
        target_tpr = 0.5
    else:
        target_tpr = float(numpy.median(list(group_tprs.values())))

    print(f"Target TPR (median across groups): {target_tpr:.4f}")

    thesholds = {}
    for subgroup, mask in masks.items():
        num = mask.sum()
        if num < config6.MinGroup:
            continue
        sub_t = y_true[mask]
        sub_p = y_prob[mask]
        if len(set(sub_t)) < 2:
            continue
        thesholds[subgroup] = TargetTPRThresh(sub_t, sub_p, target_tpr)
    return thesholds

def ApplyThresh(y_true: numpy.ndarray, y_prob: numpy.ndarray, masks: dict, globalthresh: float, groupthresh: dict) -> pandas.DataFrame:

    rows = []
    for subgroup, mask in masks.items():
        num = mask.sum()
        if num < config6.MinGroup:
            continue
        sub_t = y_true[mask]
        sub_p = y_prob[mask]
        if len(set(sub_t)) < 2:
            continue

        Tunedthresh = groupthresh.get(subgroup, globalthresh)

        GlobalPrediction = (sub_p >= globalthresh).astype(int)
        
        TunedPrediction = (sub_p >= Tunedthresh).astype(int)
        
        position = (sub_t == 1).sum()

        if position > 0:
            TPRglobal = GlobalPrediction[sub_t == 1].mean()
            TPRtuned = TunedPrediction[sub_t == 1].mean()
        else:
            TPRglobal, TPRtuned = 0.0, 0.0       

        neg = (sub_t == 0).sum()
        if neg > 0:
            FPRglobal = GlobalPrediction[sub_t == 0].mean()
            FPRtuned = TunedPrediction[sub_t == 0].mean()
        else:
            FPRglobal, FPRtuned = 0.0, 0.0
        
        F1global = f1_score(sub_t, GlobalPrediction, zero_division=0)
        F1tuned = f1_score(sub_t, TunedPrediction, zero_division=0)

        rows.append({
            "subgroup": subgroup,
            "n": int(num),
            "global_thresh": round(globalthresh, 4),
            "tuned_thresh": round(Tunedthresh, 4),
            "tpr_global": round(float(TPRglobal), 4),
            "tpr_tuned": round(float(TPRtuned), 4),
            "fpr_global": round(float(FPRglobal), 4),
            "fpr_tuned": round(float(FPRtuned), 4),
            "f1_global": round(float(F1global), 4),
            "f1_tuned": round(float(F1tuned), 4),
        })

    DATA = pandas.DataFrame(rows)

    if len(DATA) >= 2:
        tpr_gap_before = DATA["tpr_global"].max() - DATA["tpr_global"].min()
        tpr_gap_after  = DATA["tpr_tuned"].max()  - DATA["tpr_tuned"].min()
        reduction = tpr_gap_before - tpr_gap_after
        print(f"PR gap: {tpr_gap_before:.4f} -> {tpr_gap_after:.4f}\n(reduction: {reduction:+.4f})")
    return DATA



    

#For each subgroup, find the optimal threshold (max F1).Compare AUROC/AUPRC before and after (AUC doesn't change but effective precision and recall at the chosen threshold do).
def ThreshTune(y_true: numpy.ndarray, y_prob: numpy.ndarray, masks: dict) -> pandas.DataFrame:
    #single thresh used for all
    globalThresh = BestF1Thresh(y_true, y_prob)

    rows  = []
    for Subgroup, mask in masks.items():
        nm = mask.sum()

        if nm < config6.MinGroup:
            continue
        sub_t = y_true[mask]
        sub_p = y_prob[mask]
        if len(set(sub_t)) < 2:
            continue

        SubgOpThresh = BestF1Thresh(sub_t, sub_p)

        GlobalPreds = (sub_p >= globalThresh).astype(int)

        TPRglobal = GlobalPreds[sub_t==1].mean() if (sub_t == 1).sum() > 0 else 0

        TunedPreds = (sub_p >= SubgOpThresh).astype(int)
        TunedTPR = TunedPreds[sub_t == 1].mean() if (sub_t == 1).sum() > 0 else 0

        #F1 @ Global thresh 
        F1_global = f1_score(sub_t, GlobalPreds, zero_division=0)
        #F1 @ sub tunded thresh
        f1_tuned = f1_score(sub_t, TunedPreds, zero_division=0)

        rows.append({
            "subgroup": Subgroup,
            "n": int(nm),
            "global_thresh":round(globalThresh, 4),
            "tuned_thresh": round(SubgOpThresh, 4),
            "tpr_global": round(TPRglobal, 4),
            "tpr_tuned": round(TunedTPR, 4),
            "f1_global": round(F1_global, 4),
            "f1_tuned": round(f1_tuned, 4)
        })

    datafram = pandas.DataFrame(rows)

        #Output Gap summarie
    if len(datafram) >= 2:
        TPRgapBefore = datafram["tpr_global"].max() - datafram["tpr_global"].min()
        TPRgapPost = datafram["tpr_tuned"].max() - datafram["tpr_tuned"].min()
        print(f"{config6.AG}TPR GAP BEFORE: {TPRgapBefore:.4f}{config6.AG}\n{config6.AG}TPR GAP AFTER: {TPRgapPost}{config6.AG}\n{config6.BAG} REDUCTION: {TPRgapBefore - TPRgapPost}{config6.BAG}")

    return datafram

#intervention numero dos: Platt recalibration: Fit a logistic regression (Platt scaling) per subgroup to recalibrate probabilities, then re-measure AUROC / AUPRC.
def PlattRecal(y_true: numpy.ndarray, y_prob: numpy.ndarray, masks:dict) -> pandas.DataFrame:
    rows = []
    for subgroup, mask in masks.items():
        ms = mask.sum()
        if ms < config6.MinGroup:
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
            "subgroup": subgroup,
            "n": int(ms),
            "orig_auroc": round(OGAUROC, 4),
            "recal_auroc": round(rocRecall, 4),
            "orig_auprc": round(OGAUPRC, 4),
            "recal_auprc": round(prcRecall, 4),
        })
    return pandas.DataFrame(rows)

def ExecuteInterns(model: str, y_true: numpy.ndarray, y_prob:numpy.ndarray, masks: dict) -> dict:
    print(f"{config6.BAG}\nIntervention on: {model}\n{config6.BAG}")
    print(f"{config6.AG}Running Per subgroup threshold tuneing..{config6.AG}")
    ThreshCal = ThreshTune(y_true, y_prob, masks)
    print(f"{config6.AG}Running Per subgroup Platt recalibratio..{config6.AG}")
    Platt = PlattRecal(y_true, y_prob, masks)
    return{"threshold_tuning": ThreshCal, "platt_recalibration": Platt}