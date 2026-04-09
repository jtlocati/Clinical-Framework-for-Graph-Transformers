import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import Task9.config
import Task6.config

#platt: =

def PlattFitPredict(y_true, y_prob):
    log_odds = numpy.log(y_prob / (1-y_prob + 1e-8)).reshape(-1, 1)
    platt = LogisticRegression(solver="lbfgs", max_iter=500) #Change manually as needed
    platt.fit(log_odds, y_true)
    return platt.predict_proba(log_odds)[:, 1]

def IsoFitPredict(y_true, y_prob):
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(y_prob, y_true)
    return iso.predict(y_prob)

def CalSubgroup(y_true: numpy.ndarray, y_prob: numpy.ndarray, masks: dict, method: str = "platt") -> numpy.ndarray:

    if method == Task9.config.CAL_METHODS[0]:
        fit_fn = PlattFitPredict
    elif method == Task9.config.CAL_METHODS[1]:
        fit_fn = IsoFitPredict
    else:
        raise ValueError(f"The method of: {method} is not valid, CHECK SPELLING")
    
    calabrated = y_prob.copy()
    covered = numpy.zeros(len(y_prob), dtype=bool)

    for subgroup, mask in masks.items():
        num = mask.sum()
        if num < Task6.config.MinGroup:
            continue
        sub_t = y_true[mask]
        sub_p = y_prob[mask]

        if len(set(sub_t)) < 2:
            continue

        cal_p = fit_fn(sub_t, sub_p)
        calabrated[mask] = cal_p
        covered |= mask
    uncovered = (~covered).sum()
    if uncovered > 0:
        print(f"UNCOVERED FILES: {uncovered}")

def CalAllModels(predictions: dict, masks: dict) -> dict:
    result = {"original": {}, "platt": {}, "isotonic": {}}

    for model, (y_true, y_prob) in predictions.items():

        result["original"][model] = (y_true, y_prob)

        for method in Task9.config.CAL_METHODS:
            print(f"{Task9.config.AG} Calebrating: {model} with {method} {Task6.config.AG}")
            cal_prob = CalSubgroup(y_true, y_prob, masks, method)
            result[method][model] = (y_true, y_prob)
    return result