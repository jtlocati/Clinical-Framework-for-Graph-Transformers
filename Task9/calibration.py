import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import Task9.config
import Task6.config

#platt: =

def PlattFitPredict(y_true, y_prob):

    # Clamp probabilities to avoid log(0) and log(inf)
    y_prob_clipped = numpy.clip(y_prob, 1e-8, 1 - 1e-8)
    log_odds = numpy.log(y_prob_clipped / (1 - y_prob_clipped)).reshape(-1, 1)
    platt = LogisticRegression(solver="lbfgs", max_iter=500) #Change manually as needed
    platt.fit(log_odds, y_true)
    calibrated = platt.predict_proba(log_odds)[:, 1]
    # output has to be in [0, 1]
    return numpy.clip(calibrated, 0.0, 1.0)

def IsoFitPredict(y_true, y_prob):
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(y_prob, y_true)
    return iso.predict(y_prob)

def CalSubgroup(y_true: numpy.ndarray, y_prob: numpy.ndarray, masks: dict, method: str = "platt") -> numpy.ndarray:

    if method == Task9.config.CAL_METHODS[0]:  # "platt"
        calibrated = PlattFitPredict(y_true, y_prob)
    elif method == Task9.config.CAL_METHODS[1]:  # "isotonic"
        calibrated = IsoFitPredict(y_true, y_prob)
    else:
        raise ValueError(f"Invalid calibration method: {method}. Must be 'platt' or 'isotonic'")
    
    # Print calibration stats per group for monitoring
    print(f"\n  Calibration statistics ({method}):")
    attributes = {}
    for subgroup, mask in masks.items():
        attr = subgroup.split("_")[0]
        if attr not in attributes:
            attributes[attr] = []
        attributes[attr].append(subgroup)
    
    for attr in sorted(attributes.keys()):
        print(f"    {attr}: {len(attributes[attr])} subgroups calibrated")
    
    return calibrated

def CalAllModels(predictions: dict, masks: dict) -> dict:
    result = {"original": {}, "platt": {}, "isotonic": {}}

    for model, (y_true, y_prob) in predictions.items():

        result["original"][model] = (y_true, y_prob)

        for method in Task9.config.CAL_METHODS:
            print(f"{Task9.config.AG} Calebrating: {model} with {method} {Task6.config.AG}")
            cal_prob = CalSubgroup(y_true, y_prob, masks, method)
            result[method][model] = (y_true, cal_prob)
    return result