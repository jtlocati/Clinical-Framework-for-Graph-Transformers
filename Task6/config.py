import os
from dataclasses import dataclass

BQ_PROJECT = "project-bb25f1c3-ff0a-47c9-9d0"
MIMIC = "physionet-data.mimiciv_3_1_derived"

BASELINE_MODELS = ["LSTM", "StandaredTransformer", "ChronoFormer", "PatchTST"]
TRANSFORMMERS = ["temporal_only", "clinical_only", "full_model"]

seed =  45 #same see for all cuz we are comparing directly

AGE_BINS = [0, 40, 65, 200]
AGE_LABELS = ["<40", "40-65", "65+"]

#Race comes from task 2

MinGroup = 30

RESULTS_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__), "T6_results"))
os.makedirs(RESULTS_DIRECTORY, exist_ok=True)