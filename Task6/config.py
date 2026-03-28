import os
from dataclasses import dataclass

AG = ("-" * 10)
BAG = ("=" * 60)

BQ_PROJECT = "project-bb25f1c3-ff0a-47c9-9d0"
MIMIC = "physionet-data.mimiciv_3_1_derived"
DemographicsPath = r"C:\Users\jetlo\OneDrive\Documents\GitHub\Clinical-Framework-for-Graph-Transformers\Tests\outputs\Task2Demographics.csv"

BASELINE_MODELS = ["LSTM", "StandaredTransformer", "ChronoFormer", "PatchTST"]
TRANSFORMMERS = ["temporal_only", "clinical_only", "full_model"]

seed =  45 #same see for all cuz we are comparing directly

AGE_BINS = [0, 40, 65, 200]
AGE_LABELS = ["<40", "40-65", "65+"]
SUBGROUP_COLS = {"sex": "sex", "age": "age_group", "race": "race"}

#Race comes from task 2

MinGroup = 30

RESULTS_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_T6")
os.makedirs(RESULTS_DIRECTORY, exist_ok=True)