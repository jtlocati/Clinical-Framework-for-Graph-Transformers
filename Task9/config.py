import os
import Task6.config
#Formatting 
AG = "-"*60
BAG = "="* 60

#Will Reference task6 for most config settingd

ALL_MODELS = Task6.config.BASELINE_MODELS + [f"tgt_{v}" for v in Task6.config.TRANSFORMMERS]

CAL_METHODS = ["platt", "isotonic"]

THRECH_SWEEP = {"start": 0.0 , "stop": 0.99, "step": .005}

METHOD_LABELS = ["Original_GlobalThresh", "Original_PerGroupThresh", "Platt_GlobalThresh", "Platt_PerGroupThresh", "Isotonic_GlobalThresh", "Isotonic_PerGroupThresh"]

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Task9Results")
os.makedirs(OUTPUT_DIR, exist_ok=True)