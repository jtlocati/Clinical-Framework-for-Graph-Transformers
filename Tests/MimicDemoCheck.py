import pandas as pd
from pathlib import Path
import os

#!!!!!!!!!TEMPORARY FILE!!!!!!!!!
    #Used to find table structure of MIMIC_demo DS

# YAIB MIMIC demo folder
DATA_ROOT = Path(r"C:\Users\jetlo\OneDrive\Documents\GitHub\YAIB\demo_data\mortality24\mimic_demo")

#DATA_ROOT = Path(os.getenv("MIMIC_DEMO_PATH"))

# Read tables
sta = pd.read_parquet(DATA_ROOT / "sta.parquet")
outc = pd.read_parquet(DATA_ROOT / "outc.parquet")

# Basic info
print(f"STA shape: {sta.shape}")
print(f"outc shape: {outc.shape}\n")

print(f"STA cols: {list(sta.columns)}")
print(f"outc cols: {list(outc.columns)}\n")

print("STA head:")
print(sta.head(), "\n")

print("outc head:")
print(outc.head(), "\n")

# Check shared key
print("Has stay_id?")
print("sta:", "stay_id" in sta.columns)
print("outc:", "stay_id" in outc.columns)

# If stay_id exists in both, check uniqueness (helps avoid row explosion on merge)
if "stay_id" in sta.columns and "stay_id" in outc.columns:
    print("\nUniqueness check:")
    print("sta stay_id unique?", sta["stay_id"].is_unique)
    print("outc stay_id unique?", outc["stay_id"].is_unique)
    print("sta rows:", len(sta), "outc rows:", len(outc))

# Quick scan for likely label/target columns in outc
targets = ["label", "target", "mort", "death", "outcome", "y"]
label_candidates = [c for c in outc.columns if any(t in c.lower() for t in targets)]
print("\nPossible label columns in outc:", label_candidates)