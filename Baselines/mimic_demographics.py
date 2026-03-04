import numpy
import pandas
from pathlib import Path


#Change to envion varible
DATA_ROOT = Path(r"C:\Users\jetlo\OneDrive\Documents\GitHub\YAIB\demo_data\mortality24\mimic_demo")

OUT_DIRECTORY = Path("outputs")
OUT_DIRECTORY .mkdir(exist_ok=True)

#loading MIMIC
sta = pandas.read_parquet(DATA_ROOT / "sta.parquet")
outc = pandas.read_parquet(DATA_ROOT / "outc.parquet")

#MERGE into one table so that one row per stat_id
table = sta.merge(outc, on="stay_id", how="inner", validate="one_to_one")

new_table = table[["stay_id", "age", "sex", "label"]].copy()

#specify Male/Female only

#Ag_groups

new_table["age_group"] = pandas.cut(new_table["age"], bins=[-numpy.inf, 40, 65, numpy.inf], labels=["<40", "40-65", "65+"],right=False)

simple_label = {    "age_<40": (new_table["age_group"] == "<40"), "age_40_65": (new_table["age_group"] == "40-65"), "age_65plus": (new_table["age_group"] == "65+"), "sex_male": (new_table["sex"] == "Male"), "sex_female": (new_table["sex"] == "Female"),}

#Distrobution table:\

row = []

for name, item in simple_label.items():
    n = item.sum()
    if n == 0:
        pos = 0
        rate = numpy.nan

    else:
        pos = new_table.loc[item, "label"].sum()
        rate = float(new_table.loc[item, "label"].mean())
    
    row.append({"group": name, "count": n, "positive_label_count": pos, "positive_rate": rate,})

dist = pandas.DataFrame(row).sort_values("count", ascending=False).reset_index(drop=True)

#30 min threshold

MAX_MIN = 30
dist["enough_samples_in30"] = dist["count"] >= MAX_MIN

dist_path = OUT_DIRECTORY / "mimic_demo_subgroup_distribution.csv"

dist.to_csv(dist_path, index=False)

filler = "================================================================================="

print(f"SAVED: {dist_path}\n{filler}\n{dist}\n{filler}")

print("Groups below 30 minimum:")
print(dist.loc[~dist["enough_samples_in30"], ["group", "count"]])


#print(new_table["age_group"].value_counts(dropna=False))
