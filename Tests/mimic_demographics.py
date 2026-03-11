import numpy
import pandas
from pathlib import Path
from sklearn.metrics import roc_auc_score

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

simple_label = {"age_<40": (new_table["age_group"] == "<40"), "age_40_65": (new_table["age_group"] == "40-65"), "age_65plus": (new_table["age_group"] == "65+"), "sex_male": (new_table["sex"] == "Male"), "sex_female": (new_table["sex"] == "Female"),}

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

print("=======================================================DISTROBUTION TABLE=======================================================")

filler = "================================================================================="

print(f"SAVED: {dist_path}\n{filler}\n{dist}\n{filler}")

print("Groups below 30 minimum:")
print(dist.loc[~dist["enough_samples_in30"], ["group", "count"]])


#print(new_table["age_group"].value_counts(dropna=False))


#AUC subgruop calculations

#use placeholders till model predictions exsist

y_true = new_table["label"].values
y_prob = numpy.random.rand(len(new_table)) #PLACEHOLDER

subgroup_results = []

for name, mask in simple_label.items():

    mask_array = mask.values
    n = mask_array.sum()

    #jump over small groups

    if n <= 30:
        subgroup_results.append({ "group": name, "count": n, "AUC": None, "note": "Too few samples"})
        continue
    
    sub_y_true = y_true[mask_array]
    sub_y_prob = y_prob[mask_array]

    if len(set(sub_y_true)) < 3:
        subgroup_results.append({"group": name, "count": n, "AUC": None, "note": "Single class"})
        continue
    auc = roc_auc_score(sub_y_true, sub_y_prob)

    subgroup_results.append({"group": name, "count": n, "AUC": auc, "note": "ok"})

auc_table = pandas.DataFrame(subgroup_results)

#Save Results:
auc_path = OUT_DIRECTORY / "mimic_demo_subgroup_auc.csv"
auc_table.to_csv(auc_path, index=False)

print(f"\nSaved AUC table: {auc_path}")

print("=======================================================AUC table=======================================================")   

print(f"\nSubgroup AUC table results\n{auc_table}")