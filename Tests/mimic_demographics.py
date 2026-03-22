#Task 4
import numpy
import pandas
from pathlib import Path
from sklearn.metrics import roc_auc_score
from google.cloud import bigquery

#Change to envion varible
super_cool_term_outp_orgaizer = "================================================================================="
BQ_PROJECT = "project-bb25f1c3-ff0a-47c9-9d0"
MIMIC_DATASET = "physionet-data.mimiciv_3_1_derived"
def get_client():
    return bigquery.Client(project=BQ_PROJECT)

OUT_DIRECTORY = Path("Tests/outputs")
OUT_DIRECTORY .mkdir(exist_ok=True)

def get_race(race):
    race = str(race).upper()
    if "WHITE" in race:
        return "White"
    elif "BLACK" in race:
        return "Black"
    elif "HISPANIC" in race:
        return "Hispanic"
    elif "ASIAN" in race:
        return "Asian"
    elif "MULTIPLE RACE/ETHNICITY" in race:
        return "Multipule"
    elif "UNKNOWN" in race:
        return "Unknown"
    return "Other"


#Extract data from mimic
print(f"Querying for cohart demographics. root:{BQ_PROJECT}.{MIMIC_DATASET}")
Client = get_client()
BQquery = f"""
SELECT stay_id, gender AS sex, admission_age AS age, race, hospital_expire_flag AS label
FROM `{MIMIC_DATASET}.icustay_detail`
WHERE admission_age >= 18 AND los_icu >= 2
ORDER BY stay_id"""

pat_data = Client.query(BQquery).to_dataframe()
print(f"Returned inital data: {len(pat_data):,}")

new_table = pat_data[["stay_id", "age", "sex", "race", "label"]].copy()
#View OG table
print(f"{super_cool_term_outp_orgaizer} Og Table {super_cool_term_outp_orgaizer}")
print(new_table)

new_table["race"] = new_table["race"].apply(get_race)

new_table["age_group"] = pandas.cut(
    new_table["age"],
    bins=[-numpy.inf, 40, 65, numpy.inf],
    labels=["<40", "40-65", "65+"],
    right=False,
)

row_level_demo = new_table[["stay_id", "age", "sex", "race", "label", "age_group"]].copy()
row_level_path = OUT_DIRECTORY / "mimic_row_level_demographics.csv"
row_level_demo.to_csv(row_level_path, index=False)
print(f"Saved row-level demographics to: {row_level_path}")


#specify Male/Female only

#Ag_groups


simple_label = {
    #age
    "age_<40": (new_table["age_group"] == "<40"), "age_40_65": (new_table["age_group"] == "40-65"), "age_65plus": (new_table["age_group"] == "65+"),
    #sex
    "sex_male": (new_table["sex"] == "M"), "sex_female": (new_table["sex"] == "F"),
    #Race
    "race_white": (new_table["race"] == "White"), "race_black": (new_table["race"] == "Black"), "race_Hispanic": (new_table["race"] == "Hispanic"), "race_asian": (new_table["race"] == "Asian"), "race_multipule": (new_table["race"] == "Multipule"), "race_other": (new_table["race"] == "Other"), "race_unknown": (new_table["race"] == "Unknown")
    
    }

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

dist_path = OUT_DIRECTORY / "mimic_subgroup_distribution.csv"

dist.to_csv(dist_path, index=False)

print("=======================================================DISTROBUTION TABLE=======================================================")

print(f"SAVED: {dist_path}\n{super_cool_term_outp_orgaizer}\n{dist}\n{super_cool_term_outp_orgaizer}")

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

    if len(set(sub_y_true)) < 2:
        subgroup_results.append({"group": name, "count": n, "AUC": None, "note": "Single class"})
        continue
    auc = roc_auc_score(sub_y_true, sub_y_prob)

    subgroup_results.append({"group": name, "count": n, "AUC": auc, "note": "ok"})

auc_table = pandas.DataFrame(subgroup_results)

#Save Results:
auc_path = OUT_DIRECTORY / "mimic_subgroup_auc.csv"
auc_table.to_csv(auc_path, index=False)

print(f"\nSaved AUC table: {auc_path}")

#Task 2 does not require computing AUC or fairness metrics yet. only test to see if subgroup has sufficient sample size to support reliable evaluation in later tasks.
print("=======================================================AUC table=======================================================")   

print(f"\nSubgroup AUC table results\n{auc_table}")