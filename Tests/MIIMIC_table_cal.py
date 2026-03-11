from google.cloud import bigquery

#Test Big Query connection
"""MIMIC = bigquery.Client(project="project-bb25f1c3-ff0a-47c9-9d0")

query = "SELECT * FROM `physionet-data.mimiciv_3_1_derived.first_day_weight` LIMIT 1000;"

TQ = MIMIC.query(query).to_dataframe()
print(TQ)"""

#Get row information

MIMIC = bigquery.Client(project="project-bb25f1c3-ff0a-47c9-9d0")

tables = [
    "physionet-data.mimiciv_3_1_derived.icustay_hourly",
    "physionet-data.mimiciv_3_1_derived.vitalsign",
    "physionet-data.mimiciv_3_1_derived.chemistry",
    "physionet-data.mimiciv_3_1_derived.complete_blood_count",
    "physionet-data.mimiciv_3_1_derived.gcs",
    "physionet-data.mimiciv_3_1_derived.urine_output_rate",
]

for table in tables:
    query = f"SELECT * FROM `{table}` LIMIT 5"
    df = MIMIC.query(query).to_dataframe()
    print(f"\n--- {table} ---")
    print(df)
    print("Columns:", list(df.columns))

# one ICU stay:
query = """
SELECT stay_id, subject_id, hadm_id
FROM `physionet-data.mimiciv_3_1_derived.icustay_detail`
LIMIT 1
"""
icustay_df = MIMIC.query(query).to_dataframe()
print("\n============= one ICU stay =============")
print(icustay_df)