"""
inspect_schema.py — Print the actual column names inside each MIMIC table
"""
from DataPipeline import get_client
import pandas

def BQ(SQL: str) -> pandas.DataFrame:
    client = get_client()
    print("Query Running....")
    df = client.query(SQL).to_dataframe()
    print(f"Returned: {len(df):,} rows")
    return df


Tables_derived = [
    "icustay_detail",
    "vitalsign",
    "chemistry",
    "complete_blood_count",
    "coagulation",
    "bg",
]


print(f"\n{'='*60}")
print("  admissions")
print(f"{'='*60}")
df = BQ("""
SELECT column_name, data_type
FROM `physionet-data.mimiciv_3_1_hosp.INFORMATION_SCHEMA.COLUMNS`
WHERE table_name = 'admissions'
ORDER BY ordinal_position
""")
for _, row in df.iterrows():
    print(f"  {row['column_name']:<30s} {row['data_type']}")

for table in Tables_derived:
    print(f"\n{'='*60}")
    print(f"  {table}")
    print(f"{'='*60}")
    df = BQ(f"""
    SELECT column_name, data_type
    FROM `physionet-data.mimiciv_3_1_derived.INFORMATION_SCHEMA.COLUMNS`
    WHERE table_name = '{table}'
    ORDER BY ordinal_position
    """)
    for _, row in df.iterrows():
        print(f"  {row['column_name']:<30s} {row['data_type']}")

