from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
demo = pd.read_csv(REPO_ROOT / "Tests" / "outputs" / "Task2Demographics.csv")
demo.columns = demo.columns.str.strip()

print("Sex counts")
print(demo["sex"].value_counts(dropna=False))
print("\nAge group counts")
print(demo["age_group"].value_counts(dropna=False))
print("\nRace counts")
print(demo["race"].value_counts(dropna=False))

out_dir = REPO_ROOT / "Task7" / "results" / "counts"
out_dir.mkdir(parents=True, exist_ok=True)

demo["sex"].value_counts(dropna=False).rename_axis("sex").reset_index(name="count").to_csv(
    out_dir / "sex_counts.csv", index=False
)
demo["age_group"].value_counts(dropna=False).rename_axis("age_group").reset_index(name="count").to_csv(
    out_dir / "age_group_counts.csv", index=False
)
demo["race"].value_counts(dropna=False).rename_axis("race").reset_index(name="count").to_csv(
    out_dir / "race_counts.csv", index=False
)

print(f"\nSaved count tables to: {out_dir}")
