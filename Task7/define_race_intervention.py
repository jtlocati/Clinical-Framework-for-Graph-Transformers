from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
demo = pd.read_csv(REPO_ROOT / "Tests" / "outputs" / "Task2Demographics.csv")
demo.columns = demo.columns.str.strip()

race_counts = demo["race"].value_counts(dropna=False).to_dict()

# Exclude ultra-rare category from direct intervention weighting
excluded_groups = {"Multipule"}

usable = {k: v for k, v in race_counts.items() if k not in excluded_groups}
total = sum(usable.values())

weights = {k: total / v for k, v in usable.items()}

df = pd.DataFrame(
    {
        "race": list(usable.keys()),
        "count": list(usable.values()),
        "inverse_freq_weight": [weights[k] for k in usable.keys()],
        "capped_weight": [min(weights[k], 10.0) for k in usable.keys()],
    }
).sort_values("count", ascending=False)

print(df.to_string(index=False))
print("\nExcluded groups from direct weighting:", excluded_groups)

out_path = REPO_ROOT / "Task7" / "results" / "race_intervention_weights.csv"
df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")