from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK6_RESULTS = REPO_ROOT / "Task6" / "results"

seed_map = {
    "LSTM": "123",
    "TGTTemporal": "123",
    "TGTClinical": "123",
    "TGTFull": "123",
}

model_order = ["LSTM", "TGTTemporal", "TGTClinical", "TGTFull"]
attribute_order = ["sex", "age", "race"]

dfs = []
for model in model_order:
    path = TASK6_RESULTS / f"{model}_seed{seed_map[model]}_subgroup_metrics.csv"
    df = pd.read_csv(path)
    df = df[df["group"] != "overall"].copy()
    df["attribute"] = df["group"].str.split("_", n=1).str[0]
    df["subgroup"] = df["group"].str.split("_", n=1).str[1]
    df["model"] = model
    dfs.append(df)

all_df = pd.concat(dfs, ignore_index=True)
all_df = all_df[all_df["note"] == "ok"].copy()

subgroup_orders = {
    "sex": ["F", "M"],
    "age": ["<40", "40-65", "65+"],
    "race": ["Asian", "Black", "Hispanic", "Other", "Unknown", "White"],
}

display_labels = {
    "sex": {"F": "Female", "M": "Male"},
    "age": {"<40": "<40", "40-65": "40-65", "65+": "65+"},
    "race": {
        "Asian": "Asian",
        "Black": "Black",
        "Hispanic": "Hispanic",
        "Other": "Other",
        "Unknown": "Unknown",
        "White": "White",
    },
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), sharey=False)

for ax, attr in zip(axes, attribute_order):
    df_attr = all_df[all_df["attribute"] == attr].copy()
    order = subgroup_orders[attr]
    x = np.arange(len(order))
    width = 0.18

    if attr == "sex":
        ymin, ymax = 0.82, 0.86
        ticks = np.arange(0.82, 0.861, 0.005)
    elif attr == "age":
        ymin, ymax = 0.80, 0.92
        ticks = np.arange(0.80, 0.921, 0.02)
    else:  #race
        ymin, ymax = 0.70, 0.90
        ticks = np.arange(0.70, 0.901, 0.02)

    ax.set_ylim(ymin, ymax)
    ax.set_yticks(np.round(ticks, 3))

    for i, model in enumerate(model_order):
        df_model = df_attr[df_attr["model"] == model].copy()
        value_map = dict(zip(df_model["subgroup"], df_model["AUROC"]))
        y = [value_map.get(g, np.nan) for g in order]
        ax.bar(x + (i - 1.5) * width, y, width=width, label=model)

    ax.set_xticks(x)
    ax.set_xticklabels([display_labels[attr][g] for g in order], rotation=20 if attr == "race" else 0)
    ax.set_title(f"{attr.capitalize()} subgroup AUROC")
    ax.set_xlabel("Subgroup")
    ax.set_ylabel("AUROC")
    ax.set_ylim(ymin, ymax)

    if attr == "sex":
        ymin, ymax = 0.82, 0.86
        ticks = np.arange(0.82, 0.861, 0.005)
    elif attr == "age":
        ymin, ymax = 0.80, 0.92
        ticks = np.arange(0.80, 0.921, 0.02)
    else:
        ymin, ymax = 0.70, 0.90
        ticks = np.arange(0.70, 0.901, 0.02)

    ax.set_ylim(ymin, ymax)
    ax.set_yticks(np.round(ticks, 3))
    ax.grid(axis="y", alpha=0.3)

axes[0].legend(loc="best")
fig.suptitle("Subgroup Performance by Model (Best Seed per Model)", y=1.02)
plt.tight_layout()

out_path = TASK6_RESULTS / "task6_subgroup_auroc_barplot.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved: {out_path}")