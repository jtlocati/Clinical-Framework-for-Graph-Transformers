from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK6_RESULTS = REPO_ROOT / "Task6" / "results"

df = pd.read_csv(TASK6_RESULTS / "fairness_gap_summary_by_model.csv")

model_order = ["LSTM", "TGTTemporal", "TGTClinical", "TGTFull"]
attr_order = ["sex", "age", "race"]

heatmap_df = df.pivot(index="model", columns="attribute", values="AUROC_gap_mean")
heatmap_df = heatmap_df.reindex(index=model_order, columns=attr_order)

values = heatmap_df.values

fig, ax = plt.subplots(figsize=(8, 4.8))
im = ax.imshow(values, aspect="auto")

cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Mean AUROC gap")

ax.set_xticks(range(len(heatmap_df.columns)))
ax.set_xticklabels(["Sex", "Age", "Race"])
ax.set_yticks(range(len(heatmap_df.index)))
ax.set_yticklabels(heatmap_df.index)

ax.set_title("Fairness: Mean Subgroup AUROC Gap")
ax.set_xlabel("Demographic attribute")
ax.set_ylabel("Model")

# annotate cells
for i in range(values.shape[0]):
    for j in range(values.shape[1]):
        val = values[i, j]
        text_color = "white" if val > np.nanmean(values) else "black"
        ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=text_color, fontsize=10)

plt.tight_layout()

out_path = TASK6_RESULTS / "task6_auroc_gap_heatmap_annotated.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")

print(f"Saved: {out_path}")
print("\nHeatmap values:")
print(heatmap_df)