import os
import numpy
import pandas
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot
import Task9.config as config9
import Task6.config as config6

METHOD_COLORS = {
    "Original_GlobalThresh":    "#2c3e50",
    "Original_PerGroupThresh":  "#7f8c8d",
    "Platt_GlobalThresh":       "#2980b9",
    "Platt_PerGroupThresh":     "#3498db",
    "Isotonic_GlobalThresh":    "#c0392b",
    "Isotonic_PerGroupThresh":  "#e74c3c",
}

METHOD_MARKERS = {
    "Original_GlobalThresh":    "o",
    "Original_PerGroupThresh":  "s",
    "Platt_GlobalThresh":       "^",
    "Platt_PerGroupThresh":     "D",
    "Isotonic_GlobalThresh":    "v",
    "Isotonic_PerGroupThresh":  "P",
}

def TradeoffPlot(comparison: pandas.DataFrame):
    save_directory = config9.OUTPUT_DIR
    models = comparison["model"].unique()

    fig, axes = matplotlib.pyplot.subplots(1, len(models), figsize=(7*len(models), 6), squeeze=False)
    for idx, model in enumerate(models):
        ax = axes[0, idx]
        mdf = comparison[comparison["model"] == model]

        for _, row in mdf.iterrows():
            method = row["method"]
            ax.scatter(
            row["max_auroc_gap"], row["overall_auroc"],
            c=METHOD_COLORS.get(method, "gray"),
            marker=METHOD_MARKERS.get(method, "o"),
            s=120, edgecolors="black", linewidths=0.5,
            zorder=3,
            )
            ax.annotate(
                method.replace("_", "\n"),
                (row["max_auroc_gap"], row["overall_auroc"]),
                fontsize=6, ha="left", va="bottom",
                xytext=(5, 5), textcoords="offset points",
            )
        ax.set_xlabel("Max AUROC Fairness Gap", fontsize=11)
        ax.set_ylabel("Overall AUROC", fontsize=11)
        ax.set_title(model, fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.invert_xaxis()  # lower gap = better = right side

    matplotlib.pyplot.suptitle("AUROC vs Fairness Gap Trade-Off", fontsize=14, y=1.02)
    matplotlib.pyplot.tight_layout()
    path = os.path.join(save_directory, "T9_tradeoff_scatter.png")
    matplotlib.pyplot.savefig(path, dpi=150, bbox_inches="tight")
    matplotlib.pyplot.close()
    print(f"Trade-off scatter -> {path}")

def plot_before_after_bars(comparison_df: pandas.DataFrame, save_dir: str = None):
    save_dir = config9.OUTPUT_DIR
    models = comparison_df["model"].unique()
    methods = comparison_df["method"].unique()
    n_models = len(models)
    n_methods = len(methods)

    x = numpy.arange(n_models)
    width = 0.8 / n_methods

    fig, ax = matplotlib.pyplot.subplots(figsize=(max(10, n_models * 3), 6))

    for i, method in enumerate(methods):
        values = []
        for model in models:
             row = comparison_df[
                (comparison_df["model"] == model) &
                   (comparison_df["method"] == method)
            ]
             values.append(row["max_auroc_gap"].values[0] if len(row) else 0)

        offset = (i - n_methods / 2 + 0.5) * width
        color = METHOD_COLORS.get(method, "gray")
        ax.bar(x + offset, values, width, label=method, color=color, alpha=0.85)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Max AUROC Fairness Gap", fontsize=12)
    ax.set_title("Fairness Gap: Before vs After Post-Hoc Methods", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9, rotation=15, ha="right")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    matplotlib.pyplot.tight_layout()
    path = os.path.join(save_dir, "T9_before_after_bars.png")
    matplotlib.pyplot.savefig(path, dpi=150)
    matplotlib.pyplot.close()
    print(f"Before & after bars -> {path}")



def plot_tpr_gap_comparison(comparison_df: pandas.DataFrame, save_dir: str = None):
    save_dir = save_dir or config9.OUTPUT_DIR
    models = comparison_df["model"].unique()
    methods = comparison_df["method"].unique()
    n_models = len(models)
    n_methods = len(methods)

    x = numpy.arange(n_models)
    width = 0.8 / n_methods

    fig, ax = matplotlib.pyplot.subplots(figsize=(max(10, n_models * 3), 6))

    for i, method in enumerate(methods):
        values = []
        for model in models:
            row = comparison_df[
                (comparison_df["model"] == model) &
                (comparison_df["method"] == method)
            ]
            values.append(row["tpr_gap"].values[0] if len(row) else 0)

        offset = (i - n_methods / 2 + 0.5) * width
        color = METHOD_COLORS.get(method, "gray")
        ax.bar(x + offset, values, width, label=method, color=color, alpha=0.85)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("TPR Gap (max - min)", fontsize=12)
    ax.set_title("TPR Gap: Before vs After Post-Hoc Methods", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9, rotation=15, ha="right")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    matplotlib.pyplot.tight_layout()
    path = os.path.join(save_dir, "T9_tpr_gap_comparison.png")
    matplotlib.pyplot.savefig(path, dpi=150)
    matplotlib.pyplot.close()
    print(f"TPR gap comparison -> {path}")

def plot_method_heatmap(all_results: dict, model_name: str, save_dir: str = None):

    save_dir = save_dir or config9.OUTPUT_DIR
    model_res = all_results.get(model_name, {})
    if not model_res:
        return

    rows = []
    for method_label, res in model_res.items():
        gdf = res["gaps"]
        entry = {"method": method_label}
        for attr in config6.SUBGROUP_COLS.keys():
            attr_rows = gdf[gdf["attribute"] == attr]
            entry[attr] = attr_rows["auroc_gap"].values[0] if len(attr_rows) else numpy.nan
        rows.append(entry)

    df = pandas.DataFrame(rows).set_index("method")

    fig, ax = matplotlib.pyplot.subplots(figsize=(8, max(4, len(df) * 0.5)))
    im = ax.imshow(df.values, cmap="YlOrRd", aspect="auto", vmin=0)

    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, fontsize=11)
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index, fontsize=8)

    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            val = df.values[i, j]
            if numpy.isnan(val):
                text = "-"
                color = "black"
            else:
                text = f"{val:.3f}"
                color = "white" if val > 0.06 else "black"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=9, color=color)

    fig.colorbar(im, ax=ax, label="AUROC gap")
    ax.set_title(f"AUROC Gap by Method — {model_name}", fontsize=12, pad=12)

    matplotlib.pyplot.tight_layout()
    safe_name = model_name.replace(" ", "_")
    path = os.path.join(save_dir, f"T9_heatmap_{safe_name}.png")
    matplotlib.pyplot.savefig(path, dpi=150)
    matplotlib.pyplot.close()
    print(f"  Heatmap ({model_name}) -> {path}")



def plot_auprc_heatmap(all_results: dict, model_name: str, save_dir: str = None):
    save_dir = save_dir or config9.OUTPUT_DIR
    model_res = all_results.get(model_name, {})
    if not model_res:
        return

    rows = []
    for method_label, res in model_res.items():
        gdf = res["gaps"]
        entry = {"method": method_label}
        for attr in config6.SUBGROUP_COLS.keys():
            attr_rows = gdf[gdf["attribute"] == attr]
            entry[attr] = attr_rows["auprc_gap"].values[0] if len(attr_rows) else numpy.nan
        rows.append(entry)

    df = pandas.DataFrame(rows).set_index("method")

    fig, ax = matplotlib.pyplot.subplots(figsize=(8, max(4, len(df) * 0.5)))
    im = ax.imshow(df.values, cmap="YlOrRd", aspect="auto", vmin=0)

    ax.set_xticks(range(len(df.columns)))
    ax.set_xticklabels(df.columns, fontsize=11)
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index, fontsize=8)

    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            val = df.values[i, j]
            if numpy.isnan(val):
                text = "-"
                color = "black"
            else:
                text = f"{val:.3f}"
                color = "white" if val > 0.06 else "black"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=9, color=color)

    fig.colorbar(im, ax=ax, label="AUPRC gap")
    ax.set_title(f"AUPRC Gap by Method — {model_name}", fontsize=12, pad=12)

    matplotlib.pyplot.tight_layout()
    safe_name = model_name.replace(" ", "_")
    path = os.path.join(save_dir, f"T9_auprc_heatmap_{safe_name}.png")
    matplotlib.pyplot.savefig(path, dpi=150)
    matplotlib.pyplot.close()
    print(f"  AUPRC Heatmap ({model_name}) -> {path}")


def plot_tpr_by_attribute(all_results: dict, model_name: str, save_dir: str = None):
    save_dir = save_dir or config9.OUTPUT_DIR
    model_res = all_results.get(model_name, {})
    if not model_res:
        return

    attributes = list(config6.SUBGROUP_COLS.keys())
    methods_global = [m for m in config9.METHOD_LABELS if "GlobalThresh" in m]
    methods_pergroup = [m for m in config9.METHOD_LABELS if "PerGroupThresh" in m]

    fig, axes = matplotlib.pyplot.subplots(1, 2, figsize=(16, 6))

    for ax, method_set, title_suffix in [
        (axes[0], methods_global, "Global Threshold"),
        (axes[1], methods_pergroup, "Per-Group Threshold"),
    ]:
        n_attrs = len(attributes)
        n_methods = len(method_set)
        x = numpy.arange(n_attrs)
        width = 0.8 / n_methods

        for i, method in enumerate(method_set):
            res = model_res.get(method)
            if res is None:
                continue
            tdf = res["threshold_detail"]
            gaps = []
            for attr in attributes:
                attr_rows = tdf[tdf["subgroup"].str.startswith(attr + "_")]
                if len(attr_rows) < 2:
                    gaps.append(0)
                    continue
                if "PerGroup" in method:
                    tpr_col = "tpr_tuned"
                else:
                    tpr_col = "tpr_global"
                gaps.append(attr_rows[tpr_col].max() - attr_rows[tpr_col].min())

            offset = (i - n_methods / 2 + 0.5) * width
            color = METHOD_COLORS.get(method, "gray")
            ax.bar(x + offset, gaps, width, label=method, color=color, alpha=0.85)

        ax.set_xlabel("Attribute", fontsize=11)
        ax.set_ylabel("TPR Gap (max - min)", fontsize=11)
        ax.set_title(f"{model_name} — TPR Gap by Attribute ({title_suffix})", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(attributes, fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(axis="y", alpha=0.3)

    matplotlib.pyplot.tight_layout()
    safe_name = model_name.replace(" ", "_")
    path = os.path.join(save_dir, f"T9_tpr_by_attribute_{safe_name}.png")
    matplotlib.pyplot.savefig(path, dpi=150)
    matplotlib.pyplot.close()
    print(f"  TPR by attribute ({model_name}) -> {path}")


def plot_overall_performance(comparison_df: pandas.DataFrame, save_dir: str = None):
    """Side-by-side bar charts showing overall AUROC and AUPRC per method."""
    save_dir = save_dir or config9.OUTPUT_DIR
    models = comparison_df["model"].unique()
    methods = comparison_df["method"].unique()
    n_models = len(models)
    n_methods = len(methods)

    fig, axes = matplotlib.pyplot.subplots(1, 2, figsize=(16, 6))

    for ax, metric, label in [
        (axes[0], "overall_auroc", "Overall AUROC"),
        (axes[1], "overall_auprc", "Overall AUPRC"),
    ]:
        x = numpy.arange(n_models)
        width = 0.8 / n_methods

        for i, method in enumerate(methods):
            values = []
            for model in models:
                row = comparison_df[
                    (comparison_df["model"] == model) &
                    (comparison_df["method"] == method)
                ]
                values.append(row[metric].values[0] if len(row) else 0)

            offset = (i - n_methods / 2 + 0.5) * width
            color = METHOD_COLORS.get(method, "gray")
            ax.bar(x + offset, values, width, label=method, color=color, alpha=0.85)

        ax.set_xlabel("Model", fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f"{label} by Post-Hoc Method", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=9, rotation=15, ha="right")
        ax.legend(fontsize=6, loc="lower right")
        ax.grid(axis="y", alpha=0.3)
        ymin = comparison_df[metric].min() - 0.01
        ymax = comparison_df[metric].max() + 0.01
        ax.set_ylim(ymin, ymax)

    matplotlib.pyplot.tight_layout()
    path = os.path.join(save_dir, "T9_overall_performance.png")
    matplotlib.pyplot.savefig(path, dpi=150)
    matplotlib.pyplot.close()
    print(f"Overall performance bars -> {path}")


def plot_before_after_auprc_bars(comparison_df: pandas.DataFrame, save_dir: str = None):
    save_dir = save_dir or config9.OUTPUT_DIR
    models = comparison_df["model"].unique()
    methods = comparison_df["method"].unique()
    n_models = len(models)
    n_methods = len(methods)

    x = numpy.arange(n_models)
    width = 0.8 / n_methods

    fig, ax = matplotlib.pyplot.subplots(figsize=(max(10, n_models * 3), 6))

    for i, method in enumerate(methods):
        values = []
        for model in models:
            row = comparison_df[
                (comparison_df["model"] == model) &
                (comparison_df["method"] == method)
            ]
            values.append(row["max_auprc_gap"].values[0] if len(row) else 0)

        offset = (i - n_methods / 2 + 0.5) * width
        color = METHOD_COLORS.get(method, "gray")
        ax.bar(x + offset, values, width, label=method, color=color, alpha=0.85)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Max AUPRC Fairness Gap", fontsize=12)
    ax.set_title("AUPRC Fairness Gap: Before vs After Post-Hoc Methods", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9, rotation=15, ha="right")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    matplotlib.pyplot.tight_layout()
    path = os.path.join(save_dir, "T9_before_after_auprc_bars.png")
    matplotlib.pyplot.savefig(path, dpi=150)
    matplotlib.pyplot.close()
    print(f"Before & after AUPRC bars -> {path}")


def generate_all_plots(comparison_df: pandas.DataFrame, all_results: dict, save_dir: str = None):
    save_dir = save_dir or config9.OUTPUT_DIR
    print(f"\n{config9.BAG} Generating Task 9 Plots {config9.BAG}")

    TradeoffPlot(comparison_df)
    plot_before_after_bars(comparison_df, save_dir)
    plot_before_after_auprc_bars(comparison_df, save_dir)
    plot_tpr_gap_comparison(comparison_df, save_dir)
    plot_overall_performance(comparison_df, save_dir)

    # heatmaps and per-attribute TPR per model
    for model_name in all_results.keys():
        plot_method_heatmap(all_results, model_name, save_dir)
        plot_auprc_heatmap(all_results, model_name, save_dir)
        plot_tpr_by_attribute(all_results, model_name, save_dir)
