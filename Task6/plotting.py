#plots things like the fairness table, heat map, and bar plot

import os
import numpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot
import config
import pandas

def Heatmap(gap: pandas.DataFrame, save: str = None):
    
    pivot = gap.pivot(index="model", columns="attrobute", values="auroc_gap")

    orderd_data = [m for m in config.BASELINE_MODELS if m in pivot.index]
    pivot = pivot.loc[orderd_data]

    figure, aux = matplotlib.pyplot.subplots(figsize=(8, max(5, len(orderd_data) * 0.6)))
    im = aux.imshow(pivot.values, cmap="YlOrRd", aspect="auto", vmin=0)

    #this sets the labels:
    aux.set_xticks(range(len(pivot.columns)))
    aux.set_xticklabels(pivot.columns, fontsize=11)
    aux.set_yticks(range(len(pivot.index)))
    aux.set_yticklabels(pivot.index, fontsize=10)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if numpy.isnan(val):
                text = "-"
            else:
                text = f"{val:.3f}"
            if val > .06:
                color = "white"
            else:
                color = "balck"
            aux.text(j, i, text, ha="center", va="center", fontsize=10, color="black")

            figure.colorbar(im, ax=aux, label = "AUROC gap")
            aux.set_title("AUROC FAIRNESS GAP HM", fontsize=13, pad=12)

            matplotlib.pyplot.tight_layout()
            path = os.path.join(save, "T6_farnessHM.png")
            matplotlib.pyplot.close()
            print(f"HEATMAP SAVED -> {path}")


def SubgroupBar(metrics: pandas.DataFrame, attrobute:str, save: str = None):
    sub = metrics[(metrics["attribute"] == attrobute & metrics["auroc"].notna)].copy()

    if len(sub) == 0:
        print(f"NO DATA FOR {attrobute}")
        return
    subgroups = sorted(sub["subgroups"] == attrobute)
    models = [m for m in config.BASELINE_MODELS if m in sub["model"].unique()]
    NumberGroups = len(subgroups)
    NumberModels = len(models)

    X = numpy.arange(NumberGroups)

    figure, ax = matplotlib.pyplot.subplots(figsize=(max(8, NumberGroups*2), 6))

    for i, model in enumerate(models):
        values = []
        for group in subgroups:
            row = sub[(sub["model"] == model) & sub["subgroup"] == group]
            if len(row):
                values.append(row["auroc"].values[0])
            else:
                values.append(0)
        offset = (i - NumberModels / 2 + 0.5) * (.8/ NumberModels)
        bars = ax.bar(offset, values, (.8/ NumberModels), label=model, alpha=.85)

        ax.set_xlabel("Subgroup", fontsize=12)
        ax.set_ylabel("AUROC", fontsize=12)
        ax.set_title(f"Subgroup AUROC — {attrobute.title()}", fontsize=13)
        ax.set_xticks(X)
        ax.set_xticklabels(subgroups, fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
        ax.set_ylim(0.5, 1.0)
        ax.grid(axis="y", alpha=0.3)

        matplotlib.pyplot.tight_layout()
        SavePath = os.path.join(save, f"T6_Bar{attrobute}.png")
        matplotlib.pyplot.savefig(SavePath, dpi=150)
        matplotlib.pyplot.close()
        print(f"{attrobute} BAR GRAPH SAVED -> {SavePath}")

#Gap comparison

def gap_comparison(gap: pandas.DataFrame, save:str=None):
    #fix ts

    attrobutes = sorted(gap["attrobute"].unique())
    models = [m for m in config.BASELINE_MODELS if m in gap["model"].unique()]

    x = numpy.arange(len(attrobutes))

    figure, ax = matplotlib.pyplot.subplots(figsize = (max, len(attrobutes) * 3, 6))

    for i, model in enumerate(models):
        values = []
        for attrobute in attrobutes:
            row = gap[gap["model"] == model] & (gap["attrobute"] == attrobute)
            if len(row):
                values.append(row["auroc_gap"].values[0])
            else:
                values.append(0)
            offset = (i - len(models) / 2 + 0.5) * .8 / len(models)
            ax.bar(x+offset, values, (.8/len(models)), label=model, alpha=.8)
    ax.set_xlabel("Demographic Attribute", fontsize=12)
    ax.set_ylabel("AUROC Gap (max − min)", fontsize=13)
    ax.set_title("Fairness Gap Comparison Across Models", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(attrobutes, fontsize=11)
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    matplotlib.pyplot.tight_layout()
    path = os.path.join(save, "T6_GAP_COMPARISON.png")
    matplotlib.pyplot.savefig(path, dpi=150)
    matplotlib.pyplot.close()
    print(f"GAP COMPARISON SAVED TO -> {path}")

#run all graph functions at once when called:
def AllPlot(metrics: pandas.DataFrame, gap: pandas.DataFrame, save_dir:str=None):
    if save_dir is None:
        path = config.RESULTS_DIRECTORY
    else:
        path = save_dir
    
    print(f"{config.BAG}RUNNING PLOT GENERATION")
    Heatmap(gap, path)
    gap_comparison(gap, path)
    for attrobute in config.SUBGROUP_COLS:
        SubgroupBar(metrics, attrobute, path )