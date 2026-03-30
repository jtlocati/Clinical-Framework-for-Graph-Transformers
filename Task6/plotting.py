#plots things like the fairness table, heat map, and bar plot

import os
import numpy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot
import config
import pandas

def Heatmap(gap: pandas.DataFrame, save: str = None):
    if save is not None:
        DIRECTORY = save
    else:
        DIRECTORY = config.RESULTS_DIRECTORY
    
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
            path = os.path.join(DIRECTORY, "T6_farnessHM.png")
            matplotlib.pyplot.close()
            print(f"HEATMAP SAVED -> {path}")


def SubgroupBar(metrics: pandas.DataFrame, attrobute:str, save: str = config.RESULTS_DIRECTORY):
    sub = metrics[(metrics["attribute"] == attrobute & metrics["auroc"].notna)].copy()

    if len(sub) == 0:
        print(f"NO DATA FOR {attrobute}")
        return
     