"""
Task 1.1a - Corrected fairness re-evaluation (fixed-split, no retraining).

Re-evaluates subgroup AUROC/AUPRC gaps from the fixed-split per-patient predictions
in Task10/results/*_test_predictions.csv (produced by Task10/task10_pred.py with
SPLIT_SEED=42, only model init varying across seeds 42/123/2024).

This deliberately does NOT touch the Task9 prediction cache (its TGTFull is seed-42
only) or any Task6/results summary / Task10_GSA_* table (those carry the stale
per-seed-resplit numbers). Subgroup membership comes from the same pipeline utilities
the paper uses (eval6.load_demographics / BuildMasks), and gaps are computed with the
repo's own metric functions (eval6.SubgroupMetrics / eval6.FairnessDif).

Outputs:
  outputs/corrected_gaps.csv          per-(model,attribute,seed) rows
  outputs/corrected_gaps_summary.csv  3-seed mean+/-std + LSTM-TGTFull paired gap diffs
  outputs/corrected_eval_notes.txt    claim (i) verdict, Table 5 check, subgroup counts
"""
import os
import sys
import contextlib
import io

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import Task6.config as c6          # noqa: E402
import Task6.eval as eval6         # noqa: E402

PRED_DIR = os.path.join(REPO_ROOT, "Task10", "results")
OUT_DIR = os.path.join(REPO_ROOT, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = ["LSTM", "TGTTemporal", "TGTClinical", "TGTFull"]
SEEDS = [42, 123, 2024]
ATTRS = ["sex", "age", "race"]

# Paper Table 5 overall numbers (matched-training ablation) for cross-check.
TABLE5 = {
    "LSTM": (0.8281, 0.4731),
    "TGTTemporal": (0.8231, 0.4532),
    "TGTClinical": (0.8324, 0.4818),
    "TGTFull": (0.8328, 0.4825),
}


def pred_path(model, seed):
    return os.path.join(PRED_DIR, f"{model}_seed{seed}_test_predictions.csv")


def load_fixed_split_ids():
    """Canonical patient id order, asserting every model/seed file shares the same set."""
    ref = pd.read_csv(pred_path("LSTM", 42))["stay_id"].to_numpy()
    ref_set = set(ref.tolist())
    for m in MODELS:
        for s in SEEDS:
            ids = pd.read_csv(pred_path(m, s))["stay_id"].to_numpy()
            if set(ids.tolist()) != ref_set:
                raise SystemExit(
                    f"FIXED-SPLIT VIOLATION: {m}_seed{s} stay_id set differs from LSTM_seed42; "
                    "these predictions are not on a common split."
                )
    return ref


def load_probs(model, seed, ids):
    """Return (y_true, y_prob) aligned to the canonical id order."""
    df = pd.read_csv(pred_path(model, seed)).set_index("stay_id").loc[ids]
    return df["y_true"].to_numpy(), df["y_score"].to_numpy()


def main():
    print(f"{c6.BAG}\nTask 1.1a - corrected fairness re-evaluation (fixed split)\n{c6.BAG}")
    ids = load_fixed_split_ids()
    print(f"Fixed-split test set: n={len(ids):,}; all 12 model/seed files share one cohort.")

    # Build subgroup masks ONCE on the fixed cohort (pipeline utilities).
    with contextlib.redirect_stdout(io.StringIO()):
        demographics = eval6.load_demographics()
        masks = eval6.BuildMasks(ids, demographics)

    # Confirmed subgroup counts (identical across seeds by construction of the fixed split).
    subgroup_counts = {name: int(m.sum()) for name, m in masks.items()}

    # ---- per-seed evaluation via the repo's own metric functions ----
    per_seed_rows = []          # corrected_gaps.csv rows
    # gaps[model][attr][metric] -> list over seeds; overall[model][metric] -> list over seeds
    gap_store = {m: {a: {"auroc": [], "auprc": []} for a in ATTRS} for m in MODELS}
    overall_store = {m: {"auroc": [], "auprc": []} for m in MODELS}

    for seed in SEEDS:
        predictions = {m: load_probs(m, seed, ids) for m in MODELS}
        # eval6.SubgroupMetrics: overall + per-subgroup AUROC/AUPRC with n<30 + single-class
        # rows set to NaN. eval6.FairnessDif: max-min gap per (model, attribute) over valid groups.
        with contextlib.redirect_stdout(io.StringIO()):
            metrics = eval6.SubgroupMetrics(predictions, masks)
            fair = eval6.FairnessDif(metrics)

        for model in MODELS:
            overall_row = metrics[(metrics.model == model) & (metrics.subgroup == "overall")].iloc[0]
            o_auroc, o_auprc = float(overall_row.auroc), float(overall_row.auprc)
            overall_store[model]["auroc"].append(o_auroc)
            overall_store[model]["auprc"].append(o_auprc)
            for attr in ATTRS:
                fr = fair[(fair.model == model) & (fair.attribute == attr)]
                g_auroc = float(fr.auroc_gap.iloc[0]) if len(fr) else np.nan
                g_auprc = float(fr.auprc_gap.iloc[0]) if len(fr) else np.nan
                gap_store[model][attr]["auroc"].append(g_auroc)
                gap_store[model][attr]["auprc"].append(g_auprc)
                per_seed_rows.append({
                    "model": model,
                    "attribute": attr,
                    "seed": seed,
                    "overall_auroc": round(o_auroc, 6),
                    "overall_auprc": round(o_auprc, 6),
                    "subgroup_gap_auroc": round(g_auroc, 6),
                    "subgroup_gap_auprc": round(g_auprc, 6),
                })

    per_seed_df = pd.DataFrame(per_seed_rows)
    per_seed_df.to_csv(os.path.join(OUT_DIR, "corrected_gaps.csv"), index=False)

    # ---- 3-seed mean +/- std summary ----
    def ms(values):
        arr = np.asarray(values, dtype=float)
        return float(np.mean(arr)), float(np.std(arr, ddof=1))

    summary_rows = []
    for model in MODELS:
        om, osd = ms(overall_store[model]["auroc"])
        opm, opsd = ms(overall_store[model]["auprc"])
        for attr in ATTRS:
            gm, gsd = ms(gap_store[model][attr]["auroc"])
            pm, psd = ms(gap_store[model][attr]["auprc"])
            summary_rows.append({
                "model": model,
                "attribute": attr,
                "overall_auroc_mean": round(om, 4),
                "overall_auroc_std": round(osd, 4),
                "overall_auprc_mean": round(opm, 4),
                "overall_auprc_std": round(opsd, 4),
                "gap_auroc_mean": round(gm, 4),
                "gap_auroc_std": round(gsd, 4),
                "gap_auprc_mean": round(pm, 4),
                "gap_auprc_std": round(psd, 4),
            })
    summary_df = pd.DataFrame(summary_rows)

    # ---- paired LSTM - TGTFull gap difference, per seed (same fixed split) ----
    paired_rows = []
    for attr in ATTRS:
        lstm = np.asarray(gap_store["LSTM"][attr]["auroc"], dtype=float)
        full = np.asarray(gap_store["TGTFull"][attr]["auroc"], dtype=float)
        diff = lstm - full  # >0 => TGTFull smaller gap (better)
        signs = ["TGTFull_smaller" if d > 0 else "TGTFull_larger" for d in diff]
        paired_rows.append({
            "comparison": "LSTM_minus_TGTFull_auroc_gap",
            "attribute": attr,
            "diff_seed42": round(float(diff[0]), 4),
            "diff_seed123": round(float(diff[1]), 4),
            "diff_seed2024": round(float(diff[2]), 4),
            "mean_diff": round(float(diff.mean()), 4),
            "std_diff": round(float(diff.std(ddof=1)), 4),
            "n_seeds_TGTFull_smaller": int((diff > 0).sum()),
            "per_seed_sign": "|".join(signs),
        })
    paired_df = pd.DataFrame(paired_rows)

    # Write summary CSV: model/attribute means stacked above the paired-difference block.
    summary_path = os.path.join(OUT_DIR, "corrected_gaps_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    with open(summary_path, "a") as fh:
        fh.write("\n# Paired LSTM - TGTFull AUROC-gap difference (per seed, fixed split; "
                 "positive => TGTFull smaller gap)\n")
    paired_df.to_csv(summary_path, mode="a", index=False)

    # ---- notes / verdicts ----
    race_paired = paired_df[paired_df.attribute == "race"].iloc[0]
    claim_i_reverses = race_paired["n_seeds_TGTFull_smaller"] == 0
    lstm_race = summary_df[(summary_df.model == "LSTM") & (summary_df.attribute == "race")]["gap_auroc_mean"].iloc[0]
    full_race = summary_df[(summary_df.model == "TGTFull") & (summary_df.attribute == "race")]["gap_auroc_mean"].iloc[0]

    # Table 5 overall reproduction check.
    table5_lines = []
    table5_reproduces = True
    for model in MODELS:
        got_au = summary_df[(summary_df.model == model) & (summary_df.attribute == "race")]["overall_auroc_mean"].iloc[0]
        got_ap = summary_df[(summary_df.model == model) & (summary_df.attribute == "race")]["overall_auprc_mean"].iloc[0]
        exp_au, exp_ap = TABLE5[model]
        d_au = got_au - exp_au
        if abs(d_au) > 0.005:
            table5_reproduces = False
        table5_lines.append(
            f"  {model:12s} AUROC {got_au:.4f} (Table5 {exp_au:.4f}, d={d_au:+.4f}) | "
            f"AUPRC {got_ap:.4f} (Table5 {exp_ap:.4f}, d={got_ap-exp_ap:+.4f})"
        )

    race_groups = {k.replace("race_", ""): v for k, v in subgroup_counts.items() if k.startswith("race_")}
    sex_groups = {k.replace("sex_", ""): v for k, v in subgroup_counts.items() if k.startswith("sex_")}
    age_groups = {k.replace("age_", ""): v for k, v in subgroup_counts.items() if k.startswith("age_")}

    notes = []
    notes.append("=" * 78)
    notes.append("Task 1.1a - Corrected fairness re-evaluation notes (fixed-split Task10 preds)")
    notes.append("=" * 78)
    notes.append("")
    notes.append("CLAIM (i)  'TGTFull achieves the smallest race AUROC gap under matched training':")
    notes.append(f"  fixed-split 3-seed mean race AUROC gap -> LSTM {lstm_race:.4f} vs TGTFull {full_race:.4f}")
    notes.append(f"  paired LSTM-TGTFull race gap per seed: "
                 f"{race_paired['diff_seed42']:+.4f} / {race_paired['diff_seed123']:+.4f} / "
                 f"{race_paired['diff_seed2024']:+.4f}  (mean {race_paired['mean_diff']:+.4f} "
                 f"+/- {race_paired['std_diff']:.4f})")
    notes.append(f"  TGTFull smaller in {race_paired['n_seeds_TGTFull_smaller']}/3 seeds.")
    if claim_i_reverses:
        notes.append("  VERDICT: claim (i) REVERSES under the matched fixed split "
                     "(TGTFull race gap is LARGER than LSTM in all 3 seeds).")
    else:
        notes.append(f"  VERDICT: claim (i) does not cleanly reverse "
                     f"(TGTFull smaller in {race_paired['n_seeds_TGTFull_smaller']}/3 seeds); "
                     "inspect per-seed signs and std.")
    notes.append("")
    notes.append("  Paired sex/age AUROC-gap differences (LSTM - TGTFull, fixed split):")
    for attr in ["sex", "age"]:
        r = paired_df[paired_df.attribute == attr].iloc[0]
        notes.append(f"    {attr:4s}: {r['diff_seed42']:+.4f} / {r['diff_seed123']:+.4f} / "
                     f"{r['diff_seed2024']:+.4f}  (mean {r['mean_diff']:+.4f} +/- {r['std_diff']:.4f}; "
                     f"TGTFull smaller {r['n_seeds_TGTFull_smaller']}/3)")
    notes.append("")
    notes.append("TABLE 5 overall numbers (matched-training) reproduction check:")
    notes.extend(table5_lines)
    notes.append(f"  -> Table 5 overall AUROC {'REPRODUCES' if table5_reproduces else 'does NOT reproduce'} "
                 "from the fixed-split predictions (tolerance 0.005).")
    notes.append("")
    notes.append("CONFIRMED subgroup counts on the fixed split (identical across all 3 seeds,")
    notes.append("no longer drifting like the stale path's Asian 182/170/179):")
    notes.append(f"  sex : {sex_groups}")
    notes.append(f"  age : {age_groups}")
    notes.append(f"  race: {race_groups}")
    notes.append(f"  (race subgroups with n<{c6.MinGroup} excluded from gaps: "
                 f"{[g for g, n in race_groups.items() if n < c6.MinGroup]})")
    notes.append("")
    notes_text = "\n".join(notes)

    with open(os.path.join(OUT_DIR, "corrected_eval_notes.txt"), "w") as fh:
        fh.write(notes_text + "\n")

    print(notes_text)
    print(f"\nWrote:\n  {os.path.join('outputs', 'corrected_gaps.csv')}"
          f"\n  {os.path.join('outputs', 'corrected_gaps_summary.csv')}"
          f"\n  {os.path.join('outputs', 'corrected_eval_notes.txt')}")


if __name__ == "__main__":
    main()
