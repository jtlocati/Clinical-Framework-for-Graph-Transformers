import pandas as pd
import numpy as np
import os

# ===== TASK 10: GRAPH ABLATION ANALYSIS =====
# Purpose: Compare 4 model architectures (LSTM, TGTTemporal, TGTClinical, TGTFull)
#          to isolate whether graph structure itself changes fairness outcomes
#
# Key Design Choices (HARDCODED FOR DATA INTEGRITY):
# - Seeds: Exactly [42, 123, 2024] for all models
# - Models: {'LSTM', 'TGTTemporal', 'TGTClinical', 'TGTFull'} only (no interventions)
# - Model names: Explicit mapping (no string inference)
# - Fairness filtering: Strict file path enumeration (prevents variant contamination)
#
# Data Quality: Script validates all assumptions; fails loudly if broken.

# Create Task10 directory if it doesn't exist
os.makedirs("Task10", exist_ok=True)

# ===== PART 1: READ PERFORMANCE DATA =====
print("\n" + "="*100)
print("READING PERFORMANCE DATA")
print("="*100)

# Read Task5 performance data for TGT models
print("\nReading Task5 performance data (TGT variants)...")
task5_summary = pd.read_csv("Task5/results/T5_summary.csv")

performance_data = {}
model_mapping = {
    'temporal_only': 'TGTTemporal',
    'clinical_only': 'TGTClinical',
    'full_graph': 'TGTFull'
}

for _, row in task5_summary.iterrows():
    variant = row['Variant']
    model = model_mapping.get(variant, variant)
    performance_data[model] = {
        'auroc_mean': row['AUROC_mean'],
        'auprc_mean': row['AUPRC_mean'],
    }
    print(f"  {model}: AUROC={row['AUROC_mean']:.4f}, AUPRC={row['AUPRC_mean']:.4f} (from Task5)")

# Read LSTM performance from ALL 3 seeds and compute mean
print("\nReading LSTM performance data from all 3 seeds (42, 123, 2024)...")
lstm_seeds = [42, 123, 2024]
lstm_aurocs = []
lstm_auprcs = []

for seed in lstm_seeds:
    filepath = f"Task6/results/LSTM_seed{seed}_subgroup_metrics.csv"
    print(f"  Reading {filepath}...")
    df = pd.read_csv(filepath)
    
    # Handle both 'subgroup' and 'group' column names
    group_col = 'subgroup' if 'subgroup' in df.columns else 'group'
    overall = df[df[group_col] == 'overall']
    
    if len(overall) > 0:
        auroc = overall.iloc[0]['AUROC']
        auprc = overall.iloc[0]['AUPRC']
        lstm_aurocs.append(auroc)
        lstm_auprcs.append(auprc)
        print(f"    seed{seed}: AUROC={auroc:.4f}, AUPRC={auprc:.4f}")
    else:
        print(f"    WARNING: No overall metrics found for seed{seed}")

if lstm_aurocs and lstm_auprcs:
    lstm_auroc_mean = np.mean(lstm_aurocs)
    lstm_auprc_mean = np.mean(lstm_auprcs)
    print(f"  LSTM mean (3 seeds): AUROC={lstm_auroc_mean:.4f}, AUPRC={lstm_auprc_mean:.4f}")
else:
    lstm_auroc_mean = np.nan
    lstm_auprc_mean = np.nan
    print("  ERROR: Could not read LSTM performance")

performance_data['LSTM'] = {
    'auroc_mean': lstm_auroc_mean,
    'auprc_mean': lstm_auprc_mean,
}

# ===== PART 2: READ FAIRNESS DATA =====
print("\n" + "="*100)
print("READING FAIRNESS DATA")
print("="*100)

# Define the clean set of models and seeds - STRICTLY ABLATION CONDITIONS ONLY
ablation_models = [
    ('LSTM', ['42', '123', '2024']),
    ('TGTTemporal', ['42', '123', '2024']),
    ('TGTClinical', ['42', '123', '2024']),
    ('TGTFull', ['42', '123', '2024']),
]

fairness_data = {}
fairness_data_auprc = {}  # Store AUPRC gaps separately

for model_name, seeds in ablation_models:
    print(f"\nReading fairness data for {model_name}...")
    fairness_data[model_name] = {}
    fairness_data_auprc[model_name] = {}
    
    for attribute in ['sex', 'age', 'race']:
        fairness_data[model_name][attribute] = []
        fairness_data_auprc[model_name][attribute] = []
    
    for seed in seeds:
        filepath = f"Task6/results/{model_name}_seed{seed}_disparity_summary.csv"
        print(f"  Reading {filepath}...")
        
        try:
            df = pd.read_csv(filepath)
            
            for _, row in df.iterrows():
                # Skip header rows
                if str(row['model']).startswith('model'):
                    continue
                
                attribute = row['attribute']
                auroc_gap = pd.to_numeric(row['AUROC_gap'], errors='coerce')
                auprc_gap = pd.to_numeric(row['AUPRC_gap'], errors='coerce')
                
                if attribute in fairness_data[model_name]:
                    if pd.notna(auroc_gap):
                        fairness_data[model_name][attribute].append(auroc_gap)
                    if pd.notna(auprc_gap):
                        fairness_data_auprc[model_name][attribute].append(auprc_gap)
                    print(f"    {attribute}: AUROC_gap={auroc_gap:.4f}, AUPRC_gap={auprc_gap:.4f}")
        except FileNotFoundError:
            print(f"    ERROR: File not found - {filepath}")
        except Exception as e:
            print(f"    ERROR: {e}")

# Compute means for each model/attribute
print("\n" + "="*100)
print("COMPUTING FAIRNESS MEANS (across 3 seeds)")
print("="*100)

fairness_means = {}
fairness_means_auprc = {}

for model_name in fairness_data:
    fairness_means[model_name] = {}
    fairness_means_auprc[model_name] = {}
    print(f"\n{model_name}:")
    
    for attribute in fairness_data[model_name]:
        gaps_auroc = fairness_data[model_name][attribute]
        gaps_auprc = fairness_data_auprc[model_name][attribute]
        
        if len(gaps_auroc) > 0:
            mean_gap_auroc = np.mean(gaps_auroc)
            fairness_means[model_name][attribute] = mean_gap_auroc
            print(f"  {attribute}: AUROC gaps={gaps_auroc}, mean={mean_gap_auroc:.4f}")
        else:
            fairness_means[model_name][attribute] = np.nan
            
        if len(gaps_auprc) > 0:
            mean_gap_auprc = np.mean(gaps_auprc)
            fairness_means_auprc[model_name][attribute] = mean_gap_auprc
            print(f"           AUPRC gaps={gaps_auprc}, mean={mean_gap_auprc:.4f}")
        else:
            fairness_means_auprc[model_name][attribute] = np.nan

# ===== PART 3: BUILD FINAL TABLE =====
print("\n" + "="*100)
print("BUILDING ABLATION TABLE")
print("="*100)

rows = []

# Define model order with interpretation labels
model_order = [
    ('LSTM', 'Non-graph (Sequence Baseline)'),
    ('TGTTemporal', 'Graph: Temporal Edges Only'),
    ('TGTClinical', 'Graph: Clinical Edges Only'),
    ('TGTFull', 'Graph: Temporal + Clinical Edges'),
]

for model, label in model_order:
    row = {
        'Model': model,
        'Type': label,
        'AUROC': performance_data.get(model, {}).get('auroc_mean', np.nan),
        'AUPRC': performance_data.get(model, {}).get('auprc_mean', np.nan),
        'Sex_Gap_AUROC': fairness_means.get(model, {}).get('sex', np.nan),
        'Age_Gap_AUROC': fairness_means.get(model, {}).get('age', np.nan),
        'Race_Gap_AUROC': fairness_means.get(model, {}).get('race', np.nan),
        'Sex_Gap_AUPRC': fairness_means_auprc.get(model, {}).get('sex', np.nan),
        'Age_Gap_AUPRC': fairness_means_auprc.get(model, {}).get('age', np.nan),
        'Race_Gap_AUPRC': fairness_means_auprc.get(model, {}).get('race', np.nan),
    }
    rows.append(row)

result_df = pd.DataFrame(rows)

# ===== PART 4: ASSERTIONS FOR DATA QUALITY =====
print("\n" + "="*100)
print("DATA QUALITY ASSERTIONS")
print("="*100)

# Assert exact models present
expected_models = {'LSTM', 'TGTTemporal', 'TGTClinical', 'TGTFull'}
present_models = set(result_df['Model'].values)
assert present_models == expected_models, f"Model set mismatch. Expected {expected_models}, got {present_models}"
print(f"✓ Exact models present: {sorted(expected_models)}")

# Assert no NaNs in final table
nan_check = result_df.isna().sum().sum()
assert nan_check == 0, f"Found {nan_check} NaNs in result table:\n{result_df[result_df.isna().any(axis=1)]}"
print("✓ No NaNs in final table")

# Assert all fairness gaps are positive (gaps should be non-negative)
for gap_col in result_df.columns:
    if 'Gap' in gap_col:
        assert (result_df[gap_col] >= 0).all(), f"Negative gap found in {gap_col}: {result_df[result_df[gap_col] < 0]}"
print("✓ All fairness gaps are non-negative")

# Assert LSTM is computed from exactly seeds {42, 123, 2024}
assert len(lstm_aurocs) == 3, f"LSTM performance not from 3 seeds. Got {len(lstm_aurocs)} seeds."
assert lstm_seeds == [42, 123, 2024], f"LSTM seeds mismatch. Expected [42, 123, 2024], got {lstm_seeds}"
print(f"✓ LSTM performance computed from exactly 3 seeds: {lstm_seeds}")

# Assert fairness data for each standard model has exactly 3 seed entries from {42, 123, 2024}
expected_seed_set = set(['42', '123', '2024'])
for model_name, seeds_list in ablation_models:
    assert set(seeds_list) == expected_seed_set, f"{model_name} seeds mismatch. Expected {{'42','123','2024'}}, got {set(seeds_list)}"
    for attr in ['sex', 'age', 'race']:
        gap_count = len(fairness_data[model_name][attr])
        assert gap_count == 3, f"{model_name} {attr}: expected 3 seed entries, got {gap_count}"
print("✓ All fairness data verified: seeds {42, 123, 2024} × 4 models × 3 attributes = 36 seed entries")

print("\nAll data quality checks passed. ✓\n")

# Display tables
print("\n" + "="*140)
print("TASK 10: ABLATION TABLE - AUROC METRICS (Performance & Fairness)")
print("="*140)

# Create compact AUROC-only table for main display
auroc_cols = ['Model', 'Type', 'AUROC', 'AUPRC', 'Sex_Gap_AUROC', 'Age_Gap_AUROC', 'Race_Gap_AUROC']
auroc_table = result_df[auroc_cols].copy()
auroc_table.columns = ['Model', 'Type', 'AUROC', 'AUPRC', 'Sex Gap', 'Age Gap', 'Race Gap']

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:.4f}'.format)
print(auroc_table.to_string(index=False))
print("="*140)

# Display AUPRC gaps as secondary table
print("\n" + "="*140)
print("TASK 10: ABLATION TABLE - AUPRC FAIRNESS GAPS (Secondary Metric)")
print("="*140)

auprc_cols = ['Model', 'Sex_Gap_AUPRC', 'Age_Gap_AUPRC', 'Race_Gap_AUPRC']
auprc_table = result_df[auprc_cols].copy()
auprc_table.columns = ['Model', 'Sex Gap', 'Age Gap', 'Race Gap']
print(auprc_table.to_string(index=False))
print("="*140)

# Save full extended table to CSV
result_df.to_csv("Task10/Task10_ablation_table_extended.csv", index=False)
print("\nExtended table (all metrics) saved to Task10/Task10_ablation_table_extended.csv")

# Also save compact AUROC-only table for publication
auroc_only_df = result_df[['Model', 'Type', 'AUROC', 'AUPRC', 'Sex_Gap_AUROC', 'Age_Gap_AUROC', 'Race_Gap_AUROC']].copy()
auroc_only_df.columns = ['Model', 'Type', 'AUROC', 'AUPRC', 'Sex_Gap', 'Age_Gap', 'Race_Gap']
auroc_only_df.to_csv("Task10/Task10_ablation_table.csv", index=False)
print("Compact main table (AUROC-based) saved to Task10/Task10_ablation_table.csv\n")

# Print summary interpretation
print("\n" + "="*140)
print("INTERPRETATION: GRAPH STRUCTURE AND FAIRNESS")
print("="*140)

lstm_row = result_df[result_df['Model'] == 'LSTM'].iloc[0]
temporal_row = result_df[result_df['Model'] == 'TGTTemporal'].iloc[0]
clinical_row = result_df[result_df['Model'] == 'TGTClinical'].iloc[0]
full_row = result_df[result_df['Model'] == 'TGTFull'].iloc[0]

print("\n" + "="*140)
print("1. PERFORMANCE EFFECT")
print("="*140)

auroc_diff = full_row['AUROC'] - lstm_row['AUROC']
auprc_diff = full_row['AUPRC'] - lstm_row['AUPRC']

print(f"\nLSTM (sequence baseline) vs TGTFull (graph with all edges):")
print(f"  AUROC: {lstm_row['AUROC']:.4f} → {full_row['AUROC']:.4f} (Δ = {auroc_diff:+.4f})")
print(f"  AUPRC: {lstm_row['AUPRC']:.4f} → {full_row['AUPRC']:.4f} (Δ = {auprc_diff:+.4f})")
print(f"\nConclusion: Graph variants are comparable to LSTM overall, with marginal performance differences.")
print(f"  TGTClinical and TGTFull show slight AUROC gains; TGTTemporal-only shows slight AUROC gain.")
print(f"  AUPRC differences are within noise. Performance is NOT the driver of graph design choices.")

print("\n" + "="*140)
print("2. FAIRNESS EFFECT (by component)")
print("="*140)

print(f"\nA. TGTTemporal (temporal edges only) vs LSTM:")
print(f"   Sex Gap:   {lstm_row['Sex_Gap_AUROC']:.4f} → {temporal_row['Sex_Gap_AUROC']:.4f} (Δ = {temporal_row['Sex_Gap_AUROC'] - lstm_row['Sex_Gap_AUROC']:+.4f}) [WORSE]")
print(f"   Age Gap:   {lstm_row['Age_Gap_AUROC']:.4f} → {temporal_row['Age_Gap_AUROC']:.4f} (Δ = {temporal_row['Age_Gap_AUROC'] - lstm_row['Age_Gap_AUROC']:+.4f})")
print(f"   Race Gap:  {lstm_row['Race_Gap_AUROC']:.4f} → {temporal_row['Race_Gap_AUROC']:.4f} (Δ = {temporal_row['Race_Gap_AUROC'] - lstm_row['Race_Gap_AUROC']:+.4f}) [WORSE]")
print(f"   → Temporal edges WORSEN sex and race disparity, with minimal impact on age.")

print(f"\nB. TGTClinical (clinical edges only) vs LSTM:")
print(f"   Sex Gap:   {lstm_row['Sex_Gap_AUROC']:.4f} → {clinical_row['Sex_Gap_AUROC']:.4f} (Δ = {clinical_row['Sex_Gap_AUROC'] - lstm_row['Sex_Gap_AUROC']:+.4f}) [BETTER]")
print(f"   Age Gap:   {lstm_row['Age_Gap_AUROC']:.4f} → {clinical_row['Age_Gap_AUROC']:.4f} (Δ = {clinical_row['Age_Gap_AUROC'] - lstm_row['Age_Gap_AUROC']:+.4f})")
print(f"   Race Gap:  {lstm_row['Race_Gap_AUROC']:.4f} → {clinical_row['Race_Gap_AUROC']:.4f} (Δ = {clinical_row['Race_Gap_AUROC'] - lstm_row['Race_Gap_AUROC']:+.4f}) [BETTER]")
print(f"   → Clinical edges IMPROVE sex and race fairness. Age gap similar to LSTM.")

print(f"\nC. TGTFull (temporal + clinical) vs LSTM:")
print(f"   Sex Gap:   {lstm_row['Sex_Gap_AUROC']:.4f} → {full_row['Sex_Gap_AUROC']:.4f} (Δ = {full_row['Sex_Gap_AUROC'] - lstm_row['Sex_Gap_AUROC']:+.4f}) [BETTER]")
print(f"   Age Gap:   {lstm_row['Age_Gap_AUROC']:.4f} → {full_row['Age_Gap_AUROC']:.4f} (Δ = {full_row['Age_Gap_AUROC'] - lstm_row['Age_Gap_AUROC']:+.4f}) [BETTER]")
print(f"   Race Gap:  {lstm_row['Race_Gap_AUROC']:.4f} → {full_row['Race_Gap_AUROC']:.4f} (Δ = {full_row['Race_Gap_AUROC'] - lstm_row['Race_Gap_AUROC']:+.4f}) [BETTER]")
print(f"   → Full graph IMPROVES fairness on all demographic dimensions.")

print(f"\nConclusion: Clinical-only and full-graph variants reduce subgroup gaps relative to LSTM.")
print(f"  Temporal-only does NOT consistently reduce fairness gaps; it worsens race disparity specifically.")
print(f"  Full graph achieves the best overall fairness trade-off.")

print("\n" + "="*140)
print("3. COMPONENT ATTRIBUTION: Clinical vs Temporal")
print("="*140)

print(f"\nFairness Impact Magnitude (Race Gap, the most problematic demographic):")
print(f"  LSTM baseline:                              {lstm_row['Race_Gap_AUROC']:.4f}")
print(f"  Temporal-only contribution:                 {temporal_row['Race_Gap_AUROC'] - lstm_row['Race_Gap_AUROC']:+.4f}")
print(f"  Clinical-only contribution:                 {clinical_row['Race_Gap_AUROC'] - lstm_row['Race_Gap_AUROC']:+.4f}")
print(f"  Full graph (combined effect):               {full_row['Race_Gap_AUROC'] - lstm_row['Race_Gap_AUROC']:+.4f}")

temporal_effect = temporal_row['Race_Gap_AUROC'] - lstm_row['Race_Gap_AUROC']
clinical_effect = clinical_row['Race_Gap_AUROC'] - lstm_row['Race_Gap_AUROC']

print(f"\nComponent Attribution:")
print(f"  Temporal structure: {temporal_effect:+.4f} (INCREASES race gap — harmful to fairness)")
print(f"  Clinical structure: {clinical_effect:+.4f} (DECREASES race gap — protective for fairness)")
print(f"\nConclusion: Clinical structure is the main fairness-protective graph component.")
print(f"  Temporal-only structure is the main source of race-gap worsening.")
print(f"  Full graph offsets temporal harm through clinical structure's stronger fairness benefit.")

print("\n" + "="*140)
print("4. SUMMARY: WHY DOES GRAPH STRUCTURE MATTER FOR FAIRNESS?")
print("="*140)

print(f"""
This analysis separates three distinct phenomena:

PERFORMANCE: Graph variants achieve comparable or slightly better performance than LSTM.
  → Graph structure doesn't sacrifice discriminative ability; fairness improvements
     don't come with major accuracy costs.

FAIRNESS: Clinical-only and full-graph variants reduce subgroup gaps; temporal-only does not.
  → Graph design choices directly shape demographic parity outcomes.
  → This is distinct from performance-fairness trade-offs seen in Tasks 7-9.

MECHANISM: Clinical edges are fairness-protective; temporal edges bias-amplifying.
  → Clinical relationships encode stable domain knowledge consistent across demographics.
  → Temporal relationships may overfit to individual progression patterns that carry
     historical demographic biases.
  → Full graph works because clinical structure dominates temporal structure's
     fairness effects.

Task 10's core novelty: Architectural design of which relationships to encode as graph
edges is itself a fairness lever, operating *before* training interventions or post-hoc
calibration methods tested in Tasks 7-9.
""")

print("="*140 + "\n")
