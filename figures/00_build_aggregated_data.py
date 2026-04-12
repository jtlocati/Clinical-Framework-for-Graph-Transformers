"""
Build aggregated result tables from Task 5, 6, and 10 outputs.
Creates the five CSVs needed for figure generation:
- main_results.csv
- subgroup_results.csv
- fairness_summary.csv
- intervention_results.csv
- demographics_counts.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Paths
REPO_ROOT = Path(__file__).resolve().parents[1]
TASK5_DIR = REPO_ROOT / "Task5" / "results"
TASK6_DIR = REPO_ROOT / "Task6" / "results"
TASK10_DIR = REPO_ROOT / "Task10"
TESTS_DIR = REPO_ROOT / "Tests" / "outputs"
FIGURES_DIR = REPO_ROOT / "figures"
DATA_DIR = FIGURES_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("AGGREGATING RESULTS FOR FIGURE GENERATION")
print("="*80)

# ============================================================================
# 1. BUILD main_results.csv
# ============================================================================
print("\n[1] Building main_results.csv...")

main_results = []

# Task 5: TGT variants (temporal_only, clinical_only, full_graph)
task5_runs = pd.read_csv(TASK5_DIR / "T5_all_runs.csv")
for _, row in task5_runs.iterrows():
    variant_map = {
        'temporal_only': 'TGT-Temporal',
        'clinical_only': 'TGT-Clinical',
        'full_graph': 'TGT-Full'
    }
    model_name = variant_map.get(row['variant'], row['variant'])
    
    main_results.append({
        'dataset': 'MIMIC-IV',
        'task': 'ICU Mortality',  # This is the task in Task5
        'model': model_name,
        'seed': int(row['seed']),
        'split': 'test',
        'auroc': float(row['test_auroc']),
        'auprc': float(row['test_auprc']),
        'loss': None
    })

# Task 6: LSTM baseline (read from seed-wise files)
for seed in [42, 123, 2024]:
    subgroup_file = TASK6_DIR / f"LSTM_seed{seed}_subgroup_metrics.csv"
    if subgroup_file.exists():
        df = pd.read_csv(subgroup_file)
        overall = df[df['group'] == 'overall'].iloc[0]
        main_results.append({
            'dataset': 'MIMIC-IV',
            'task': 'ICU Mortality',
            'model': 'LSTM',
            'seed': seed,
            'split': 'test',
            'auroc': float(overall['AUROC']),
            'auprc': float(overall['AUPRC']),
            'loss': None
        })

# Note: For a complete paper, would also include LightGBM, LogReg, Transformer, ChronoFormer, PatchTST, Pretrained TST
# For now, focus on the core models from Task5/6

main_df = pd.DataFrame(main_results)
main_df.to_csv(DATA_DIR / "main_results.csv", index=False)
print(f"  -> {len(main_df)} rows in main_results.csv")

# ============================================================================
# 2. BUILD subgroup_results.csv
# ============================================================================
print("\n[2] Building subgroup_results.csv...")

subgroup_results = []

# LSTM subgroup metrics (all 3 seeds)
for seed in [42, 123, 2024]:
    subgroup_file = TASK6_DIR / f"LSTM_seed{seed}_subgroup_metrics.csv"
    if subgroup_file.exists():
        df = pd.read_csv(subgroup_file)
        for _, row in df.iterrows():
            if row['group'] != 'overall':
                # Parse the group name: e.g., "sex_F" -> attribute="sex", subgroup="F"
                parts = row['group'].split('_', 1)
                if len(parts) == 2:
                    attr_type, subgroup_val = parts
                else:
                    attr_type, subgroup_val = 'unknown', str(row['group'])
                
                subgroup_results.append({
                    'dataset': 'MIMIC-IV',
                    'task': 'ICU Mortality',
                    'model': 'LSTM',
                    'seed': seed,
                    'attribute_type': attr_type,
                    'subgroup': subgroup_val,
                    'n': int(row['count']),
                    'auroc': float(row['AUROC']) if pd.notna(row['AUROC']) else None,
                    'auprc': float(row['AUPRC']) if pd.notna(row['AUPRC']) else None,
                    'ece': None,
                    'tpr': None,
                    'fpr': None,
                    'ppv': None,
                    'npv': None
                })

# TGT variants subgroup metrics
for model_code, model_name in [('TGTTemporal', 'TGT-Temporal'), 
                               ('TGTClinical', 'TGT-Clinical'), 
                               ('TGTFull', 'TGT-Full')]:
    for seed in [42, 123, 2024]:
        subgroup_file = TASK6_DIR / f"{model_code}_seed{seed}_subgroup_metrics.csv"
        if subgroup_file.exists():
            df = pd.read_csv(subgroup_file)
            for _, row in df.iterrows():
                if row['group'] != 'overall':
                    group_val = row['group']
                    
                    parts = str(group_val).split('_', 1)
                    if len(parts) == 2:
                        attr_type, subgroup_val = parts
                    else:
                        attr_type, subgroup_val = 'unknown', str(group_val)
                    
                    subgroup_results.append({
                        'dataset': 'MIMIC-IV',
                        'task': 'ICU Mortality',
                        'model': model_name,
                        'seed': seed,
                        'attribute_type': attr_type,
                        'subgroup': subgroup_val,
                        'n': int(row['count']),
                        'auroc': float(row['AUROC']) if pd.notna(row['AUROC']) else None,
                        'auprc': float(row['AUPRC']) if pd.notna(row['AUPRC']) else None,
                        'ece': None,
                        'tpr': None,
                        'fpr': None,
                        'ppv': None,
                        'npv': None
                    })

subgroup_df = pd.DataFrame(subgroup_results)
subgroup_df.to_csv(DATA_DIR / "subgroup_results.csv", index=False)
print(f"  -> {len(subgroup_df)} rows in subgroup_results.csv")

# ============================================================================
# 3. BUILD fairness_summary.csv
# ============================================================================
print("\n[3] Building fairness_summary.csv...")

fairness_summary = []

# Read disparity summaries for LSTM and TGT variants
for model_code, model_name in [('LSTM', 'LSTM'), 
                               ('TGTTemporal', 'TGT-Temporal'), 
                               ('TGTClinical', 'TGT-Clinical'), 
                               ('TGTFull', 'TGT-Full')]:
    for seed in [42, 123, 2024]:
        disparity_file = TASK6_DIR / f"{model_code}_seed{seed}_disparity_summary.csv"
        if disparity_file.exists():
            df = pd.read_csv(disparity_file)
            # The disparity_summary has columns: model, seed, attribute, AUROC_gap, AUROC_std, AUPRC_gap, AUPRC_std, n_groups_used
            for _, row in df.iterrows():
                fairness_summary.append({
                    'dataset': 'MIMIC-IV',
                    'task': 'ICU Mortality',
                    'model': model_name,
                    'seed': seed,
                    'attribute_type': str(row['attribute']),
                    'auroc_gap_maxmin': float(row['AUROC_gap']),
                    'auprc_gap_maxmin': float(row['AUPRC_gap']),
                    'auroc_std': float(row['AUROC_std']),
                    'auprc_std': float(row['AUPRC_std']),
                    'worst_group_auroc': None,  # Not directly available
                    'worst_group_auprc': None   # Not directly available
                })

fairness_df = pd.DataFrame(fairness_summary)
fairness_df.to_csv(DATA_DIR / "fairness_summary.csv", index=False)
print(f"  -> {len(fairness_df)} rows in fairness_summary.csv")

# ============================================================================
# 4. BUILD intervention_results.csv
# ============================================================================
print("\n[4] Building intervention_results.csv...")

# For now, create a placeholder. Actual intervention results will be populated
# when Task6 intervention code is executed or results are available
intervention_results = []

# This would be populated from intervention results once available
# For now, create minimal structure
intervention_df = pd.DataFrame(intervention_results)
intervention_df.to_csv(DATA_DIR / "intervention_results.csv", index=False)
print(f"  -> {len(intervention_df)} rows in intervention_results.csv (placeholder)")

# ============================================================================
# 5. BUILD demographics_counts.csv
# ============================================================================
print("\n[5] Building demographics_counts.csv...")

# Read demographics file
demographics = pd.read_csv(TESTS_DIR / "Task2Demographics.csv")

# Count by sex
sex_counts = demographics['sex'].value_counts().reset_index()
sex_counts.columns = ['subgroup', 'n']
sex_counts['attribute_type'] = 'sex'
sex_counts['dataset'] = 'MIMIC-IV'
sex_counts['proportion'] = sex_counts['n'] / sex_counts['n'].sum()

# Count by age_group
age_counts = demographics['age_group'].value_counts().reset_index()
age_counts.columns = ['subgroup', 'n']
age_counts['attribute_type'] = 'age'
age_counts['dataset'] = 'MIMIC-IV'
age_counts['proportion'] = age_counts['n'] / age_counts['n'].sum()

# Count by race
race_counts = demographics['race'].value_counts().reset_index()
race_counts.columns = ['subgroup', 'n']
race_counts['attribute_type'] = 'race'
race_counts['dataset'] = 'MIMIC-IV'
race_counts['proportion'] = race_counts['n'] / race_counts['n'].sum()

# Combine and reorder columns
demo_df = pd.concat([sex_counts, age_counts, race_counts], ignore_index=True)
demo_df = demo_df[['dataset', 'attribute_type', 'subgroup', 'n', 'proportion']]

# Sort by attribute and standardized order
age_order = ['<40', '40-65', '65+']
demo_df['sort_key'] = demo_df.apply(
    lambda x: (
        {'sex': 0, 'age': 1, 'race': 2}.get(x['attribute_type'], 3),
        age_order.index(x['subgroup']) if x['subgroup'] in age_order else 999,
        str(x['subgroup'])
    ),
    axis=1
)
demo_df = demo_df.sort_values('sort_key').drop('sort_key', axis=1).reset_index(drop=True)

demo_df.to_csv(DATA_DIR / "demographics_counts.csv", index=False)
print(f"  -> {len(demo_df)} rows in demographics_counts.csv")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("AGGREGATION COMPLETE")
print("="*80)
print(f"main_results.csv:        {len(main_df)} rows")
print(f"subgroup_results.csv:    {len(subgroup_df)} rows")
print(f"fairness_summary.csv:    {len(fairness_df)} rows")
print(f"intervention_results.csv: {len(intervention_df)} rows")
print(f"demographics_counts.csv: {len(demo_df)} rows")
print(f"\nAll CSVs saved to: {DATA_DIR}")
print("="*80)
