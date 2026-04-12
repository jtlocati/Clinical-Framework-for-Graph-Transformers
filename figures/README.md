# FIGURE GENERATION USAGE GUIDE

## Quick Start

All figures have been generated and are ready for publication use.

### View the figures:

```bash
# Open Figure 1 and Figure 2
open figures/main/Figure1_pipeline.pdf
open figures/main/Figure2_graph_construction.pdf

# Or in your preferred PDF viewer
```

### Data files for next figures:

All preprocessing is complete. The five aggregated CSV files in `figures/data/` are ready for use:

```python
import pandas as pd

# Load preprocessed data
main_results = pd.read_csv('figures/data/main_results.csv')
subgroup_results = pd.read_csv('figures/data/subgroup_results.csv')
fairness_summary = pd.read_csv('figures/data/fairness_summary.csv')
demographics = pd.read_csv('figures/data/demographics_counts.csv')
interventions = pd.read_csv('figures/data/intervention_results.csv')  # placeholder

print(main_results.head())
print(fairness_summary.head())
```

---

## Figure Directory Structure

```
figures/
├── main/                                    [Publication-ready figures]
│   ├── Figure1_pipeline.pdf                 
│   ├── Figure1_pipeline.svg                 (editable)
│   ├── Figure1_pipeline.png                 (600 dpi)
│   ├── Figure2_graph_construction.pdf
│   ├── Figure2_graph_construction.svg       (editable)
│   └── Figure2_graph_construction.png       (600 dpi)
│
├── data/                                    [Aggregated result tables]
│   ├── main_results.csv                     (12 rows)
│   ├── subgroup_results.csv                 (144 rows)
│   ├── fairness_summary.csv                 (36 rows)
│   ├── intervention_results.csv             (placeholder)
│   └── demographics_counts.csv              (12 rows)
│
├── 00_build_aggregated_data.py              [Scripts]
├── 01_figure_pipeline.py
├── 02_figure_graph_construction.py
├── 03_figure_benchmark.py                   [coming next]
├── 04_figure_fairness_heatmap.py
├── 05_figure_intervention_tradeoff.py
├── 06_figure_graph_ablation.py
│
└── DELIVERABLES.md                          [Summary of work completed]
```

---

## File Format Guidelines

| Format | Use Case | Size | Software |
|--------|----------|------|----------|
| **PDF** | Paper submission, printing | ~37 KB | Adobe, free PDF readers |
| **SVG** | Fine-tuning colors/fonts | ~86–126 KB | Inkscape, Illustrator, browser |
| **PNG** | Presentations, web, quick review | ~400–500 KB @ 600 dpi | Any image viewer |

---

## Color Palette Reference

All figures use this official palette:

```python
COLORS = {
    'blue_dark':   '#1f3b5c',   # Edges, dark outlines, emphasis
    'blue_med':    '#64728b',   # Baseline models, medium emphasis
    'blue_light':  '#a9afbd',   # Light fills, node backgrounds
    'neutral':     '#f1f1f1',   # Plot/panel backgrounds, midpoints
    'warm_light':  '#d9bdb6',   # Light fairness highlights
    'warm_med':    '#be8b7e',   # Fairness edge styling
    'warm_dark':   '#a05a4a'    # Dark fairness emphasis
}
```

**Usage**:
- Blue side (dark → light) = data/model construction
- Warm side (light → dark) = evaluation/fairness/interventions
- Divergent heatmaps: blue=low gap, white=neutral, warm=high gap

---

## CSV Field Descriptions

### main_results.csv
| Column | Type | Values |
|--------|------|--------|
| dataset | string | 'MIMIC-IV' |
| task | string | 'ICU Mortality' |
| model | string | 'LSTM', 'TGT-Temporal', 'TGT-Clinical', 'TGT-Full' |
| seed | int | 42, 123, 2024 |
| split | string | 'test' |
| auroc | float | [0.8, 0.85] |
| auprc | float | [0.45, 0.51] |
| loss | NaN | not populated |

### subgroup_results.csv
| Column | Type | Values |
|--------|------|--------|
| dataset | string | 'MIMIC-IV' |
| task | string | 'ICU Mortality' |
| model | string | 'LSTM', 'TGT-Temporal/Clinical/Full' |
| seed | int | 42, 123, 2024 |
| attribute_type | string | 'sex', 'age', 'race' |
| subgroup | string | 'M'/'F', '<40'/'40-65'/'65+', etc. |
| n | int | cohort size |
| auroc | float | subgroup-specific AUROC or NaN |
| auprc | float | subgroup-specific AUPRC or NaN |
| ece / tpr / fpr / ppv / npv | NaN | placeholders for future |

### fairness_summary.csv
| Column | Type | Values |
|--------|------|--------|
| dataset | string | 'MIMIC-IV' |
| task | string | 'ICU Mortality' |
| model | string | 'LSTM', 'TGT-Temporal/Clinical/Full' |
| seed | int | 42, 123, 2024 |
| attribute_type | string | 'sex', 'age', 'race' |
| auroc_gap_maxmin | float | max − min AUROC across subgroups |
| auprc_gap_maxmin | float | max − min AUPRC across subgroups |
| auroc_std | float | std dev of subgroup AUROCs |
| auprc_std | float | std dev of subgroup AUPRCs |
| worst_group_auroc | NaN | placeholder |
| worst_group_auprc | NaN | placeholder |

### demographics_counts.csv
| Column | Type | Values |
|--------|------|--------|
| dataset | string | 'MIMIC-IV' |
| attribute_type | string | 'sex', 'age', 'race' |
| subgroup | string | 'M'/'F', '<40'/'40-65'/'65+', etc. |
| n | int | count |
| proportion | float | count / total |

---

## Generating Remaining Figures

To generate Figures 3–6, use the corresponding scripts:

```bash
# In the figures/ directory:
source ../.venv/bin/activate

# Figure 3: Main Benchmark Comparison
python 03_figure_benchmark.py

# Figure 4: Main Fairness Heatmap
python 04_figure_fairness_heatmap.py

# Figure 6: Intervention Trade-off (requires intervention data)
python 05_figure_intervention_tradeoff.py

# Figure 5: Graph Ablation
python 06_figure_graph_ablation.py
```

Each script will:
1. Load data from `figures/data/`
2. Generate matplotlib figure
3. Export to PDF + SVG + PNG (600 dpi)
4. Save to `figures/main/`

---

## Customization

### Changing Colors

Edit the `COLORS` dictionary at the top of any figure script:

```python
COLORS = {
    'blue_dark':   '#YOUR_HEX_CODE',
    # ... etc
}
```

Then regenerate the figure.

### Adjusting Figure Size

Change `figsize=(width, height)` in the matplotlib call. Standard widths:
- Single column: 3.5–4.5 inches
- Full page width: 6.8–7.2 inches (used in all figures)
- Two-column spread: 14+ inches

### Exporting to Different DPI

In the export commands, adjust:
```python
fig.savefig(..., dpi=300)  # Default (good for screen)
fig.savefig(..., dpi=600)  # High-res (printing)
fig.savefig(..., dpi=1200) # Ultra-high (publication)
```

---

## Reviewing Figures for Publication

Before submitting:

1. **Open each PDF in Adobe Reader**
   - Check that text is readable (min 8pt, typically 10–12pt)
   - Verify colors are accurate
   - Ensure no cutoff at figure edges

2. **Check SVG in browser**
   - Open in Firefox or Chrome
   - Confirm interactive elements (if any) work
   - Check colors match PDF version

3. **View PNG on screen**
   - 600 dpi should appear crisp
   - Check for artifacts from PNG compression (rare at 600 dpi)

4. **Run grayscale check**
   - Print figures to grayscale (or use image converter)
   - Ensure all information is visible without color
   - All text should be legible

5. **Verify citations and captions**
   - Figure captions match the paper text
   - All abbreviations are defined
   - Model names are consistent with main text

---

## Common Issues & Solutions

### SVG displays poorly in Inkscape
→ Use PDF instead if modifying layout. SVG is best for color-only tweaks.

### PNG is too large
→ Use lossy PNG compression. `pngquant` or ImageMagick reduce file size by 50–70% with minimal quality loss:
```bash
convert Figure1_pipeline.png -colors 256 Figure1_pipeline_compressed.png
```

### PDF fonts look wrong when inserted in LaTeX
→ Ensure all fonts are embedded. Use `pdffonts Figure1_pipeline.pdf` to check.

### Figure appears blurry in paper
→ Regenerate PNG at 1200 dpi for publication PDF insertion.

---

## Next Steps

1. ✓ Figures 1–2 complete and ready
2. Generate Figures 3–6 using the prepared data CSVs
3. Populate `intervention_results.csv` from Task 6 output (if interventions are run)
4. Generate appendix figures (A1–A12) using the same color palette
5. Compile all figure PDFs into the main paper manuscript

---

**Last Updated**: April 11, 2026
**Palette Version**: 7-color divergent (user-specified)
**Export Formats**: PDF (publication), SVG (editable), PNG (web/presentation)
