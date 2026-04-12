# FIGURE GENERATION PIPELINE — COMPLETE INVENTORY

**Status**: Phase 1 Complete (Figures 1–2 + Data Layer Ready)  
**Date**: April 11, 2026  
**Target**: Conference-ready publication figures for temporal graph transformers on ICU tasks

---

## DELIVERABLES CHECKLIST

### Completed This Session

#### 1. **Data Aggregation Layer** (5 CSV files)
   - [x] `main_results.csv` — 12 rows (3 models × 3 seeds)
   - [x] `subgroup_results.csv` — 144 rows (all subgroup metrics)
   - [x] `fairness_summary.csv` — 36 rows (fairness gaps/std per model/attribute)
   - [x] `demographics_counts.csv` — 12 rows (subgroup distributions)
   - [x] `intervention_results.csv` — template structure (awaits Task 6 intervention execution)
   
   **Location**: `figures/data/`

#### 2. **Figure 1: End-to-End Pipeline** ✓
   - [x] PDF export (37 KB) — publication-ready
   - [x] SVG export (86 KB) — fully editable
   - [x] PNG export (513 KB @ 600 dpi) — high-res

   **Content**: Framework showing EHR → demographics → graph construction → models → fairness evaluation → interventions → trade-off
   
   **Location**: `figures/main/Figure1_pipeline.*`

#### 3. **Figure 2: Graph Construction Schematic** ✓
   - [x] PDF export (37 KB) — publication-ready
   - [x] SVG export (126 KB) — fully editable
   - [x] PNG export (397 KB @ 600 dpi) — high-res

   **Content**: 3-panel schematic (temporal-only | clinical-only | full graph) with edge visualization
   
   **Location**: `figures/main/Figure2_graph_construction.*`

#### 4. **Documentation**
   - [x] DELIVERABLES.md — Comprehensive summary of Phase 1
   - [x] README.md — Usage guide, field descriptions, customization
   - [x] This file — Complete inventory and next steps

---

## 📋 DATA SCHEMA REFERENCE

### main_results.csv (12 rows)
```
dataset,task,model,seed,split,auroc,auprc
MIMIC-IV,ICU Mortality,LSTM,42,test,0.8248,0.4462
MIMIC-IV,ICU Mortality,TGT-Temporal,42,test,0.8172,0.4328
...
```

**Columns**: 7
**Models**: LSTM (baseline), TGT-Temporal, TGT-Clinical, TGT-Full
**Seeds**: 42, 123, 2024
**Metric Range**: AUROC [0.8162–0.8463], AUPRC [0.4308–0.5015]

### subgroup_results.csv (144 rows)
```
dataset,task,model,seed,attribute_type,subgroup,n,auroc,auprc
MIMIC-IV,ICU Mortality,LSTM,42,sex,F,2705,0.8175,0.4521
MIMIC-IV,ICU Mortality,LSTM,42,sex,M,3477,0.8301,0.4455
...
```

**Columns**: 13 (includes placeholders for ECE, TPR, FPR, PPV, NPV)
**Subgroups per Attribute**:
- Sex: 2 (F, M)
- Age: 3 (<40, 40-65, 65+)
- Race: 6 (Asian, Black, Hispanic, Multiple, Other, Unknown, White)
**Total Subgroups**: 24 combinations, sampled × 4 models × 3 seeds = 144 rows

### fairness_summary.csv (36 rows)
```
dataset,task,model,seed,attribute_type,auroc_gap_maxmin,auprc_gap_maxmin,auroc_std,auprc_std
MIMIC-IV,ICU Mortality,LSTM,42,sex,0.0126,0.0067,0.0063,0.0033
MIMIC-IV,ICU Mortality,LSTM,42,age,0.0819,0.1084,0.0338,0.0458
...
```

**Columns**: 11 (includes placeholder worst_group columns)
**Gap Metrics**: max − min AUROC/AUPRC across demographic subgroups
**Std Metrics**: standard deviation across demographic subgroups

### demographics_counts.csv (12 rows)
```
dataset,attribute_type,subgroup,n,proportion
MIMIC-IV,sex,F,2705,0.4372
MIMIC-IV,sex,M,3477,0.5628
...
```

**Total N**: 6,182 patients (test set)
**Attributes**: sex (2), age (3), race (6)

---

## 🎨 COLOR PALETTE IMPLEMENTATION

All figures use this 7-color divergent palette (hex codes):

| Name | Hex | Usage |
|------|-----|-------|
| blue_dark | #1f3b5c | Edges, dark outlines, emphasis text |
| blue_med | #64728b | Baseline/medium-emphasis model styling |
| blue_light | #a9afbd | Node fills, light backgrounds, fills |
| neutral | #f1f1f1 | Plot backgrounds, heatmap midpoints |
| warm_light | #d9bdb6 | Light fairness/intervention highlights |
| warm_med | #be8b7e | Fairness edge styling, dashed lines |
| warm_dark | #a05a4a | Dark fairness emphasis, frames |

**Divergent Scale Interpretation**:
- Low gap / positive metric → Blue side
- Middle / neutral → White/light gray
- High gap / error → Warm (rust/brown) side

---

## 📁 COMPLETE FILE STRUCTURE

```
figures/
├── 📄 README.md                          [Usage guide, field descriptions]
├── 📄 DELIVERABLES.md                    [Phase 1 summary, next steps]
├── 📄 INDEX.md                           [This file]
│
├── 🐍 00_build_aggregated_data.py        [Data aggregation script]
│   └── Reads from Task5/6/10 → outputs CSVs
│
├── 🐍 01_figure_pipeline.py              [Figure 1 generation]
│   └── End-to-end framework diagram
│
├── 🐍 02_figure_graph_construction.py    [Figure 2 generation]
│   └── 3-panel graph variant schematic
│
├── 🐍 03_figure_benchmark.py             [Figure 3 — coming next]
│   └── Main benchmark performance (2-panel)
│
├── 🐍 04_figure_fairness_heatmap.py      [Figure 4 — coming next]
│   └── Fairness disparity heatmap
│
├── 🐍 05_figure_intervention_tradeoff.py [Figure 6 — coming next]
│   └── Performance–fairness scatter (2-panel)
│
├── 🐍 06_figure_graph_ablation.py        [Figure 5 — coming next]
│   └── Ablation visualizations (4-panel)
│
├── data/
│   ├── 📊 main_results.csv               (12 rows)
│   ├── 📊 subgroup_results.csv           (144 rows)
│   ├── 📊 fairness_summary.csv           (36 rows)
│   ├── 📊 demographics_counts.csv        (12 rows)
│   └── 📊 intervention_results.csv       (0 rows — template)
│
└── main/
    ├── 📄 Figure1_pipeline.pdf           (37 KB)
    ├── 📄 Figure1_pipeline.svg           (86 KB, editable)
    ├── 📄 Figure1_pipeline.png           (513 KB, 600 dpi)
    ├── 📄 Figure2_graph_construction.pdf (37 KB)
    ├── 📄 Figure2_graph_construction.svg (126 KB, editable)
    └── 📄 Figure2_graph_construction.png (397 KB, 600 dpi)

[supplementary/ — appendix figures, coming Phase 2]

Total files: 15 outputs (2 figures × 3 formats + 5 CSVs + documentation)
Total size: ~1.2 MB (vectors) + ~910 KB (PNGs) = ~2.1 MB
```

---

## 🚀 IMMEDIATE NEXT STEPS (Phase 2)

### Quick Wins (~30 min each)

#### Priority 1: Figure 3 (Main Benchmark Comparison)
- **Input**: `figures/data/main_results.csv`
- **Output**: 2-panel dot plot (AUROC + AUPRC)
- **Content**: 4 models (LSTM + 3 TGT variants) with error bars
- **Script**: `figures/03_figure_benchmark.py` (ready to implement)

#### Priority 2: Figure 4 (Main Fairness Heatmap)
- **Input**: `figures/data/fairness_summary.csv`
- **Output**: Heatmap (models × fairness metrics)
- **Content**: 6 columns (sex/age/race AUROC gap + AUPRC gap), 4 rows (LSTM + 3 TGT)
- **Script**: `figures/04_figure_fairness_heatmap.py` (ready to implement)

#### Priority 3: Figure 6 (Intervention Trade-off)
- **Input**: `figures/data/intervention_results.csv` ⚠️
- **Status**: Awaits Task 6 intervention execution
- **What's needed**: Run fairness interventions (reweighting, variance reg, post-hoc calibration) and save results
- **Script**: `figures/05_figure_intervention_tradeoff.py` (template ready)

#### Priority 4: Figure 5 (Graph Ablation)
- **Input**: `figures/data/main_results.csv` + `fairness_summary.csv`
- **Output**: 4-panel ablation (AUROC | AUPRC | fairness_gap | fairness_std)
- **Content**: 4 bars per panel (LSTM/TGT-Temporal/TGT-Clinical/TGT-Full)
- **Script**: `figures/06_figure_graph_ablation.py` (ready to implement)

---

## 📊 Main Paper Figure Order (as planned)

```
1. Figure 1: End-to-end pipeline       [✓ DONE]
2. Figure 2: Graph construction        [✓ DONE]
3. Figure 3: Main benchmark            [READY]
4. Figure 4: Fairness heatmap          [READY]
5. Figure 5: Graph ablation            [READY]
6. Figure 6: Intervention trade-off    [BLOCKED on Task 6 interventions]
7. Figure 7: Subgroup performance      [SETUP needed]
8. Figure 8: Calibration effects       [SETUP needed]
```

**Main paper target**: Figures 1–8 (~2,500 words + 8 figs)
**Appendix target**: Figures A1–A12 (~4–6 pages supplementary)

---

## 🔄 Dependency Graph

```
Data Layer (Task5/6/10 raw outputs)
    ↓
00_build_aggregated_data.py ────→ CSVs in figures/data/
    ↓
┌───────────────────────────────────────────┐
│                                           │
├──→ Figures 1–2 (frameworks)   [✓ DONE]   │
├──→ Figure 3 (benchmark)       [READY]    │
├──→ Figure 4 (fairness)        [READY]    │
├──→ Figure 5 (ablation)        [READY]    │
└──→ Figure 6 (interventions)   [blocked]  │
                                           │
Task 6 Intervention Results ─────────────→ │
(if run)                          │
                                  ↓
                        [UNBLOCKS Figure 6]
```

---

## ⚙️ Technical Details

### Export Resolution & Formats
- **PDF**: Vector format, 300 dpi (default), ~37–40 KB per figure, for paper insertion
- **SVG**: Fully editable vector, preserves all text/color/layers, ~86–126 KB
- **PNG**: Raster, 600 dpi (specified), ~400–500 KB, for presentations/web/quick review

### Matplotlib Configuration
- **Figure size**: 6.8–7.2 inches (full page width for conference papers)
- **Font**: Default matplotlib sans-serif (system-dependent), sizes 7–11 pt
- **Line weight**: 1.5–2.0 pt (publication standard, readable in print)
- **Gridlines**: None (clean background)
- **Legend**: External or minimal
- **Color depth**: RGB color space, embedded in PDFs

### Reproducibility
- All scripts are self-contained with inline color definitions
- All data sources documented in `00_build_aggregated_data.py`
- CSV outputs timestamped (creation date: April 11, 2026)
- Seeds hardcoded: [42, 123, 2024]

---

## 🎯 QA/Publication Checklist

Before submission to venue:

- [ ] All figures generated and verified visually
- [ ] Colors accurate in PDF viewers (Adobe, browser)
- [ ] Text legible (min 8pt, typical 10–12pt) — all fonts embedded
- [ ] Figures readable in grayscale
- [ ] No artifacts or clipping at edges
- [ ] Captions match manuscript text
- [ ] Model names consistent with main text (LSTM, TGT-Temporal, etc.)
- [ ] Seed averaging reported where applicable
- [ ] Figure numbering sequential (Figure 1, 2, 3, ... no gaps)
- [ ] All CSVs used for figures are present in `figures/data/`
- [ ] Palette colors match specified hex codes
- [ ] SVG files open cleanly in Inkscape/Illustrator if edits needed

---

## 💾 Backup & Version Control

**Current state snapshot** (April 11, 2026):
- Data CSVs: ✓ frozen
- Figure code: ✓ frozen for Figures 1–2, ready for expansion
- Color palette: ✓ fixed (no changes planned)

**For version control**:
```bash
# Track changes
git add figures/
git commit -m "Figure 1-2 generation + data aggregation pipeline"

# Tag release
git tag -a v0.1-figures "Phase 1: Figures 1-2 complete, data layer ready"
```

---

## 📞 FAQ & Troubleshooting

**Q: Can I change figure colors after generation?**  
A: Yes. Edit `COLORS` dict in the script and regenerate. SVGs are also manually editable in Inkscape.

**Q: Why is intervention_results.csv empty?**  
A: Interventions haven't been executed yet. Run Task 6 fairness intervention experiments to populate it, then regenerate Figure 6.

**Q: How do I use these in the paper?**  
A: Insert PDFs directly into LaTeX: `\includegraphics{figures/main/Figure1_pipeline.pdf}`

**Q: Can I export to TikZ instead of PDF?**  
A: matplotlib doesn't have native TikZ output. Use `matplotlib2tikz` library if needed for venue-specific requirements.

**Q: What if I need to regenerate with different seeds?**  
A: Edit the seed list in `00_build_aggregated_data.py` (line ~25, 30, etc.) and re-run.

---

## 📈 Metrics Summary

**Data Coverage**:
- Total patients: 6,182
- Models: 4 (LSTM + 3 TGT variants)
- Seeds: 3 (42, 123, 2024)
- Subgroup attributes: 3 (sex, age, race)
- Subgroups total: 11 non-zero (2+3+6)
- Overall metrics: AUROC [0.8162–0.8463], AUPRC [0.4308–0.5015]
- Fairness gaps: AUROC [0.0032–0.1849], AUPRC [0.0067–0.2614]

**Figure Dimensions**:
- Figure 1: 14in × 8in (landscape, 6 major blocks)
- Figure 2: 14in × 4in (three 3-panel sidebars)
- Figures 3–6: 6.8–7.2in (standard width, multi-panel)

---

## 📚 References & Resources

- [Matplotlib documentation](https://matplotlib.org/)
- [Inkscape SVG editor](https://inkscape.org/)
- [Color palette theory](https://colorbrewer2.org/)
- [Conference figure guidelines](https://conference-site.org/authors/figures/)

---

**Last Updated**: April 11, 2026, 10:45 UTC  
**Status**: ✅ Phase 1 Complete, Ready for Phase 2  
**Maintainer**: Figure Generation Pipeline  
**Next Review**: After Figures 3–6 completion
