"""
Figure 1: End-to-end study overview / pipeline
Shows the complete Data → Model → Evaluation → Intervention flow.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Colors from the specified palette
COLORS = {
    'blue_dark': '#1f3b5c',
    'blue_med': '#64728b',
    'blue_light': '#a9afbd',
    'neutral': '#f1f1f1',
    'warm_light': '#d9bdb6',
    'warm_med': '#be8b7e',
    'warm_dark': '#a05a4a'
}

def figure1_pipeline():
    """Create Figure 1: End-to-end pipeline overview."""
    
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    fig.suptitle('Figure 1: End-to-End Framework for Temporal Graph Transformers on ICU Tasks',
                 fontsize=13, fontweight='bold', y=0.98)
    
    # Y-positions for three main flows
    y_top = 6.5
    y_mid = 4.0
    y_bot = 1.5
    
    # ======================================================================
    # STAGE 1: DATA PREPARATION (left, blue colors)
    # ======================================================================
    x_pos = 0.5
    
    # Input block
    rect1 = FancyBboxPatch((x_pos, y_top-0.6), 1.8, 0.8,
                           boxstyle="round,pad=0.05", 
                           edgecolor=COLORS['blue_dark'], facecolor=COLORS['blue_light'],
                           linewidth=1.5)
    ax.add_patch(rect1)
    ax.text(x_pos+0.9, y_top-0.2, 'EHR\nTime Series', ha='center', va='center',
           fontsize=8, fontweight='bold')
    
    # Arrow 1
    arrow1 = FancyArrowPatch((x_pos+1.8, y_top-0.2), (x_pos+2.5, y_top-0.2),
                            arrowstyle='->', mutation_scale=20, linewidth=1.5,
                            color=COLORS['blue_med'])
    ax.add_patch(arrow1)
    
    # Demographics block
    x_pos += 2.5
    rect2 = FancyBboxPatch((x_pos, y_top-0.6), 1.8, 0.8,
                           boxstyle="round,pad=0.05",
                           edgecolor=COLORS['blue_dark'], facecolor=COLORS['blue_light'],
                           linewidth=1.5)
    ax.add_patch(rect2)
    ax.text(x_pos+0.9, y_top-0.2, 'Demo-\ngraphics', ha='center', va='center',
           fontsize=8, fontweight='bold')
    
    # Arrow 2
    arrow2 = FancyArrowPatch((x_pos+1.8, y_top-0.2), (x_pos+2.5, y_top-0.2),
                            arrowstyle='->', mutation_scale=20, linewidth=1.5,
                            color=COLORS['blue_med'])
    ax.add_patch(arrow2)
    
    # ======================================================================
    # STAGE 2: GRAPH CONSTRUCTION (center, blue colors)
    # ======================================================================
    x_pos += 2.5
    
    # Graph construction box - larger to show variants
    rect3_main = FancyBboxPatch((x_pos-0.2, y_top-1.2), 2.5, 1.4,
                               boxstyle="round,pad=0.1",
                               edgecolor=COLORS['blue_med'], facecolor=COLORS['neutral'],
                               linewidth=2)
    ax.add_patch(rect3_main)
    
    ax.text(x_pos+1.05, y_top+0.05, 'Graph Construction', ha='center', va='bottom',
           fontsize=8, fontweight='bold')
    ax.text(x_pos+0.3, y_top-0.5, '• Temporal edges', ha='left', va='top',
           fontsize=7)
    ax.text(x_pos+0.3, y_top-0.75, '• Clinical edges', ha='left', va='top',
           fontsize=7)
    ax.text(x_pos+0.3, y_top-1.0, '• Adjacency masks', ha='left', va='top',
           fontsize=7)
    
    # Arrow 3
    arrow3 = FancyArrowPatch((x_pos+2.3, y_top-0.2), (x_pos+3.0, y_top-0.2),
                            arrowstyle='->', mutation_scale=20, linewidth=1.5,
                            color=COLORS['blue_med'])
    ax.add_patch(arrow3)
    
    # ======================================================================
    # STAGE 3: MODEL ARCHITECTURES (center-right, blue colors)
    # ======================================================================
    x_pos += 3.0
    
    rect4_main = FancyBboxPatch((x_pos-0.2, y_top-1.2), 2.5, 1.4,
                               boxstyle="round,pad=0.1",
                               edgecolor=COLORS['blue_med'], facecolor=COLORS['neutral'],
                               linewidth=2)
    ax.add_patch(rect4_main)
    
    ax.text(x_pos+1.05, y_top+0.05, 'Model Variants', ha='center', va='bottom',
           fontsize=8, fontweight='bold')
    ax.text(x_pos+0.3, y_top-0.5, '• Baselines (LSTM, Trans.)', ha='left', va='top',
           fontsize=7)
    ax.text(x_pos+0.3, y_top-0.75, '• TGT-Temporal', ha='left', va='top',
           fontsize=7)
    ax.text(x_pos+0.3, y_top-1.0, '• TGT-Clinical/Full', ha='left', va='top',
           fontsize=7)
    
    # Arrow 4 (downward to evaluation)
    arrow4 = FancyArrowPatch((x_pos+1.15, y_top-1.2), (x_pos+1.15, y_mid+0.45),
                            arrowstyle='->', mutation_scale=20, linewidth=1.5,
                            color=COLORS['warm_dark'])
    ax.add_patch(arrow4)
    
    # ======================================================================
    # STAGE 4: PERFORMANCE & FAIRNESS EVALUATION (middle, warm colors)
    # ======================================================================
    x_eval = x_pos
    
    rect5_main = FancyBboxPatch((x_eval-0.2, y_mid-1.0), 2.5, 1.3,
                               boxstyle="round,pad=0.1",
                               edgecolor=COLORS['warm_dark'], facecolor=COLORS['neutral'],
                               linewidth=2)
    ax.add_patch(rect5_main)
    
    ax.text(x_eval+1.05, y_mid+0.25, 'Evaluation', ha='center', va='bottom',
           fontsize=8, fontweight='bold', color=COLORS['warm_dark'])
    ax.text(x_eval+0.3, y_mid-0.15, '• AUROC, AUPRC', ha='left', va='top',
           fontsize=7)
    ax.text(x_eval+0.3, y_mid-0.4, '• Subgroup metrics', ha='left', va='top',
           fontsize=7)
    ax.text(x_eval+0.3, y_mid-0.65, '• Fairness gaps', ha='left', va='top',
           fontsize=7)
    
    # Arrow 5 (downward to interventions)
    arrow5 = FancyArrowPatch((x_eval+1.15, y_mid-1.0), (x_eval+1.15, y_bot+0.45),
                            arrowstyle='->', mutation_scale=20, linewidth=1.5,
                            color=COLORS['warm_dark'])
    ax.add_patch(arrow5)
    
    # ======================================================================
    # STAGE 5: FAIRNESS INTERVENTIONS (bottom-left, warm colors)
    # ======================================================================
    x_int = 1.0
    y_int = y_bot
    
    rect6_main = FancyBboxPatch((x_int-0.2, y_int-0.95), 3.2, 1.2,
                               boxstyle="round,pad=0.1",
                               edgecolor=COLORS['warm_dark'], facecolor=COLORS['neutral'],
                               linewidth=2)
    ax.add_patch(rect6_main)
    
    ax.text(x_int+1.4, y_int+0.2, 'Fairness Interventions', ha='center', va='bottom',
           fontsize=8, fontweight='bold', color=COLORS['warm_dark'])
    ax.text(x_int+0.2, y_int-0.25, '• Balanced sampling', ha='left', va='top',
           fontsize=7)
    ax.text(x_int+0.2, y_int-0.5, '• Reweighting', ha='left', va='top',
           fontsize=7)
    ax.text(x_int+0.2, y_int-0.75, '• Variance regularization', ha='left', va='top',
           fontsize=7)
    ax.text(x_int+1.8, y_int-0.25, '• Post-hoc thresholding', ha='left', va='top',
           fontsize=7)
    ax.text(x_int+1.8, y_int-0.5, '• Calibration', ha='left', va='top',
           fontsize=7)
    
    # Arrow 6 (right to comparison)
    arrow6 = FancyArrowPatch((x_int+3.0, y_int-0.3), (x_int+3.7, y_int-0.3),
                            arrowstyle='->', mutation_scale=20, linewidth=1.5,
                            color=COLORS['warm_dark'])
    ax.add_patch(arrow6)
    
    # ======================================================================
    # STAGE 6: FINAL COMPARISON & OUTPUT (bottom-right, warm colors)
    # ======================================================================
    x_comp = x_int + 3.7
    
    rect7 = FancyBboxPatch((x_comp, y_int-0.7), 2.0, 1.0,
                          boxstyle="round,pad=0.05",
                          edgecolor=COLORS['warm_dark'], facecolor=COLORS['warm_light'],
                          linewidth=1.5)
    ax.add_patch(rect7)
    ax.text(x_comp+1.0, y_int-0.2, 'Performance–\nFairness\nTrade-off', ha='center', va='center',
           fontsize=8, fontweight='bold')
    
    # ======================================================================
    # LEGEND / ANNOTATIONS
    # ======================================================================
    # Blue box = data/model path
    legend_y = 0.3
    rect_legend_blue = FancyBboxPatch((0.5, legend_y-0.15), 0.3, 0.3,
                                     boxstyle="round,pad=0.02",
                                     edgecolor=COLORS['blue_dark'], facecolor=COLORS['blue_light'],
                                     linewidth=1.5)
    ax.add_patch(rect_legend_blue)
    ax.text(1.2, legend_y, 'Data & Model Construction', ha='left', va='center',
           fontsize=7, style='italic')
    
    # Warm box = fairness path
    rect_legend_warm = FancyBboxPatch((5.5, legend_y-0.15), 0.3, 0.3,
                                     boxstyle="round,pad=0.02",
                                     edgecolor=COLORS['warm_dark'], facecolor=COLORS['warm_light'],
                                     linewidth=1.5)
    ax.add_patch(rect_legend_warm)
    ax.text(6.2, legend_y, 'Evaluation & Fairness Analysis', ha='left', va='center',
           fontsize=7, style='italic')
    
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    fig = figure1_pipeline()
    
    # Export
    output_dir = 'figures/main'
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    fig.savefig(f'{output_dir}/Figure1_pipeline.pdf', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    fig.savefig(f'{output_dir}/Figure1_pipeline.svg', format='svg', bbox_inches='tight',
               facecolor='white', edgecolor='none')
    fig.savefig(f'{output_dir}/Figure1_pipeline.png', dpi=600, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    
    print("✓ Figure 1 saved to figures/main/")
    print("  - Figure1_pipeline.pdf")
    print("  - Figure1_pipeline.svg")
    print("  - Figure1_pipeline.png (600 dpi)")
    
    plt.close()
