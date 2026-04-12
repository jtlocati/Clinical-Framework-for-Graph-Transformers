"""
Figure 2: Temporal graph construction schematic
Shows how a patient time series becomes a graph, with three variants:
- temporal-only (chain adjacency)
- clinical-only (clinical measurement patterns)
- full graph (combined)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch, Rectangle
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

def draw_nodes(ax, x_offset, y_base, n_nodes=8, labeled=False, title=""):
    """Draw n circular nodes in a horizontal sequence."""
    x_positions = []
    for i in range(n_nodes):
        x = x_offset + i * 0.6
        x_positions.append(x)
        circle = Circle((x, y_base), 0.2, facecolor=COLORS['blue_light'],
                       edgecolor=COLORS['blue_dark'], linewidth=1.5, zorder=10)
        ax.add_patch(circle)
        if labeled:
            ax.text(x, y_base, f't{i+1}', ha='center', va='center',
                   fontsize=6, fontweight='bold')
    
    if title:
        ax.text(x_offset + (n_nodes-1)*0.3, y_base + 0.55, title,
               ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    return x_positions

def draw_temporal_edges(ax, x_positions, y_base):
    """Draw temporal chain edges (bidirectional)."""
    for i in range(len(x_positions)-1):
        # Forward edge
        ax.plot([x_positions[i], x_positions[i+1]], 
               [y_base-0.25, y_base-0.25],
               color=COLORS['blue_med'], linewidth=1.5, zorder=5)
        # Backward edge
        ax.plot([x_positions[i], x_positions[i+1]], 
               [y_base-0.35, y_base-0.35],
               color=COLORS['blue_med'], linewidth=1.5, zorder=5)

def draw_clinical_edges(ax, x_positions, y_base, skip_pattern=[2, 3]):
    """Draw clinical edges based on measurement skip patterns."""
    for skip in skip_pattern:
        for i in range(len(x_positions)-skip):
            # Jump edges
            ax.plot([x_positions[i], x_positions[i+skip]], 
                   [y_base+0.25, y_base+0.25],
                   color=COLORS['warm_med'], linewidth=1.5,
                   linestyle='--', zorder=5, alpha=0.8)
            ax.plot([x_positions[i], x_positions[i+skip]], 
                   [y_base+0.35, y_base+0.35],
                   color=COLORS['warm_med'], linewidth=1.5,
                   linestyle='--', zorder=5, alpha=0.8)

def draw_self_loops(ax, x_positions, y_base):
    """Draw self-loop arcs on each node."""
    for x in x_positions:
        # Draw self-loop on top of node
        arc = mpatches.Arc((x, y_base), 0.4, 0.4, angle=0, theta1=45, theta2=135,
                          color=COLORS['blue_dark'], linewidth=1.0, linestyle=':',
                          zorder=8)
        ax.add_patch(arc)

def figure2_graph_construction():
    """Create Figure 2: Temporal graph construction schematic."""
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    fig.suptitle('Figure 2: Temporal Graph Construction Across Variants',
                fontsize=13, fontweight='bold', y=0.98)
    
    n_nodes = 8
    y_base = 1.5
    
    # ======================================================================
    # PANEL A: TEMPORAL-ONLY
    # ======================================================================
    ax = axes[0]
    ax.set_xlim(-1, 6)
    ax.set_ylim(0, 2.5)
    ax.axis('off')
    ax.text(-0.5, 2.3, 'A', fontsize=11, fontweight='bold')
    
    # Title and description
    ax.text(2.0, 2.15, 'Temporal-Only Graph', ha='center', fontsize=9, fontweight='bold')
    ax.text(2.0, 1.95, 'Adjacent time steps connected', ha='center', fontsize=7, style='italic')
    
    # Draw nodes and temporal edges
    x_pos_a = draw_nodes(ax, 0.4, y_base, n_nodes, labeled=True)
    draw_temporal_edges(ax, x_pos_a, y_base)
    draw_self_loops(ax, x_pos_a, y_base)
    
    # Legend
    ax.plot([0.5, 1.0], [0.5, 0.5], color=COLORS['blue_med'], linewidth=1.5)
    ax.text(1.1, 0.5, 'Temporal edge', va='center', fontsize=7)
    ax.plot([0.5, 1.0], [0.2, 0.2], color=COLORS['blue_dark'], linewidth=1.0, linestyle=':')
    ax.text(1.1, 0.2, 'Self-loop', va='center', fontsize=7)
    
    # ======================================================================
    # PANEL B: CLINICAL-ONLY
    # ======================================================================
    ax = axes[1]
    ax.set_xlim(-1, 6)
    ax.set_ylim(0, 2.5)
    ax.axis('off')
    ax.text(-0.5, 2.3, 'B', fontsize=11, fontweight='bold')
    
    ax.text(2.0, 2.15, 'Clinical-Only Graph', ha='center', fontsize=9, fontweight='bold')
    ax.text(2.0, 1.95, 'Jump edges by clinical measurement patterns', ha='center', fontsize=7, style='italic')
    
    x_pos_b = draw_nodes(ax, 0.4, y_base, n_nodes, labeled=True)
    draw_clinical_edges(ax, x_pos_b, y_base, skip_pattern=[2, 3])
    draw_self_loops(ax, x_pos_b, y_base)
    
    # Legend
    ax.plot([0.5, 1.0], [0.5, 0.5], color=COLORS['warm_med'], linewidth=1.5, linestyle='--')
    ax.text(1.1, 0.5, 'Clinical edge', va='center', fontsize=7)
    ax.plot([0.5, 1.0], [0.2, 0.2], color=COLORS['blue_dark'], linewidth=1.0, linestyle=':')
    ax.text(1.1, 0.2, 'Self-loop', va='center', fontsize=7)
    
    # ======================================================================
    # PANEL C: FULL GRAPH
    # ======================================================================
    ax = axes[2]
    ax.set_xlim(-1, 6)
    ax.set_ylim(0, 2.5)
    ax.axis('off')
    ax.text(-0.5, 2.3, 'C', fontsize=11, fontweight='bold')
    
    ax.text(2.0, 2.15, 'Full Graph', ha='center', fontsize=9, fontweight='bold')
    ax.text(2.0, 1.95, 'Temporal + Clinical edges combined', ha='center', fontsize=7, style='italic')
    
    x_pos_c = draw_nodes(ax, 0.4, y_base, n_nodes, labeled=True)
    draw_temporal_edges(ax, x_pos_c, y_base)
    draw_clinical_edges(ax, x_pos_c, y_base, skip_pattern=[2, 3])
    draw_self_loops(ax, x_pos_c, y_base)
    
    # Legend
    ax.plot([0.5, 1.0], [0.55, 0.55], color=COLORS['blue_med'], linewidth=1.5)
    ax.text(1.1, 0.55, 'Temporal edge', va='center', fontsize=7)
    ax.plot([0.5, 1.0], [0.35, 0.35], color=COLORS['warm_med'], linewidth=1.5, linestyle='--')
    ax.text(1.1, 0.35, 'Clinical edge', va='center', fontsize=7)
    ax.plot([0.5, 1.0], [0.15, 0.15], color=COLORS['blue_dark'], linewidth=1.0, linestyle=':')
    ax.text(1.1, 0.15, 'Self-loop', va='center', fontsize=7)
    
    # ======================================================================
    # GLOBAL CAPTION / NOTES
    # ======================================================================
    fig.text(0.1, 0.02, 
            'Note: Each circle represents one time step. Node features contain vitals, labs, medications, and diagnoses + missingness mask. '
            'Edges encode predefined clinical relationships and temporal continuity; they do not assert causality.',
            fontsize=7, style='italic', wrap=True)
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    return fig


if __name__ == '__main__':
    fig = figure2_graph_construction()
    
    # Export
    output_dir = 'figures/main'
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    fig.savefig(f'{output_dir}/Figure2_graph_construction.pdf', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    fig.savefig(f'{output_dir}/Figure2_graph_construction.svg', format='svg', bbox_inches='tight',
               facecolor='white', edgecolor='none')
    fig.savefig(f'{output_dir}/Figure2_graph_construction.png', dpi=600, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    
    print("✓ Figure 2 saved to figures/main/")
    print("  - Figure2_graph_construction.pdf")
    print("  - Figure2_graph_construction.svg")
    print("  - Figure2_graph_construction.png (600 dpi)")
    
    plt.close()
