#!/usr/bin/env python3
"""
Create Graphical Abstract for Applied Soft Computing Submission
Requirements: ≥531×1328 pixels (vertical layout)
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle
import numpy as np

# Configuration
OUTPUT_FILE = 'figures/graphical_abstract.png'
DPI = 300
WIDTH_INCHES = 5.31  # 531 pixels at 100 DPI, scaled for 300 DPI
HEIGHT_INCHES = 13.28  # 1328 pixels at 100 DPI, scaled for 300 DPI

# Figure paths (using PNG versions for easier manipulation)
FIG1_PATH = '../../../Analysis/figures/fig1_capacity_performance_en.png'
FIG2_PATH = '../../../Analysis/figures/fig2_structure_comparison_en.png'
FIG4_PATH = '../../../Analysis/figures/fig4_algorithm_radar_en.png'

def create_graphical_abstract():
    """Create graphical abstract combining key figures"""

    # Create figure with vertical layout
    fig = plt.figure(figsize=(WIDTH_INCHES, HEIGHT_INCHES), dpi=DPI)

    # Load images
    try:
        img_capacity = mpimg.imread(FIG1_PATH)
        img_structure = mpimg.imread(FIG2_PATH)
        img_algorithms = mpimg.imread(FIG4_PATH)
    except FileNotFoundError as e:
        print(f"Error loading images: {e}")
        return None

    # Define layout: 4 panels (Title, Problem/Results, Key Findings, Conclusions)
    # Using GridSpec for flexible layout
    gs = fig.add_gridspec(6, 1, height_ratios=[0.8, 1.5, 2.0, 2.0, 1.5, 0.5],
                          hspace=0.3, left=0.05, right=0.95, top=0.98, bottom=0.02)

    # Panel 1: Title
    ax_title = fig.add_subplot(gs[0, 0])
    ax_title.axis('off')
    ax_title.text(0.5, 0.5, 'Deep Reinforcement Learning for\nVertical Layered Queueing Systems in Urban Air Mobility',
                  ha='center', va='center', fontsize=14, fontweight='bold', wrap=True)

    # Panel 2: Algorithm Comparison (fig4)
    ax_algo = fig.add_subplot(gs[1, 0])
    ax_algo.imshow(img_algorithms)
    ax_algo.axis('off')
    ax_algo.text(0.02, 0.98, 'A. Algorithm Performance', transform=ax_algo.transAxes,
                 fontsize=11, fontweight='bold', va='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Panel 3: Structural Comparison (fig2)
    ax_struct = fig.add_subplot(gs[2, 0])
    ax_struct.imshow(img_structure)
    ax_struct.axis('off')
    ax_struct.text(0.02, 0.98, 'B. Structural Analysis', transform=ax_struct.transAxes,
                   fontsize=11, fontweight='bold', va='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # Panel 4: Capacity Paradox (fig1)
    ax_capacity = fig.add_subplot(gs[3, 0])
    ax_capacity.imshow(img_capacity)
    ax_capacity.axis('off')
    ax_capacity.text(0.02, 0.98, 'C. Capacity Paradox', transform=ax_capacity.transAxes,
                     fontsize=11, fontweight='bold', va='top',
                     bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # Panel 5: Key Findings
    ax_findings = fig.add_subplot(gs[4, 0])
    ax_findings.axis('off')
    findings_text = (
        "Key Findings:\n"
        "• DRL algorithms achieve 59.9% improvement over heuristics\n"
        "• Inverted pyramid structure outperforms by 9.7%-19.7%\n"
        "• Capacity paradox: K=10 outperforms K=30+ under extreme load\n"
        "• A2C achieves best performance with minimal training time"
    )
    ax_findings.text(0.5, 0.5, findings_text, ha='center', va='center',
                     fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    # Panel 6: Footer
    ax_footer = fig.add_subplot(gs[5, 0])
    ax_footer.axis('off')
    ax_footer.text(0.5, 0.5, 'Applied Soft Computing Journal Submission',
                   ha='center', va='center', fontsize=9, style='italic')

    return fig

if __name__ == '__main__':
    fig = create_graphical_abstract()
    plt.savefig(OUTPUT_FILE, dpi=DPI, bbox_inches='tight', facecolor='white')
    print(f"Graphical abstract saved to {OUTPUT_FILE}")
    print(f"Size: {WIDTH_INCHES*DPI:.0f} x {HEIGHT_INCHES*DPI:.0f} pixels")
