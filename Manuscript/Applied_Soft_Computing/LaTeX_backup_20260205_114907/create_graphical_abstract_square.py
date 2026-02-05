#!/usr/bin/env python3
"""
Create Square Graphical Abstract for Applied Soft Computing Submission
Requirements: 5×5 cm at 300 DPI = 590×590 pixels
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
import numpy as np

# Configuration
OUTPUT_FILE = 'figures/graphical_abstract_square.png'
DPI = 300
SIZE_CM = 5.0  # 5×5 cm
SIZE_INCHES = SIZE_CM / 2.54  # Convert cm to inches
SIZE_PIXELS = int(SIZE_INCHES * DPI)  # 590 pixels

def create_square_graphical_abstract():
    """Create compact square graphical abstract"""

    # Create square figure
    fig, ax = plt.subplots(figsize=(SIZE_INCHES, SIZE_INCHES), dpi=DPI)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title (top)
    ax.text(5, 9.5, 'DRL for UAM Queueing Systems',
            ha='center', va='top', fontsize=9, fontweight='bold')

    # System Architecture (upper section)
    # Draw 5-layer pyramid structure
    layer_heights = [8.5, 7.8, 7.1, 6.4, 5.7]
    layer_widths = [1.5, 2.0, 2.5, 3.0, 3.5]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

    for i, (h, w, c) in enumerate(zip(layer_heights, layer_widths, colors)):
        rect = FancyBboxPatch((5-w/2, h-0.3), w, 0.5,
                              boxstyle="round,pad=0.05",
                              facecolor=c, edgecolor='black', linewidth=0.8)
        ax.add_patch(rect)
        ax.text(5, h, f'L{5-i}', ha='center', va='center',
                fontsize=6, fontweight='bold')

    # DRL Agent box
    agent_box = FancyBboxPatch((0.5, 5.0), 2.5, 1.2,
                               boxstyle="round,pad=0.1",
                               facecolor='#A8E6CF', edgecolor='black', linewidth=1)
    ax.add_patch(agent_box)
    ax.text(1.75, 5.8, 'DRL Agent', ha='center', va='center',
            fontsize=7, fontweight='bold')
    ax.text(1.75, 5.4, 'A2C/PPO/TD3', ha='center', va='center', fontsize=5)

    # Arrow from agent to system
    arrow = FancyArrowPatch((3.0, 5.6), (4.2, 6.5),
                           arrowstyle='->', mutation_scale=15,
                           linewidth=1.5, color='black')
    ax.add_patch(arrow)
    ax.text(3.6, 6.3, 'Actions', ha='center', fontsize=5,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    # Key Results (middle section)
    results_box = FancyBboxPatch((0.3, 2.8), 9.4, 2.0,
                                 boxstyle="round,pad=0.1",
                                 facecolor='#FFE5B4', edgecolor='black', linewidth=1)
    ax.add_patch(results_box)

    ax.text(5, 4.5, 'Key Findings', ha='center', va='top',
            fontsize=8, fontweight='bold')

    findings = [
        '• 59.9% improvement over heuristics',
        '• Inverted pyramid: +9.7%-19.7%',
        '• Capacity paradox: K=10 > K=30+'
    ]

    for i, finding in enumerate(findings):
        ax.text(5, 4.0 - i*0.45, finding, ha='center', va='top', fontsize=5.5)

    # Performance metrics (bottom section)
    metrics_box = FancyBboxPatch((0.3, 0.3), 4.4, 2.2,
                                 boxstyle="round,pad=0.1",
                                 facecolor='#E8F4F8', edgecolor='black', linewidth=1)
    ax.add_patch(metrics_box)

    ax.text(2.5, 2.2, 'Best Algorithm: A2C', ha='center', va='top',
            fontsize=6, fontweight='bold')
    ax.text(2.5, 1.8, 'Reward: 4437.86', ha='center', fontsize=5)
    ax.text(2.5, 1.5, 'Training: 6.9 min', ha='center', fontsize=5)
    ax.text(2.5, 1.2, 'Crash Rate: 0%', ha='center', fontsize=5)
    ax.text(2.5, 0.9, 'Stability: 99.8%', ha='center', fontsize=5)
    ax.text(2.5, 0.6, 'Cohen's d: 6.31', ha='center', fontsize=5)

    # Statistical validation (bottom right)
    stats_box = FancyBboxPatch((5.0, 0.3), 4.7, 2.2,
                               boxstyle="round,pad=0.1",
                               facecolor='#F0E6FF', edgecolor='black', linewidth=1)
    ax.add_patch(stats_box)

    ax.text(7.35, 2.2, 'Statistical Validation', ha='center', va='top',
            fontsize=6, fontweight='bold')
    ax.text(7.35, 1.8, 'n = 30 per group', ha='center', fontsize=5)
    ax.text(7.35, 1.5, 'p < 10⁻⁶⁸', ha='center', fontsize=5)
    ax.text(7.35, 1.2, '500K timesteps', ha='center', fontsize=5)
    ax.text(7.35, 0.9, '15 algorithms tested', ha='center', fontsize=5)
    ax.text(7.35, 0.6, '5 random seeds', ha='center', fontsize=5)

    # Footer
    ax.text(5, 0.1, 'Applied Soft Computing', ha='center', va='bottom',
            fontsize=5, style='italic', color='gray')

    plt.tight_layout(pad=0.1)
    return fig

if __name__ == '__main__':
    fig = create_square_graphical_abstract()
    plt.savefig(OUTPUT_FILE, dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.05)
    print(f"Square graphical abstract saved to {OUTPUT_FILE}")
    print(f"Target size: {SIZE_PIXELS} × {SIZE_PIXELS} pixels (5×5 cm at 300 DPI)")
    print(f"Actual size: Check with 'file {OUTPUT_FILE}'")
    plt.close()
