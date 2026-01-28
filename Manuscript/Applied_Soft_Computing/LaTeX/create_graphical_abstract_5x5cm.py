#!/usr/bin/env python3
"""
Create Precise Square Graphical Abstract for Applied Soft Computing
Requirements: Exactly 5×5 cm at 300 DPI = 590×590 pixels
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Configuration
OUTPUT_FILE = 'figures/graphical_abstract_5x5cm.png'
DPI = 300
SIZE_CM = 5.0
SIZE_INCHES = SIZE_CM / 2.54  # 1.9685 inches
SIZE_PIXELS = 590  # Exact requirement

def create_graphical_abstract():
    """Create precise 5×5 cm graphical abstract"""

    # Create figure with exact dimensions
    fig = plt.figure(figsize=(SIZE_INCHES, SIZE_INCHES), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1])  # Full figure, no margins
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Background
    ax.add_patch(plt.Rectangle((0, 0), 100, 100, facecolor='white', zorder=0))

    # Title section (top 15%)
    title_box = FancyBboxPatch((2, 88), 96, 10,
                               boxstyle="round,pad=0.5",
                               facecolor='#E8F4F8', edgecolor='#2C3E50', linewidth=1.5)
    ax.add_patch(title_box)
    ax.text(50, 93, 'DRL for UAM Layered Queueing', ha='center', va='center',
            fontsize=11, fontweight='bold', color='#2C3E50')

    # System architecture section (middle-top 30%)
    arch_box = FancyBboxPatch((2, 55), 96, 30,
                              boxstyle="round,pad=0.5",
                              facecolor='#FFF9E6', edgecolor='#2C3E50', linewidth=1.2)
    ax.add_patch(arch_box)

    # Draw 5-layer pyramid
    layer_y = [78, 73, 68, 63, 58]
    layer_widths = [15, 20, 25, 30, 35]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

    for i, (y, w, c) in enumerate(zip(layer_y, layer_widths, colors)):
        rect = FancyBboxPatch((50-w/2, y-2), w, 3.5,
                              boxstyle="round,pad=0.3",
                              facecolor=c, edgecolor='black', linewidth=0.8)
        ax.add_patch(rect)
        ax.text(50, y, f'L{5-i}', ha='center', va='center',
                fontsize=7, fontweight='bold')

    # DRL Agent
    agent_box = FancyBboxPatch((8, 65), 20, 10,
                               boxstyle="round,pad=0.5",
                               facecolor='#A8E6CF', edgecolor='black', linewidth=1.2)
    ax.add_patch(agent_box)
    ax.text(18, 72, 'DRL', ha='center', va='center',
            fontsize=8, fontweight='bold')
    ax.text(18, 68, 'Agent', ha='center', va='center',
            fontsize=7, fontweight='bold')

    # Arrow
    arrow = FancyArrowPatch((28, 70), (42, 70),
                           arrowstyle='->', mutation_scale=20,
                           linewidth=2, color='#2C3E50')
    ax.add_patch(arrow)
    ax.text(35, 72.5, 'Actions', ha='center', fontsize=6,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    # Key findings section (middle 25%)
    findings_box = FancyBboxPatch((2, 28), 96, 24,
                                  boxstyle="round,pad=0.5",
                                  facecolor='#FFE5B4', edgecolor='#2C3E50', linewidth=1.2)
    ax.add_patch(findings_box)

    ax.text(50, 48, 'Key Findings', ha='center', va='center',
            fontsize=9, fontweight='bold', color='#2C3E50')

    findings = [
        '✓ 59.9% improvement over heuristics',
        '✓ Inverted pyramid: +9.7%-19.7%',
        '✓ Capacity paradox: K=10 > K=30+',
        '✓ A2C optimal: 4438 reward, 6.9 min'
    ]

    for i, finding in enumerate(findings):
        ax.text(50, 42 - i*4.5, finding, ha='center', va='center',
                fontsize=6.5, color='#2C3E50')

    # Results section (bottom 25%)
    results_box = FancyBboxPatch((2, 2), 96, 23,
                                 boxstyle="round,pad=0.5",
                                 facecolor='#E8F4F8', edgecolor='#2C3E50', linewidth=1.2)
    ax.add_patch(results_box)

    ax.text(50, 21, 'Statistical Validation', ha='center', va='center',
            fontsize=9, fontweight='bold', color='#2C3E50')

    # Two columns for metrics
    # Left column
    metrics_left = [
        'n = 30 per group',
        'p < 10⁻⁶⁸',
        'Cohen\'s d = 6.31'
    ]

    for i, metric in enumerate(metrics_left):
        ax.text(25, 15 - i*4, metric, ha='center', va='center',
                fontsize=6, color='#2C3E50')

    # Right column
    metrics_right = [
        '500K timesteps',
        '15 algorithms',
        '5 random seeds'
    ]

    for i, metric in enumerate(metrics_right):
        ax.text(75, 15 - i*4, metric, ha='center', va='center',
                fontsize=6, color='#2C3E50')

    return fig

if __name__ == '__main__':
    fig = create_graphical_abstract()

    # Save with exact dimensions
    plt.savefig(OUTPUT_FILE, dpi=DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0)

    print(f"✅ Graphical abstract saved to {OUTPUT_FILE}")
    print(f"📐 Required: 590×590 pixels (5×5 cm at 300 DPI)")
    print(f"📊 Figure size: {SIZE_INCHES:.4f} × {SIZE_INCHES:.4f} inches")
    print(f"🎯 DPI: {DPI}")

    plt.close()

    # Verify file
    import os
    if os.path.exists(OUTPUT_FILE):
        size = os.path.getsize(OUTPUT_FILE)
        print(f"✅ File created: {size/1024:.1f} KB")
    else:
        print("❌ File creation failed")
