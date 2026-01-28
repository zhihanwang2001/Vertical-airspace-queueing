#!/usr/bin/env python3
"""
Create EXACT 590×590 pixels Square Graphical Abstract
Applied Soft Computing requirement: 5×5 cm at 300 DPI
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import matplotlib.patches as mpatches

# Exact configuration
OUTPUT_FILE = 'figures/graphical_abstract_final.png'
DPI = 300
SIZE_PIXELS = 590  # Exact requirement
SIZE_INCHES = SIZE_PIXELS / DPI  # 1.9666... inches

def create_final_graphical_abstract():
    """Create exact 590×590 pixel graphical abstract"""

    # Create figure with NO padding
    fig = plt.figure(figsize=(SIZE_INCHES, SIZE_INCHES), dpi=DPI)
    ax = fig.add_axes([0, 0, 1, 1], frameon=False)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # White background
    ax.add_patch(plt.Rectangle((0, 0), 100, 100, facecolor='white', zorder=0))

    # === TITLE SECTION (Top 12%) ===
    title_bg = FancyBboxPatch((1, 89), 98, 10,
                              boxstyle="round,pad=0.3",
                              facecolor='#2C3E50', edgecolor='none')
    ax.add_patch(title_bg)
    ax.text(50, 94, 'Deep RL for UAM Queueing Systems', ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')

    # === SYSTEM ARCHITECTURE (Middle-Top 28%) ===
    arch_bg = FancyBboxPatch((1, 59), 98, 28,
                             boxstyle="round,pad=0.3",
                             facecolor='#ECF0F1', edgecolor='#34495E', linewidth=1)
    ax.add_patch(arch_bg)

    # 5-layer pyramid (inverted)
    layers = [
        (80, 18, '#E74C3C', 'L5'),  # Top layer (widest)
        (75, 15, '#E67E22', 'L4'),
        (70, 12, '#F39C12', 'L3'),
        (65, 9, '#27AE60', 'L2'),
        (60, 6, '#3498DB', 'L1')   # Bottom layer (narrowest)
    ]

    for y, width, color, label in layers:
        x_center = 65
        rect = FancyBboxPatch((x_center - width/2, y), width, 3.5,
                              boxstyle="round,pad=0.2",
                              facecolor=color, edgecolor='black', linewidth=0.7)
        ax.add_patch(rect)
        ax.text(x_center, y + 1.75, label, ha='center', va='center',
                fontsize=6, fontweight='bold', color='white')

    # DRL Agent box
    agent_bg = FancyBboxPatch((8, 68), 22, 12,
                              boxstyle="round,pad=0.4",
                              facecolor='#1ABC9C', edgecolor='black', linewidth=1.2)
    ax.add_patch(agent_bg)
    ax.text(19, 76, 'DRL Agent', ha='center', va='center',
            fontsize=7, fontweight='bold', color='white')
    ax.text(19, 72, 'A2C/PPO', ha='center', va='center',
            fontsize=6, color='white')
    ax.text(19, 69, 'TD3', ha='center', va='center',
            fontsize=6, color='white')

    # Action arrow
    arrow = FancyArrowPatch((30, 74), (48, 74),
                           arrowstyle='->', mutation_scale=18,
                           linewidth=2.5, color='#E74C3C')
    ax.add_patch(arrow)
    ax.text(39, 77, 'Actions', ha='center', fontsize=5.5,
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                     edgecolor='#E74C3C', linewidth=1))

    # === KEY FINDINGS (Middle 28%) ===
    findings_bg = FancyBboxPatch((1, 29), 98, 28,
                                 boxstyle="round,pad=0.3",
                                 facecolor='#FFF9E6', edgecolor='#F39C12', linewidth=1.5)
    ax.add_patch(findings_bg)

    ax.text(50, 53, 'Key Findings', ha='center', va='center',
            fontsize=8.5, fontweight='bold', color='#2C3E50')

    findings_list = [
        ('✓', '59.9% improvement over heuristics'),
        ('✓', 'Inverted pyramid: +9.7%-19.7%'),
        ('✓', 'Capacity paradox: K=10 > K=30+'),
        ('✓', 'A2C: 4438 reward in 6.9 min')
    ]

    for i, (bullet, text) in enumerate(findings_list):
        y_pos = 47 - i * 5
        ax.text(8, y_pos, bullet, ha='center', va='center',
                fontsize=8, fontweight='bold', color='#27AE60')
        ax.text(15, y_pos, text, ha='left', va='center',
                fontsize=6.5, color='#2C3E50')

    # === STATISTICAL VALIDATION (Bottom 28%) ===
    stats_bg = FancyBboxPatch((1, 1), 98, 26,
                              boxstyle="round,pad=0.3",
                              facecolor='#E8F4F8', edgecolor='#3498DB', linewidth=1.5)
    ax.add_patch(stats_bg)

    ax.text(50, 23, 'Statistical Validation', ha='center', va='center',
            fontsize=8.5, fontweight='bold', color='#2C3E50')

    # Left column metrics
    left_metrics = [
        ('Sample Size:', 'n = 30/group'),
        ('Significance:', 'p < 10⁻⁶⁸'),
        ('Effect Size:', 'd = 6.31')
    ]

    for i, (label, value) in enumerate(left_metrics):
        y_pos = 17 - i * 4.5
        ax.text(8, y_pos, label, ha='left', va='center',
                fontsize=5.5, fontweight='bold', color='#34495E')
        ax.text(8, y_pos - 1.5, value, ha='left', va='center',
                fontsize=6, color='#2C3E50')

    # Right column metrics
    right_metrics = [
        ('Training:', '500K steps'),
        ('Algorithms:', '15 tested'),
        ('Seeds:', '5 random')
    ]

    for i, (label, value) in enumerate(right_metrics):
        y_pos = 17 - i * 4.5
        ax.text(55, y_pos, label, ha='left', va='center',
                fontsize=5.5, fontweight='bold', color='#34495E')
        ax.text(55, y_pos - 1.5, value, ha='left', va='center',
                fontsize=6, color='#2C3E50')

    # Footer
    ax.text(50, 2.5, 'Applied Soft Computing', ha='center', va='center',
            fontsize=5, style='italic', color='#7F8C8D')

    return fig

if __name__ == '__main__':
    fig = create_final_graphical_abstract()

    # Save with NO extra padding to get exact 590×590
    plt.savefig(OUTPUT_FILE, dpi=DPI, bbox_inches=None, pad_inches=0,
                facecolor='white', edgecolor='none')

    print(f"Final graphical abstract saved: {OUTPUT_FILE}")
    print(f"Target: 590×590 pixels (5×5 cm at 300 DPI)")
    print(f"Figure: {SIZE_INCHES:.6f} inches × {SIZE_INCHES:.6f} inches")
    print(f"DPI: {DPI}")
    print(f"Expected output: {SIZE_PIXELS}×{SIZE_PIXELS} pixels")

    plt.close()

    # Verify
    import os
    if os.path.exists(OUTPUT_FILE):
        size_kb = os.path.getsize(OUTPUT_FILE) / 1024
        print(f"File size: {size_kb:.1f} KB")
        print(f"
Verify with: file {OUTPUT_FILE}")
    else:
        print("Error: File not created")
