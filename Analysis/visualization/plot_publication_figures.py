"""
Publication-Quality Figure Generation Script
Generates Figures 3 and 5 with professional styling for academic journals
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

# Data paths
project_root = Path(__file__).parent.parent.parent
data_file = project_root / 'Data' / 'summary' / 'comprehensive_experiments_data.json'
output_dir = project_root / 'Analysis' / 'figures'
output_dir.mkdir(parents=True, exist_ok=True)

# Read data
with open(data_file, 'r') as f:
    data = json.load(f)

experiments = data['experiments']

# Convert to DataFrame
df = pd.DataFrame(experiments)

# ============================================================================
# Figure 3: Algorithm Robustness (Publication Quality)
# ============================================================================

def plot_algorithm_robustness_publication():
    """
    Publication-quality algorithm robustness comparison
    Shows crash rate vs capacity for A2C, PPO, TD7
    """
    # Set publication style
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 11,
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3
    })

    # Figure size for double-column (7 inches)
    fig, ax = plt.subplots(figsize=(7, 4), dpi=300)

    # Professional color palette (colorblind-friendly)
    colors = {
        'A2C': '#0173B2',    # Blue
        'PPO': '#DE8F05',    # Orange
        'TD7': '#029E73'     # Green
    }

    # Distinct markers
    markers = {
        'A2C': 'o',   # Circle
        'PPO': 's',   # Square
        'TD7': '^'    # Triangle
    }

    # Line styles
    linestyles = {
        'A2C': '-',      # Solid
        'PPO': '--',     # Dashed
        'TD7': '-.'      # Dash-dot
    }

    # Group data by capacity and algorithm
    algo_stats = df.groupby(['total_capacity', 'algorithm'])['crash_rate'].mean().unstack()
    algo_stats = algo_stats.sort_index()

    # Plot each algorithm
    for algo in ['A2C', 'PPO', 'TD7']:
        if algo in algo_stats.columns:
            ax.plot(
                algo_stats.index,
                algo_stats[algo] * 100,
                marker=markers[algo],
                color=colors[algo],
                linestyle=linestyles[algo],
                linewidth=1.5,
                markersize=6,
                markeredgewidth=0.5,
                markeredgecolor='white',
                label=algo,
                alpha=0.9,
                zorder=3
            )

    # Styling
    ax.set_xlabel('Total System Capacity', fontweight='normal')
    ax.set_ylabel('Crash Rate (%)', fontweight='normal')
    ax.set_title('Algorithm Robustness: Crash Rate vs Capacity',
                 fontweight='bold', pad=10)

    # Grid
    ax.grid(True, linestyle=':', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    # Legend
    ax.legend(
        loc='best',
        frameon=True,
        framealpha=0.95,
        edgecolor='gray',
        fancybox=False,
        shadow=False
    )

    # Axis limits
    ax.set_ylim(-5, 105)
    ax.set_xlim(algo_stats.index.min() - 1, algo_stats.index.max() + 1)

    # Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # Save in multiple formats
    plt.savefig(output_dir / 'fig3_algorithm_robustness_en.pdf',
                dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.savefig(output_dir / 'fig3_algorithm_robustness_en.png',
                dpi=300, bbox_inches='tight', pad_inches=0.05)

    print(f"✓ Saved: fig3_algorithm_robustness_en (PDF + PNG)")
    plt.close()


# ============================================================================
# Figure 5: Performance Heatmap (Publication Quality)
# ============================================================================

def plot_heatmap_publication():
    """
    Publication-quality performance heatmap
    Configuration × Algorithm crash rate matrix
    """
    # Set publication style
    sns.set_style("white")
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8
    })

    # Prepare data
    df_plot = df.copy()

    config_display_map = {
        'low_capacity': 'Low (K=10)',
        'capacity_4x5': 'Uniform (K=20)',
        'inverted_pyramid': 'Inverted (K=23)',
        'reverse_pyramid': 'Reverse (K=23)',
        'uniform': 'Uniform (K=25)',
        'capacity_6x5': 'Uniform (K=30)',
        'high_capacity': 'High (K=40)'
    }

    df_plot['config_display'] = df_plot['config_name'].map(config_display_map)

    pivot = df_plot.pivot_table(
        values='crash_rate',
        index='config_display',
        columns='algorithm'
    ) * 100

    # Sort by capacity
    capacity_order = [
        'Low (K=10)', 'Uniform (K=20)', 'Inverted (K=23)',
        'Reverse (K=23)', 'Uniform (K=25)', 'Uniform (K=30)', 'High (K=40)'
    ]
    pivot = pivot.reindex([c for c in capacity_order if c in pivot.index])

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    # Use seaborn heatmap with professional colormap
    sns.heatmap(
        pivot,
        annot=True,
        fmt='.0f',
        cmap='RdYlGn_r',
        vmin=0,
        vmax=100,
        cbar_kws={
            'label': 'Crash Rate (%)',
            'shrink': 0.8,
            'aspect': 20
        },
        linewidths=0.5,
        linecolor='white',
        square=False,
        ax=ax,
        annot_kws={'fontsize': 8, 'fontweight': 'normal'}
    )

    # Styling
    ax.set_xlabel('Algorithm', fontweight='normal', labelpad=8)
    ax.set_ylabel('Configuration', fontweight='normal', labelpad=8)
    ax.set_title('Performance Heatmap: Crash Rate by Configuration and Algorithm',
                 fontweight='bold', pad=12)

    # Rotate labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()

    # Save
    plt.savefig(output_dir / 'fig5_heatmap_en.pdf',
                dpi=300, bbox_inches='tight', pad_inches=0.05)
    plt.savefig(output_dir / 'fig5_heatmap_en.png',
                dpi=300, bbox_inches='tight', pad_inches=0.05)

    print(f"✓ Saved: fig5_heatmap_en (PDF + PNG)")
    plt.close()


# ============================================================================
# Generate all figures
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("Generating Publication-Quality Figures")
    print("=" * 60)

    print("\n[1/2] Generating Figure 3: Algorithm Robustness...")
    plot_algorithm_robustness_publication()

    print("\n[2/2] Generating Figure 5: Performance Heatmap...")
    plot_heatmap_publication()

    print("\n" + "=" * 60)
    print("✓ All figures generated successfully!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
