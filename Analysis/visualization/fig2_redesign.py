"""
================================================================================
Figure 2: Algorithm Robustness Analysis - Complete Redesign
================================================================================
Matching Figure 1's hand-drawn aesthetic with professional publication quality.

Key Design Elements from Figure 1:
- Soft, rounded visual style
- Pastel color palette with depth
- Clean white background
- Professional but approachable aesthetic
- Clear visual hierarchy
================================================================================
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================================
# DESIGN SYSTEM - Matching Figure 1's Hand-drawn Style
# ============================================================================

# Color palette extracted from Figure 1
COLORS = {
    'A2C': '#3B82F6',      # Vibrant blue (primary)
    'PPO': '#F97316',      # Warm orange (secondary)
    'TD7': '#22C55E',      # Fresh green (tertiary)

    # Background and accents
    'bg_light': '#F8FAFC',
    'grid': '#E2E8F0',
    'text_dark': '#1E293B',
    'text_medium': '#64748B',
    'border': '#CBD5E1',
}

# Algorithm display configuration
ALGO_CONFIG = {
    'A2C': {
        'color': COLORS['A2C'],
        'color_light': '#DBEAFE',
        'marker': 'o',
        'label': 'A2C',
        'order': 1,
    },
    'PPO': {
        'color': COLORS['PPO'],
        'color_light': '#FFEDD5',
        'marker': 's',
        'label': 'PPO',
        'order': 2,
    },
    'TD7': {
        'color': COLORS['TD7'],
        'color_light': '#DCFCE7',
        'marker': '^',
        'label': 'TD7',
        'order': 3,
    },
}


def load_data():
    """Load experimental data."""
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / 'Data' / 'summary' / 'comprehensive_experiments_data.json'

    with open(data_path, 'r') as f:
        data = json.load(f)

    return pd.DataFrame(data['experiments'])


def create_figure2_redesign():
    """
    Create a completely redesigned Figure 2 with modern, clean aesthetic
    matching Figure 1's style.
    """
    # Load and process data
    df = load_data()

    # Aggregate by capacity and algorithm
    stats = df.groupby(['total_capacity', 'algorithm'])['crash_rate'].agg(['mean', 'std']).reset_index()
    stats.columns = ['capacity', 'algorithm', 'mean', 'std']
    stats['mean_pct'] = stats['mean'] * 100
    stats['std_pct'] = stats['std'] * 100

    # Setup figure with custom style
    plt.style.use('default')

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Plot each algorithm with enhanced styling
    algorithms = ['A2C', 'PPO', 'TD7']

    for algo in algorithms:
        config = ALGO_CONFIG[algo]
        algo_data = stats[stats['algorithm'] == algo].sort_values('capacity')

        if algo_data.empty:
            continue

        x = algo_data['capacity'].values
        y = algo_data['mean_pct'].values
        std = algo_data['std_pct'].values

        # Plot confidence band (subtle fill)
        ax.fill_between(
            x,
            np.maximum(0, y - std),
            np.minimum(100, y + std),
            color=config['color'],
            alpha=0.15,
            linewidth=0,
            zorder=config['order']
        )

        # Plot main line with enhanced styling
        line, = ax.plot(
            x, y,
            color=config['color'],
            linewidth=3,
            solid_capstyle='round',
            solid_joinstyle='round',
            zorder=config['order'] + 10,
            label=config['label']
        )

        # Plot markers with white edge for pop
        ax.scatter(
            x, y,
            s=120,
            c=config['color'],
            marker=config['marker'],
            edgecolors='white',
            linewidths=2.5,
            zorder=config['order'] + 20
        )

        # Add value labels at key points
        for i, (xi, yi) in enumerate(zip(x, y)):
            if yi > 5 and i % 2 == 0:  # Label every other point if significant
                offset = 12 if yi < 90 else -18
                ax.annotate(
                    f'{yi:.0f}%',
                    (xi, yi),
                    textcoords='offset points',
                    xytext=(0, offset),
                    ha='center',
                    fontsize=9,
                    fontweight='medium',
                    color=config['color'],
                    alpha=0.9
                )

    # Styling - Clean, modern look
    ax.set_xlim(8, 42)
    ax.set_ylim(-5, 110)

    # Grid - subtle horizontal lines only
    ax.yaxis.grid(True, linestyle='-', linewidth=0.8, color=COLORS['grid'], alpha=0.7)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    # Spines - minimal
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color(COLORS['border'])
        ax.spines[spine].set_linewidth(1.5)

    # Labels with modern typography
    ax.set_xlabel(
        'Total System Capacity (K)',
        fontsize=13,
        fontweight='semibold',
        color=COLORS['text_dark'],
        labelpad=12
    )
    ax.set_ylabel(
        'Crash Rate (%)',
        fontsize=13,
        fontweight='semibold',
        color=COLORS['text_dark'],
        labelpad=12
    )

    # Ticks
    ax.tick_params(
        axis='both',
        which='major',
        labelsize=11,
        colors=COLORS['text_medium'],
        length=6,
        width=1.5
    )
    ax.set_xticks([10, 15, 20, 25, 30, 35, 40])
    ax.set_yticks([0, 20, 40, 60, 80, 100])

    # Custom legend with colored boxes
    legend_elements = []
    for algo in algorithms:
        config = ALGO_CONFIG[algo]
        legend_elements.append(
            plt.Line2D(
                [0], [0],
                marker=config['marker'],
                color=config['color'],
                linewidth=3,
                markersize=10,
                markeredgecolor='white',
                markeredgewidth=2,
                label=config['label']
            )
        )

    legend = ax.legend(
        handles=legend_elements,
        loc='upper left',
        frameon=True,
        framealpha=0.95,
        edgecolor=COLORS['border'],
        fancybox=True,
        shadow=False,
        fontsize=11,
        title='Algorithm',
        title_fontsize=12,
        borderpad=0.8,
        labelspacing=0.6,
        handlelength=2.5
    )
    legend.get_frame().set_linewidth(1.5)
    legend.get_title().set_fontweight('semibold')

    # Add subtle annotation for key insight
    ax.annotate(
        'Critical threshold',
        xy=(25, 60),
        xytext=(32, 45),
        fontsize=10,
        color=COLORS['text_medium'],
        style='italic',
        arrowprops=dict(
            arrowstyle='->',
            color=COLORS['text_medium'],
            connectionstyle='arc3,rad=0.2',
            linewidth=1.5
        )
    )

    # Tight layout
    plt.tight_layout(pad=1.5)

    return fig, ax


def save_figure(fig, output_dir):
    """Save figure in multiple formats."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = 'fig2_algorithm_robustness'

    # Save PDF (vector)
    pdf_path = output_dir / f'{base_name}.pdf'
    fig.savefig(
        pdf_path,
        format='pdf',
        bbox_inches='tight',
        pad_inches=0.1,
        facecolor='white',
        edgecolor='none'
    )
    print(f"  ✓ PDF: {pdf_path}")

    # Save PNG (high-res raster)
    png_path = output_dir / f'{base_name}.png'
    fig.savefig(
        png_path,
        format='png',
        dpi=600,
        bbox_inches='tight',
        pad_inches=0.1,
        facecolor='white',
        edgecolor='none'
    )
    print(f"  ✓ PNG: {png_path}")

    # Save SVG (vector)
    svg_path = output_dir / f'{base_name}.svg'
    fig.savefig(
        svg_path,
        format='svg',
        bbox_inches='tight',
        pad_inches=0.1,
        facecolor='white',
        edgecolor='none'
    )
    print(f"  ✓ SVG: {svg_path}")


def main():
    """Main execution."""
    print("=" * 70)
    print("Figure 2 Redesign - Modern Publication Quality")
    print("=" * 70)

    # Setup output path
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / 'Manuscript' / 'Applied_Soft_Computing' / 'LaTeX' / 'figures'

    print("\n[1/2] Creating redesigned figure...")
    fig, ax = create_figure2_redesign()

    print("\n[2/2] Saving outputs...")
    save_figure(fig, output_dir)

    plt.close(fig)

    print("\n" + "=" * 70)
    print("✓ Figure 2 redesign complete!")
    print("=" * 70)

    print("\nDesign Features:")
    print("  • Vibrant color palette (blue/orange/green)")
    print("  • Confidence bands showing variance")
    print("  • Large markers with white edges")
    print("  • Value annotations at key points")
    print("  • Horizontal grid lines only")
    print("  • Modern legend with algorithm title")
    print("  • Critical threshold annotation")


if __name__ == '__main__':
    main()
