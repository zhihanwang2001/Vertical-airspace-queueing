"""
================================================================================
Figure 2: Algorithm Robustness Analysis - Publication-Ready Version
================================================================================
Senior Research Visualization Engineer Refactor

This script generates a publication-quality Figure 2 that is visually consistent
with Figure 1's design system (color palette, typography, stroke weights).

Design System extracted from Figure 1:
- Primary Blue: #4A90D9 (main elements, DRL agent)
- Secondary Blue: #7AB8E8 (lighter accents)
- Orange/Coral: #E8956A (highlights, action indicators)
- Green: #6BBF8A (positive indicators, servers)
- Neutral Gray: #5D5D5D (text, borders)
- Light Gray: #E8E8E8 (backgrounds, grids)
================================================================================
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ============================================================================
# STYLE SYSTEM - Unified with Figure 1
# ============================================================================

# Color Palette (extracted from Figure 1 architecture diagram)
PALETTE = {
    # Primary colors (from Fig.1)
    'primary_blue': '#4A90D9',      # Main DRL agent color
    'secondary_blue': '#7AB8E8',    # Lighter blue accents
    'orange': '#E8956A',            # Action/highlight color
    'green': '#6BBF8A',             # Server/positive indicator

    # Neutral colors
    'dark_gray': '#4D4D4D',         # Primary text
    'medium_gray': '#808080',       # Secondary text
    'light_gray': '#E8E8E8',        # Grid lines
    'white': '#FFFFFF',             # Background

    # Extended palette for additional series (same hues, varied alpha)
    'primary_blue_light': '#4A90D9',  # alpha=0.6
    'orange_light': '#E8956A',        # alpha=0.6
}

# Algorithm-specific styling (consistent with Fig.1 color system)
ALGO_STYLES = {
    'A2C': {
        'color': '#4A90D9',      # Primary blue (main signal)
        'marker': 'o',          # Circle
        'linestyle': '-',       # Solid (primary)
        'linewidth': 2.0,       # Thicker for primary
        'markersize': 8,
        'alpha': 1.0,
        'zorder': 10,           # On top
    },
    'PPO': {
        'color': '#E8956A',     # Orange (secondary signal)
        'marker': 's',          # Square
        'linestyle': '--',      # Dashed (secondary)
        'linewidth': 1.8,
        'markersize': 7,
        'alpha': 0.9,
        'zorder': 9,
    },
    'TD7': {
        'color': '#6BBF8A',     # Green (tertiary signal)
        'marker': '^',          # Triangle
        'linestyle': '-.',      # Dash-dot (tertiary)
        'linewidth': 1.6,
        'markersize': 7,
        'alpha': 0.85,
        'zorder': 8,
    },
}

# Typography settings (matching Fig.1 clean sans-serif style)
FONT_CONFIG = {
    'family': 'sans-serif',
    'sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'size': 10,
    'weight': 'normal',
}

# Figure dimensions
FIGURE_CONFIG = {
    'width': 7.0,           # inches (double-column width)
    'height': 4.5,          # inches
    'dpi': 600,             # High resolution for publication
}

# Line and marker settings
LINE_CONFIG = {
    'grid_alpha': 0.25,
    'grid_linewidth': 0.5,
    'grid_color': '#E0E0E0',
    'spine_linewidth': 0.8,
    'tick_length': 4,
    'tick_width': 0.8,
}


def setup_publication_style():
    """
    Configure matplotlib rcParams for publication-quality output.
    Ensures consistency with Figure 1's visual system.
    """
    plt.rcParams.update({
        # Font settings
        'font.family': FONT_CONFIG['family'],
        'font.sans-serif': FONT_CONFIG['sans-serif'],
        'font.size': FONT_CONFIG['size'],

        # Axes settings
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'axes.labelweight': 'medium',
        'axes.titleweight': 'bold',
        'axes.linewidth': LINE_CONFIG['spine_linewidth'],
        'axes.edgecolor': PALETTE['dark_gray'],
        'axes.labelcolor': PALETTE['dark_gray'],
        'axes.spines.top': False,
        'axes.spines.right': False,

        # Tick settings
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'xtick.color': PALETTE['dark_gray'],
        'ytick.color': PALETTE['dark_gray'],
        'xtick.major.size': LINE_CONFIG['tick_length'],
        'ytick.major.size': LINE_CONFIG['tick_length'],
        'xtick.major.width': LINE_CONFIG['tick_width'],
        'ytick.major.width': LINE_CONFIG['tick_width'],
        'xtick.direction': 'out',
        'ytick.direction': 'out',

        # Legend settings
        'legend.fontsize': 9,
        'legend.frameon': True,
        'legend.framealpha': 0.95,
        'legend.edgecolor': PALETTE['light_gray'],
        'legend.fancybox': False,

        # Grid settings
        'grid.alpha': LINE_CONFIG['grid_alpha'],
        'grid.linewidth': LINE_CONFIG['grid_linewidth'],
        'grid.color': LINE_CONFIG['grid_color'],

        # Figure settings
        'figure.facecolor': PALETTE['white'],
        'figure.edgecolor': PALETTE['white'],
        'figure.dpi': 150,  # Screen display
        'savefig.dpi': FIGURE_CONFIG['dpi'],
        'savefig.facecolor': PALETTE['white'],
        'savefig.edgecolor': PALETTE['white'],
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,

        # Line settings
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'lines.markeredgewidth': 0.8,

        # Math text
        'mathtext.fontset': 'dejavusans',
    })


def load_data(data_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load experimental data from JSON file.

    Args:
        data_path: Path to data file. If None, uses default location.

    Returns:
        DataFrame with experiment results.
    """
    if data_path is None:
        project_root = Path(__file__).parent.parent.parent
        data_path = project_root / 'Data' / 'summary' / 'comprehensive_experiments_data.json'

    with open(data_path, 'r') as f:
        data = json.load(f)

    return pd.DataFrame(data['experiments'])


def plot_fig2(
    df: pd.DataFrame,
    algorithms: List[str] = ['A2C', 'PPO', 'TD7'],
    title: Optional[str] = None,
    show_grid: bool = True,
    show_legend: bool = True,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Generate publication-quality Figure 2: Algorithm Robustness Analysis.

    Shows crash rate vs. total system capacity for multiple DRL algorithms,
    demonstrating performance consistency across different configurations.

    Args:
        df: DataFrame containing experiment data with columns:
            - 'total_capacity': System capacity values
            - 'algorithm': Algorithm names
            - 'crash_rate': Crash rate values (0-1 scale)
        algorithms: List of algorithms to plot (in order of visual priority)
        title: Optional custom title (default: standard academic title)
        show_grid: Whether to show subtle grid lines
        show_legend: Whether to show legend

    Returns:
        Tuple of (Figure, Axes) objects for further customization if needed.

    Design Notes:
        - Visual hierarchy: A2C (primary) > PPO (secondary) > TD7 (tertiary)
        - Colors match Figure 1's palette for paper consistency
        - Clean, minimal design suitable for academic publication
    """
    # Setup style
    setup_publication_style()

    # Create figure
    fig, ax = plt.subplots(
        figsize=(FIGURE_CONFIG['width'], FIGURE_CONFIG['height']),
        dpi=150  # Screen display; save will use 600
    )

    # Aggregate data: mean crash rate by capacity and algorithm
    algo_stats = df.groupby(['total_capacity', 'algorithm'])['crash_rate'].agg(['mean', 'std']).reset_index()
    algo_stats.columns = ['total_capacity', 'algorithm', 'crash_rate_mean', 'crash_rate_std']

    # Plot each algorithm with its designated style
    for algo in algorithms:
        if algo not in ALGO_STYLES:
            print(f"Warning: No style defined for algorithm '{algo}', skipping.")
            continue

        style = ALGO_STYLES[algo]
        algo_data = algo_stats[algo_stats['algorithm'] == algo].sort_values('total_capacity')

        if algo_data.empty:
            continue

        x = algo_data['total_capacity'].values
        y = algo_data['crash_rate_mean'].values * 100  # Convert to percentage

        # Plot main line with markers
        ax.plot(
            x, y,
            color=style['color'],
            marker=style['marker'],
            linestyle=style['linestyle'],
            linewidth=style['linewidth'],
            markersize=style['markersize'],
            markeredgecolor=PALETTE['white'],
            markeredgewidth=1.0,
            alpha=style['alpha'],
            zorder=style['zorder'],
            label=algo,
        )

        # Optional: Add subtle error bands if std data is meaningful
        # (Commented out for cleaner look; uncomment if needed)
        # if 'crash_rate_std' in algo_data.columns:
        #     std = algo_data['crash_rate_std'].values * 100
        #     ax.fill_between(x, y - std, y + std,
        #                     color=style['color'], alpha=0.1, zorder=1)

    # Axis labels with proper formatting
    ax.set_xlabel('Total System Capacity ($K$)', fontsize=11, fontweight='medium')
    ax.set_ylabel('Crash Rate (%)', fontsize=11, fontweight='medium')

    # Title (optional - many journals prefer no title in figure)
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold', pad=12)

    # Axis limits with padding
    x_all = algo_stats['total_capacity'].unique()
    ax.set_xlim(x_all.min() - 2, x_all.max() + 2)
    ax.set_ylim(-5, 105)

    # Y-axis: percentage formatting
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}'))

    # X-axis: integer ticks only
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=8))

    # Grid (subtle, behind data)
    if show_grid:
        ax.grid(True, linestyle=':', alpha=LINE_CONFIG['grid_alpha'],
                color=LINE_CONFIG['grid_color'], zorder=0)
        ax.set_axisbelow(True)

    # Legend (positioned to not occlude data)
    if show_legend:
        legend = ax.legend(
            loc='upper left',
            bbox_to_anchor=(0.02, 0.98),
            frameon=True,
            framealpha=0.95,
            edgecolor=PALETTE['light_gray'],
            fancybox=False,
            shadow=False,
            borderpad=0.6,
            labelspacing=0.4,
            handlelength=2.5,
            handletextpad=0.6,
        )
        legend.get_frame().set_linewidth(0.8)

    # Remove top and right spines (cleaner look)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(PALETTE['dark_gray'])
    ax.spines['bottom'].set_color(PALETTE['dark_gray'])

    # Tight layout
    fig.tight_layout()

    return fig, ax


def save_figure(
    fig: plt.Figure,
    output_path: Path,
    formats: List[str] = ['pdf', 'svg', 'png'],
    dpi: int = 600,
) -> None:
    """
    Save figure in multiple formats for publication.

    Args:
        fig: Matplotlib figure object
        output_path: Base path without extension (e.g., 'figures/fig2')
        formats: List of output formats ('pdf', 'svg', 'png')
        dpi: Resolution for raster formats (default: 600 for publication)

    Output files:
        - PDF: Vector format for print
        - SVG: Vector format for web/editing
        - PNG: High-resolution raster (600 dpi)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        save_path = output_path.with_suffix(f'.{fmt}')

        if fmt == 'png':
            fig.savefig(save_path, format=fmt, dpi=dpi,
                       bbox_inches='tight', pad_inches=0.05,
                       facecolor=PALETTE['white'], edgecolor='none')
        else:
            fig.savefig(save_path, format=fmt,
                       bbox_inches='tight', pad_inches=0.05,
                       facecolor=PALETTE['white'], edgecolor='none')

        print(f"  ✓ Saved: {save_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Generate publication-ready Figure 2: Algorithm Robustness Analysis.
    """
    print("=" * 70)
    print("Figure 2: Algorithm Robustness Analysis - Publication Quality")
    print("=" * 70)

    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / 'Manuscript' / 'Applied_Soft_Computing' / 'LaTeX' / 'figures'

    # Load data
    print("\n[1/3] Loading experimental data...")
    df = load_data()
    print(f"      Loaded {len(df)} experiment records")

    # Generate figure
    print("\n[2/3] Generating Figure 2...")
    fig, ax = plot_fig2(
        df,
        algorithms=['A2C', 'PPO', 'TD7'],
        title=None,  # No title for cleaner journal figure
        show_grid=True,
        show_legend=True,
    )

    # Save in multiple formats
    print("\n[3/3] Saving figure in multiple formats...")
    save_figure(
        fig,
        output_dir / 'fig2_algorithm_robustness',
        formats=['pdf', 'svg', 'png'],
        dpi=600
    )

    plt.close(fig)

    print("\n" + "=" * 70)
    print("✓ Figure 2 generation complete!")
    print("=" * 70)

    # Print verification checklist
    print("\n" + "-" * 70)
    print("VERIFICATION CHECKLIST")
    print("-" * 70)
    print("  [✓] Palette matches Fig.1 (blue #4A90D9, orange #E8956A, green #6BBF8A)")
    print("  [✓] Exports: PDF (vector), SVG (vector), PNG (600 dpi raster)")
    print("  [✓] Font: Sans-serif (Arial/Helvetica), consistent sizing")
    print("  [✓] No clipped labels (tight_layout + bbox_inches='tight')")
    print("  [✓] Linewidth/markers: Hierarchical (A2C=2.0, PPO=1.8, TD7=1.6)")
    print("  [✓] Visual hierarchy: Primary (solid) > Secondary (dashed) > Tertiary (dash-dot)")
    print("-" * 70)


if __name__ == '__main__':
    main()


# ============================================================================
# DESIGN NOTES
# ============================================================================
"""
Design Notes - Figure 2 Refactor
================================

1. PALETTE EXTRACTION (from Figure 1):
   - Primary Blue (#4A90D9): Extracted from DRL Agent box in Fig.1
   - Orange (#E8956A): Extracted from action/reward flow indicators
   - Green (#6BBF8A): Extracted from server indicators
   - These colors maintain visual continuity across the paper.

2. FIGURE 1-2 CONSISTENCY:
   - Same font family (sans-serif: Arial/Helvetica)
   - Same color palette (no new hues introduced)
   - Similar stroke weights and clean aesthetic
   - White background, minimal decoration

3. VISUAL HIERARCHY IMPROVEMENTS:
   - A2C (primary): Thickest line (2.0), solid, highest alpha, on top
   - PPO (secondary): Medium line (1.8), dashed, slightly lower alpha
   - TD7 (tertiary): Thinner line (1.6), dash-dot, lowest alpha
   - This creates clear visual separation without clutter.

4. PUBLICATION-READY ENHANCEMENTS:
   - 600 DPI for print quality
   - Vector formats (PDF/SVG) for scalability
   - Clean axis labels with math formatting ($K$)
   - Subtle grid (alpha=0.25) for readability
   - Legend positioned to avoid data occlusion
   - Removed top/right spines for modern academic look

5. CODE QUALITY:
   - Separated style system at top for easy modification
   - Reusable functions (plot_fig2, save_figure)
   - Type hints and docstrings for maintainability
   - Clear comments explaining design decisions
"""
