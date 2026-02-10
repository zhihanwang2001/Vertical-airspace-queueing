"""
================================================================================
Figures 3, 4, 5 Redesign - Publication Quality
================================================================================
Figure 3: Structural Comparison (Inverted vs Normal Pyramid)
Figure 4: Capacity Paradox (Performance degradation under extreme load)
Figure 5: Performance Heatmap (Configuration × Algorithm, axes swapped)
================================================================================
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path

# ============================================================================
# DESIGN SYSTEM - Consistent with Figure 2 redesign
# ============================================================================

COLORS = {
    # Algorithm colors
    'A2C': '#3B82F6',      # Vibrant blue
    'PPO': '#F97316',      # Warm orange
    'TD7': '#22C55E',      # Fresh green

    # Structural colors
    'inverted': '#8B5CF6',  # Purple for inverted pyramid
    'normal': '#EC4899',    # Pink for normal/reverse pyramid
    'uniform': '#6B7280',   # Gray for uniform

    # Performance colors
    'reward': '#3B82F6',    # Blue for reward
    'crash': '#EF4444',     # Red for crash rate

    # Background and accents
    'bg_light': '#F8FAFC',
    'grid': '#E2E8F0',
    'text_dark': '#1E293B',
    'text_medium': '#64748B',
    'border': '#CBD5E1',
}

# ============================================================================
# Data Loading
# ============================================================================

def load_data():
    """Load experimental data."""
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / 'Data' / 'summary' / 'comprehensive_experiments_data.json'

    with open(data_path, 'r') as f:
        data = json.load(f)

    return pd.DataFrame(data['experiments'])


# ============================================================================
# Figure 3: Structural Comparison
# ============================================================================

def create_figure3_structural_comparison():
    """
    Figure 3: Structural Comparison
    Compare Inverted Pyramid [8,6,4,3,2] vs Normal Pyramid [2,3,4,6,8]
    """
    df = load_data()

    # Setup figure
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    fig.patch.set_facecolor('white')

    # Filter data for pyramid structures
    configs = ['inverted_pyramid', 'reverse_pyramid']
    df_pyramids = df[df['config_name'].isin(configs) & df['algorithm'].isin(['A2C', 'PPO', 'TD7'])].copy()

    # Map config names to display labels
    config_labels = {
        'inverted_pyramid': 'Inverted\n[8,6,4,3,2]',
        'reverse_pyramid': 'Normal\n[2,3,4,6,8]'
    }
    df_pyramids['config_label'] = df_pyramids['config_name'].map(config_labels)

    # Pivot for grouped bar chart
    pivot_reward = df_pyramids.pivot_table(
        values='mean_reward',
        index='config_label',
        columns='algorithm'
    )

    pivot_crash = df_pyramids.pivot_table(
        values='crash_rate',
        index='config_label',
        columns='algorithm'
    ) * 100

    # Reorder columns
    algo_order = ['A2C', 'PPO', 'TD7']
    pivot_reward = pivot_reward[[a for a in algo_order if a in pivot_reward.columns]]
    pivot_crash = pivot_crash[[a for a in algo_order if a in pivot_crash.columns]]

    # Bar positions
    x = np.arange(len(pivot_reward.index))
    width = 0.25

    # Colors for algorithms
    algo_colors = {'A2C': COLORS['A2C'], 'PPO': COLORS['PPO'], 'TD7': COLORS['TD7']}

    # ===== Left Panel: Average Reward =====
    for i, algo in enumerate(pivot_reward.columns):
        offset = (i - 1) * width
        bars = ax1.bar(
            x + offset,
            pivot_reward[algo],
            width,
            label=algo,
            color=algo_colors[algo],
            edgecolor='white',
            linewidth=1.5,
            alpha=0.9,
            zorder=3
        )
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(
                f'{int(height):,}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords='offset points',
                ha='center', va='bottom',
                fontsize=9,
                fontweight='medium',
                color=COLORS['text_dark']
            )

    ax1.set_xlabel('Capacity Structure', fontsize=12, fontweight='semibold', color=COLORS['text_dark'])
    ax1.set_ylabel('Average Reward', fontsize=12, fontweight='semibold', color=COLORS['text_dark'])
    ax1.set_xticks(x)
    ax1.set_xticklabels(pivot_reward.index, fontsize=11)
    ax1.legend(title='Algorithm', fontsize=10, title_fontsize=11, frameon=True,
               framealpha=0.95, edgecolor=COLORS['border'])
    ax1.yaxis.grid(True, linestyle='-', linewidth=0.8, color=COLORS['grid'], alpha=0.7)
    ax1.set_axisbelow(True)
    for spine in ['top', 'right']:
        ax1.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax1.spines[spine].set_color(COLORS['border'])
        ax1.spines[spine].set_linewidth(1.5)

    # ===== Right Panel: Crash Rate =====
    for i, algo in enumerate(pivot_crash.columns):
        offset = (i - 1) * width
        bars = ax2.bar(
            x + offset,
            pivot_crash[algo],
            width,
            label=algo,
            color=algo_colors[algo],
            edgecolor='white',
            linewidth=1.5,
            alpha=0.9,
            zorder=3
        )
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(
                f'{height:.0f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 5),
                textcoords='offset points',
                ha='center', va='bottom',
                fontsize=9,
                fontweight='medium',
                color=COLORS['text_dark']
            )

    ax2.set_xlabel('Capacity Structure', fontsize=12, fontweight='semibold', color=COLORS['text_dark'])
    ax2.set_ylabel('Crash Rate (%)', fontsize=12, fontweight='semibold', color=COLORS['text_dark'])
    ax2.set_xticks(x)
    ax2.set_xticklabels(pivot_crash.index, fontsize=11)
    ax2.set_ylim(0, 110)
    ax2.legend(title='Algorithm', fontsize=10, title_fontsize=11, frameon=True,
               framealpha=0.95, edgecolor=COLORS['border'])
    ax2.yaxis.grid(True, linestyle='-', linewidth=0.8, color=COLORS['grid'], alpha=0.7)
    ax2.set_axisbelow(True)
    for spine in ['top', 'right']:
        ax2.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax2.spines[spine].set_color(COLORS['border'])
        ax2.spines[spine].set_linewidth(1.5)

    plt.tight_layout(pad=2.0)

    return fig


# ============================================================================
# Figure 4: Capacity Paradox
# ============================================================================

def create_figure4_capacity_paradox():
    """
    Figure 4: Capacity Paradox
    Performance degradation as total capacity increases under extreme load
    Shows counter-intuitive "less is more" phenomenon
    """
    df = load_data()

    # Setup figure
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    fig.patch.set_facecolor('white')

    # Filter for on-policy algorithms (A2C, PPO) - same evaluation protocol
    df_onpolicy = df[df['algorithm'].isin(['A2C', 'PPO'])].copy()

    # Group by capacity
    capacity_stats = df_onpolicy.groupby('total_capacity').agg({
        'mean_reward': ['mean', 'std'],
        'crash_rate': ['mean', 'std']
    }).reset_index()
    capacity_stats.columns = ['capacity', 'reward_mean', 'reward_std', 'crash_mean', 'crash_std']
    capacity_stats = capacity_stats.sort_values('capacity')

    x = capacity_stats['capacity'].values

    # ===== Left Panel: Reward vs Capacity =====
    y_reward = capacity_stats['reward_mean'].values
    std_reward = capacity_stats['reward_std'].values

    # Confidence band
    ax1.fill_between(
        x, y_reward - std_reward, y_reward + std_reward,
        color=COLORS['reward'], alpha=0.15, linewidth=0, zorder=2
    )

    # Main line
    ax1.plot(
        x, y_reward,
        color=COLORS['reward'], linewidth=3,
        marker='o', markersize=10,
        markeredgecolor='white', markeredgewidth=2,
        solid_capstyle='round', zorder=10
    )

    # Find optimal point
    max_idx = capacity_stats['reward_mean'].idxmax()
    max_capacity = capacity_stats.loc[max_idx, 'capacity']
    max_reward = capacity_stats.loc[max_idx, 'reward_mean']

    # Annotate optimal point
    ax1.annotate(
        f'Optimal\nK={int(max_capacity)}',
        xy=(max_capacity, max_reward),
        xytext=(max_capacity + 5, max_reward + 1500),
        fontsize=10, fontweight='medium',
        color=COLORS['TD7'],
        arrowprops=dict(
            arrowstyle='->',
            color=COLORS['TD7'],
            connectionstyle='arc3,rad=0.2',
            linewidth=2
        )
    )

    # Annotate performance cliff
    cliff_data = capacity_stats[capacity_stats['capacity'] >= 25]
    if len(cliff_data) >= 2:
        cliff_start = cliff_data.iloc[0]
        cliff_end = cliff_data.iloc[-1]
        ax1.annotate(
            'Performance\nCliff',
            xy=(cliff_end['capacity'], cliff_end['reward_mean']),
            xytext=(cliff_end['capacity'] - 8, cliff_end['reward_mean'] + 3000),
            fontsize=10, fontweight='medium',
            color=COLORS['crash'],
            arrowprops=dict(
                arrowstyle='->',
                color=COLORS['crash'],
                connectionstyle='arc3,rad=-0.2',
                linewidth=2
            )
        )

    ax1.set_xlabel('Total System Capacity (K)', fontsize=12, fontweight='semibold', color=COLORS['text_dark'])
    ax1.set_ylabel('Average Reward', fontsize=12, fontweight='semibold', color=COLORS['text_dark'])
    ax1.yaxis.grid(True, linestyle='-', linewidth=0.8, color=COLORS['grid'], alpha=0.7)
    ax1.set_axisbelow(True)
    for spine in ['top', 'right']:
        ax1.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax1.spines[spine].set_color(COLORS['border'])
        ax1.spines[spine].set_linewidth(1.5)

    # ===== Right Panel: Crash Rate vs Capacity =====
    y_crash = capacity_stats['crash_mean'].values * 100
    std_crash = capacity_stats['crash_std'].values * 100

    # Confidence band
    ax2.fill_between(
        x,
        np.maximum(0, y_crash - std_crash),
        np.minimum(100, y_crash + std_crash),
        color=COLORS['crash'], alpha=0.15, linewidth=0, zorder=2
    )

    # Main line
    ax2.plot(
        x, y_crash,
        color=COLORS['crash'], linewidth=3,
        marker='s', markersize=10,
        markeredgecolor='white', markeredgewidth=2,
        solid_capstyle='round', zorder=10
    )

    # Find stability boundary
    stable_capacities = capacity_stats[capacity_stats['crash_mean'] < 0.5]['capacity']
    if len(stable_capacities) > 0:
        boundary = stable_capacities.max()
        ax2.axvline(x=boundary, color=COLORS['TD7'], linestyle='--', linewidth=2, alpha=0.8, zorder=5)
        ax2.annotate(
            f'Stability\nBoundary\nK={int(boundary)}',
            xy=(boundary, 25),
            xytext=(boundary - 8, 40),
            fontsize=10, fontweight='medium',
            color=COLORS['TD7'],
            ha='right'
        )

    ax2.set_xlabel('Total System Capacity (K)', fontsize=12, fontweight='semibold', color=COLORS['text_dark'])
    ax2.set_ylabel('Crash Rate (%)', fontsize=12, fontweight='semibold', color=COLORS['text_dark'])
    ax2.set_ylim(-5, 105)
    ax2.yaxis.grid(True, linestyle='-', linewidth=0.8, color=COLORS['grid'], alpha=0.7)
    ax2.set_axisbelow(True)
    for spine in ['top', 'right']:
        ax2.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax2.spines[spine].set_color(COLORS['border'])
        ax2.spines[spine].set_linewidth(1.5)

    plt.tight_layout(pad=2.0)

    return fig


# ============================================================================
# Figure 5: Performance Heatmap (Axes Swapped)
# ============================================================================

def create_figure5_heatmap():
    """
    Figure 5: Performance Heatmap
    X-axis: Configuration
    Y-axis: Algorithm
    """
    df = load_data()

    # Setup figure
    plt.style.use('default')
    sns.set_style("white")

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    fig.patch.set_facecolor('white')

    # Prepare data
    config_display_map = {
        'low_capacity': 'Low\n(K=10)',
        'capacity_4x5': 'Uniform\n(K=20)',
        'inverted_pyramid': 'Inverted\n(K=23)',
        'reverse_pyramid': 'Normal\n(K=23)',
        'uniform': 'Uniform\n(K=25)',
        'capacity_6x5': 'Uniform\n(K=30)',
        'high_capacity': 'High\n(K=40)'
    }

    df_plot = df.copy()
    df_plot['config_display'] = df_plot['config_name'].map(config_display_map)

    # Pivot with SWAPPED axes: index=algorithm, columns=config
    pivot = df_plot.pivot_table(
        values='crash_rate',
        index='algorithm',
        columns='config_display'
    ) * 100

    # Sort columns by capacity
    capacity_order = [
        'Low\n(K=10)', 'Uniform\n(K=20)', 'Inverted\n(K=23)',
        'Normal\n(K=23)', 'Uniform\n(K=25)', 'Uniform\n(K=30)', 'High\n(K=40)'
    ]
    pivot = pivot[[c for c in capacity_order if c in pivot.columns]]

    # Sort rows by algorithm
    algo_order = ['A2C', 'PPO', 'TD7']
    pivot = pivot.reindex([a for a in algo_order if a in pivot.index])

    # Create heatmap
    sns.heatmap(
        pivot,
        annot=True,
        fmt='.0f',
        cmap='RdYlGn_r',
        center=50,
        vmin=0,
        vmax=100,
        cbar_kws={
            'label': 'Crash Rate (%)',
            'shrink': 0.8,
            'aspect': 20,
            'ticks': [0, 25, 50, 75, 100]
        },
        linewidths=2,
        linecolor='white',
        square=False,
        ax=ax,
        annot_kws={'fontsize': 12, 'fontweight': 'bold'}
    )

    # Styling
    ax.set_xlabel('Configuration', fontsize=12, fontweight='semibold', color=COLORS['text_dark'], labelpad=10)
    ax.set_ylabel('Algorithm', fontsize=12, fontweight='semibold', color=COLORS['text_dark'], labelpad=10)

    # Rotate x labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center', fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=11, fontweight='medium')

    # Colorbar styling
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('Crash Rate (%)', fontsize=11, fontweight='semibold', color=COLORS['text_dark'])

    plt.tight_layout(pad=1.5)

    return fig


# ============================================================================
# Save Functions
# ============================================================================

def save_figure(fig, output_dir, base_name):
    """Save figure in multiple formats."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save PDF
    pdf_path = output_dir / f'{base_name}.pdf'
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', pad_inches=0.1,
                facecolor='white', edgecolor='none')
    print(f"  ✓ PDF: {pdf_path}")

    # Save PNG
    png_path = output_dir / f'{base_name}.png'
    fig.savefig(png_path, format='png', dpi=600, bbox_inches='tight', pad_inches=0.1,
                facecolor='white', edgecolor='none')
    print(f"  ✓ PNG: {png_path}")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Generate all redesigned figures."""
    print("=" * 70)
    print("Figures 3, 4, 5 Redesign - Publication Quality")
    print("=" * 70)

    # Output directory
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / 'Manuscript' / 'Applied_Soft_Computing' / 'LaTeX' / 'figures'

    # Figure 3: Structural Comparison
    print("\n[1/3] Creating Figure 3: Structural Comparison...")
    fig3 = create_figure3_structural_comparison()
    save_figure(fig3, output_dir, 'fig3_structural_comparison')
    plt.close(fig3)

    # Figure 4: Capacity Paradox
    print("\n[2/3] Creating Figure 4: Capacity Paradox...")
    fig4 = create_figure4_capacity_paradox()
    save_figure(fig4, output_dir, 'fig4_capacity_paradox')
    plt.close(fig4)

    # Figure 5: Performance Heatmap
    print("\n[3/3] Creating Figure 5: Performance Heatmap...")
    fig5 = create_figure5_heatmap()
    save_figure(fig5, output_dir, 'fig5_heatmap')
    plt.close(fig5)

    print("\n" + "=" * 70)
    print("✓ All figures redesigned successfully!")
    print("=" * 70)

    print("\nDesign Features:")
    print("  • Figure 3: Grouped bar chart comparing pyramid structures")
    print("  • Figure 4: Capacity paradox with confidence bands and annotations")
    print("  • Figure 5: Heatmap with swapped axes (Config × Algorithm)")
    print("  • Consistent color palette across all figures")
    print("  • 600 DPI output for publication quality")


if __name__ == '__main__':
    main()
