"""
Professional Figure Generation Script for Applied Soft Computing Journal
Generate publication-quality figures with professional styling
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Configure professional style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'lines.linewidth': 2.5,
    'lines.markersize': 8,
    'axes.linewidth': 1.2,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.8,
})

# Colorblind-friendly palette (Okabe-Ito colors)
COLORS = {
    'blue': '#0173B2',
    'orange': '#DE8F05',
    'green': '#029E73',
    'purple': '#CC78BC',
    'brown': '#CA9161',
    'gray': '#949494',
    'red': '#D55E00',
    'yellow': '#ECE133'
}

# Data paths
project_root = Path(__file__).parent.parent.parent
data_file = project_root / 'Data' / 'summary' / 'comprehensive_experiments_data.json'
output_dir = project_root / 'Manuscript' / 'Applied_Soft_Computing' / 'LaTeX' / 'figures'
output_dir.mkdir(parents=True, exist_ok=True)

# Read data
with open(data_file, 'r') as f:
    data = json.load(f)

experiments = data['experiments']
df = pd.DataFrame(experiments)

print(f"Loaded {len(df)} experiments")
print(f"Configurations: {df['config_type'].unique()}")
print(f"Algorithms: {df['algorithm'].unique()}")


# ============================================================================
# Figure 3: Algorithm Robustness Comparison (Professional)
# ============================================================================

def plot_algorithm_robustness_professional():
    """Professional version of algorithm robustness comparison"""

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Group by capacity and algorithm
    algo_stats = df.groupby(['total_capacity', 'algorithm'])['crash_rate'].mean().unstack()
    algo_stats = algo_stats.sort_index()

    # Plot curves with professional styling
    markers = {'A2C': 'o', 'PPO': 's', 'TD7': '^'}
    colors_map = {'A2C': COLORS['blue'], 'PPO': COLORS['orange'], 'TD7': COLORS['green']}
    linestyles = {'A2C': '-', 'PPO': '--', 'TD7': '-.'}

    for algo in ['A2C', 'PPO', 'TD7']:
        if algo in algo_stats.columns:
            ax.plot(algo_stats.index, algo_stats[algo]*100,
                   marker=markers[algo], linewidth=2.5, markersize=8,
                   color=colors_map[algo], label=algo, alpha=0.9,
                   linestyle=linestyles[algo], markeredgewidth=1.5,
                   markeredgecolor='white')

    # Add horizontal reference line at 50%
    ax.axhline(y=50, color=COLORS['gray'], linestyle=':', linewidth=1.5, alpha=0.6)
    ax.text(15, 52, '50% Threshold', fontsize=9, color=COLORS['gray'], style='italic')

    # Highlight TD7 zero-crash region
    td7_zero = algo_stats[algo_stats['TD7'] == 0.0].index
    if len(td7_zero) > 0:
        ax.fill_between(td7_zero, 0, 5, color=COLORS['green'], alpha=0.1,
                        label='TD7 Zero-Crash Region')

    # Professional styling
    ax.set_xlabel('Total Capacity', fontsize=13, weight='bold')
    ax.set_ylabel('Crash Rate (%)', fontsize=13, weight='bold')
    ax.set_title('Algorithm Robustness Analysis', fontsize=14, weight='bold', pad=15)
    ax.legend(loc='upper left', frameon=True, shadow=False,
              fancybox=True, framealpha=0.95, edgecolor='gray')
    ax.grid(True, alpha=0.3, linewidth=0.8)
    ax.set_ylim(-5, 105)
    ax.set_xlim(8, 42)

    # Add minor ticks
    ax.minorticks_on()
    ax.tick_params(which='minor', length=3, color='gray')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_algorithm_robustness_en.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig3_algorithm_robustness_en.pdf', bbox_inches='tight')
    print(f"✓ Saved: fig3_algorithm_robustness_en.png/pdf")
    plt.close()


# ============================================================================
# Figure 4: Algorithm Radar Chart (Professional)
# ============================================================================

def plot_algorithm_radar_professional():
    """Professional version of radar chart"""

    # Calculate average metrics for algorithms (only viable configs, capacity <= 25)
    df_viable = df[df['total_capacity'] <= 25].copy()

    # Only compare A2C and PPO (same evaluation protocol)
    df_onpolicy_viable = df_viable[df_viable['algorithm'].isin(['A2C', 'PPO'])]

    algo_metrics_onpolicy = df_onpolicy_viable.groupby('algorithm').agg({
        'mean_reward': 'mean',
        'crash_rate': 'mean',
        'completion_rate': 'mean',
        'mean_episode_length': 'mean'
    }).reset_index()

    # Normalize
    max_reward = algo_metrics_onpolicy['mean_reward'].max()
    max_length = algo_metrics_onpolicy['mean_episode_length'].max()

    algo_metrics_onpolicy['reward_norm'] = algo_metrics_onpolicy['mean_reward'] / max_reward
    algo_metrics_onpolicy['stability_norm'] = 1 - algo_metrics_onpolicy['crash_rate']
    algo_metrics_onpolicy['completion_norm'] = algo_metrics_onpolicy['completion_rate']
    algo_metrics_onpolicy['length_norm'] = algo_metrics_onpolicy['mean_episode_length'] / max_length

    # Radar chart setup
    categories = ['Reward\n(Normalized)', 'Stability\n(1-Crash)',
                  'Completion\nRate', 'Episode\nLength']
    N = len(categories)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

    colors_radar = {'A2C': COLORS['blue'], 'PPO': COLORS['orange']}
    markers_radar = {'A2C': 'o', 'PPO': 's'}

    for idx, row in algo_metrics_onpolicy.iterrows():
        algo = row['algorithm']
        values = [
            row['reward_norm'],
            row['stability_norm'],
            row['completion_norm'],
            row['length_norm']
        ]
        values += values[:1]

        ax.plot(angles, values, marker=markers_radar[algo], linewidth=2.5,
               label=algo, color=colors_radar[algo], alpha=0.9, markersize=8,
               markeredgewidth=1.5, markeredgecolor='white')
        ax.fill(angles, values, alpha=0.2, color=colors_radar[algo])

    # Professional styling
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12, weight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    ax.set_title('Multi-Dimensional Performance Comparison\n(Viable Configurations, K≤25)',
                 fontsize=14, weight='bold', pad=25)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1),
              fontsize=12, frameon=True, shadow=False, edgecolor='gray')
    ax.grid(True, alpha=0.3, linewidth=0.8)

    # Add radial grid lines
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8, 1.0], angle=45, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_algorithm_radar_en.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig4_algorithm_radar_en.pdf', bbox_inches='tight')
    print(f"✓ Saved: fig4_algorithm_radar_en.png/pdf")
    plt.close()


# ============================================================================
# Figure 5: Performance Heatmap (Professional)
# ============================================================================

def plot_performance_heatmap_professional():
    """Professional version of performance heatmap"""

    # Prepare data - crash rate heatmap
    df_plot = df.copy()

    # Create display labels
    config_display_map = {
        'low_capacity': 'Low Cap (10)',
        'capacity_4x5': 'Uniform (20)',
        'inverted_pyramid': 'Inverted (23)',
        'reverse_pyramid': 'Reverse (23)',
        'uniform': 'Uniform (25)',
        'capacity_6x5': 'Uniform (30)',
        'high_capacity': 'High Cap (40)'
    }

    df_plot['config_display'] = df_plot['config_name'].map(config_display_map)

    pivot = df_plot.pivot_table(
        values='crash_rate',
        index='config_display',
        columns='algorithm'
    ) * 100

    # Sort by total capacity
    capacity_order_map = {
        'Low Cap (10)': 10,
        'Uniform (20)': 20,
        'Inverted (23)': 23,
        'Reverse (23)': 23.5,
        'Uniform (25)': 25,
        'Uniform (30)': 30,
        'High Cap (40)': 40
    }

    pivot['_sort_key'] = pivot.index.map(capacity_order_map)
    pivot = pivot.sort_values('_sort_key').drop('_sort_key', axis=1)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Create heatmap with professional colormap
    im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto',
                   vmin=0, vmax=100, interpolation='nearest')

    # Set ticks
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, fontsize=11, weight='bold')
    ax.set_yticklabels(pivot.index, fontsize=11)

    # Selective annotations (only for extreme values and key points)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.values[i, j]
            # Only annotate 0% and 100% crash rates
            if value == 0 or value == 100:
                color = 'white' if value > 50 else 'black'
                ax.text(j, i, f'{value:.0f}%', ha="center", va="center",
                       color=color, fontsize=10, weight='bold')

    # Professional styling
    ax.set_title('Crash Rate Across Configurations and Algorithms',
                 fontsize=14, weight='bold', pad=20)
    ax.set_xlabel('Algorithm', fontsize=13, weight='bold')
    ax.set_ylabel('Configuration (Type, Capacity)', fontsize=13, weight='bold')

    # Colorbar with professional styling
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Crash Rate (%)', fontsize=12, weight='bold')
    cbar.ax.tick_params(labelsize=10)

    # Add subtle grid lines
    ax.set_xticks(np.arange(len(pivot.columns)+1)-.5, minor=True)
    ax.set_yticks(np.arange(len(pivot.index)+1)-.5, minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", size=0)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_heatmap_en.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'fig5_heatmap_en.pdf', bbox_inches='tight')
    print(f"✓ Saved: fig5_heatmap_en.png/pdf")
    plt.close()


# ============================================================================
# Generate all professional figures
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("Generating Professional Figures for Applied Soft Computing Journal")
    print("="*80 + "\n")

    print("Configuration:")
    print(f"  - Font: Times New Roman (serif)")
    print(f"  - DPI: 300")
    print(f"  - Color scheme: Colorblind-friendly (Okabe-Ito)")
    print(f"  - Output directory: {output_dir}")
    print()

    plot_algorithm_robustness_professional()
    plot_algorithm_radar_professional()
    plot_performance_heatmap_professional()

    print("\n" + "="*80)
    print(f"All professional figures generated successfully!")
    print(f"Output location: {output_dir}")
    print("="*80 + "\n")

    # List generated files
    figures = list(output_dir.glob('fig[345]*_en.png'))
    print("Generated figure files:")
    for fig in sorted(figures):
        size_mb = fig.stat().st_size / (1024 * 1024)
        print(f"  ✓ {fig.name} ({size_mb:.2f} MB)")
