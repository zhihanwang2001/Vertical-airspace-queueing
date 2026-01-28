"""
Visualization Analysis Script - Generate Paper Figures
Visualization Script - Generate Paper Figures
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Set Chinese font - Fix Chinese display issues
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'

# Chinese fonts available on macOS
import platform
if platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'PingFang SC', 'STHeiti']
else:  # Windows/Linux
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']

plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display issue

# Set chart style
plt.style.use('seaborn-v0_8-darkgrid')

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

print(f"Loaded {len(df)} experiment records")
print(f"Configuration types: {df['config_type'].unique()}")
print(f"Algorithms: {df['algorithm'].unique()}")


# ============================================================================
# Figure 1: Capacity-Performance Curve (Core Contribution)
# ============================================================================

def plot_capacity_performance():
    """Capacity vs Performance curve - showing capacity paradox and performance cliff"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Prepare data - only A2C and PPO (same evaluation protocol)
    df_onpolicy = df[df['algorithm'].isin(['A2C', 'PPO'])].copy()

    # Group by capacity
    capacity_stats = df_onpolicy.groupby('total_capacity').agg({
        'mean_reward': 'mean',
        'crash_rate': 'mean'
    }).reset_index()

    capacity_stats = capacity_stats.sort_values('total_capacity')

    # Left: Average Reward vs Capacity
    ax1.plot(capacity_stats['total_capacity'], capacity_stats['mean_reward'],
             'o-', linewidth=2, markersize=8, color='steelblue', label='Average Reward')

    # Annotate key points
    max_idx = capacity_stats['mean_reward'].idxmax()
    max_capacity = capacity_stats.loc[max_idx, 'total_capacity']
    max_reward = capacity_stats.loc[max_idx, 'mean_reward']
    ax1.annotate(f'Optimal: Capacity {int(max_capacity)}\nReward {max_reward:.0f}',
                xy=(max_capacity, max_reward),
                xytext=(max_capacity+3, max_reward+2000),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, color='red', weight='bold')

    # Annotate performance cliff
    cliff_data = capacity_stats[capacity_stats['total_capacity'] >= 25]
    if len(cliff_data) >= 2:
        cliff_start = cliff_data.iloc[0]
        cliff_end = cliff_data.iloc[1]
        ax1.annotate(f'Performance Cliff\n-99.8%',
                    xy=(cliff_end['total_capacity'], cliff_end['mean_reward']),
                    xytext=(cliff_end['total_capacity']-5, cliff_end['mean_reward']+3000),
                    arrowprops=dict(arrowstyle='->', color='darkred', lw=2),
                    fontsize=10, color='darkred', weight='bold')

    ax1.set_xlabel('Total Capacity', fontsize=12, weight='bold')
    ax1.set_ylabel('Average Reward', fontsize=12, weight='bold')
    ax1.set_title('Capacity-Performance Relationship (A2C+PPO Average)', fontsize=14, weight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('symlog')  # Log scale for handling negative values

    # Right: Crash Rate vs Capacity
    ax2.plot(capacity_stats['total_capacity'], capacity_stats['crash_rate']*100,
             'o-', linewidth=2, markersize=8, color='crimson', label='Crash Rate')

    # Annotate stability boundary
    boundary = capacity_stats[capacity_stats['crash_rate'] < 1.0]['total_capacity'].max()
    ax2.axvline(x=boundary, color='green', linestyle='--', linewidth=2, alpha=0.7)
    ax2.text(boundary-2, 50, f'Stability Boundary\nCapacity {int(boundary)}',
             fontsize=10, color='green', weight='bold',
             ha='right', va='center')

    ax2.set_xlabel('Total Capacity', fontsize=12, weight='bold')
    ax2.set_ylabel('Crash Rate (%)', fontsize=12, weight='bold')
    ax2.set_title('Capacity-Crash Rate Relationship', fontsize=14, weight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-5, 105)

    plt.tight_layout()
    plt.savefig(output_dir / 'figure1_capacity_performance.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figure1_capacity_performance.pdf', bbox_inches='tight')
    print(f"Saved: figure1_capacity_performance.png/pdf")
    plt.close()


# ============================================================================
# Figure 2: Structure Comparison (Validate Design Advantage)
# ============================================================================

def plot_structure_comparison():
    """Compare Inverted Pyramid, Normal Pyramid, Uniform (Capacity 23)"""

    # Filter capacity 23 structures
    configs_23 = ['inverted_pyramid', 'reverse_pyramid']
    df_23 = df[df['config_name'].isin(configs_23) & df['algorithm'].isin(['A2C', 'PPO'])].copy()

    # Add uniform 25 for comparison
    df_uniform = df[(df['config_name'] == 'uniform') & df['algorithm'].isin(['A2C', 'PPO'])].copy()
    df_comparison = pd.concat([df_23, df_uniform])

    # Map config names
    config_labels = {
        'inverted_pyramid': 'Inverted Pyramid\n[8,6,4,3,2]',
        'reverse_pyramid': 'Normal Pyramid\n[2,3,4,6,8]',
        'uniform': 'Uniform\n[5,5,5,5,5]'
    }
    df_comparison['config_label'] = df_comparison['config_name'].map(config_labels)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Average Reward Comparison
    pivot_reward = df_comparison.pivot_table(
        values='mean_reward',
        index='config_label',
        columns='algorithm'
    )

    x = np.arange(len(pivot_reward))
    width = 0.35

    ax1.bar(x - width/2, pivot_reward['A2C'], width, label='A2C', color='steelblue', alpha=0.8)
    ax1.bar(x + width/2, pivot_reward['PPO'], width, label='PPO', color='coral', alpha=0.8)

    ax1.set_xlabel('Capacity Structure', fontsize=12, weight='bold')
    ax1.set_ylabel('Average Reward', fontsize=12, weight='bold')
    ax1.set_title('Structure Comparison - Average Reward', fontsize=14, weight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(pivot_reward.index, fontsize=10)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')

    # Right: Crash Rate Comparison
    pivot_crash = df_comparison.pivot_table(
        values='crash_rate',
        index='config_label',
        columns='algorithm'
    ) * 100

    ax2.bar(x - width/2, pivot_crash['A2C'], width, label='A2C', color='steelblue', alpha=0.8)
    ax2.bar(x + width/2, pivot_crash['PPO'], width, label='PPO', color='coral', alpha=0.8)

    ax2.set_xlabel('Capacity Structure', fontsize=12, weight='bold')
    ax2.set_ylabel('Crash Rate (%)', fontsize=12, weight='bold')
    ax2.set_title('Structure Comparison - Crash Rate', fontsize=14, weight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(pivot_crash.index, fontsize=10)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'figure2_structure_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figure2_structure_comparison.pdf', bbox_inches='tight')
    print(f"Saved: figure2_structure_comparison.png/pdf")
    plt.close()


# ============================================================================
# Figure 3: Algorithm Robustness Comparison
# ============================================================================

def plot_algorithm_robustness():
    """Compare A2C, PPO, TD7 crash rates across different capacities"""

    fig, ax = plt.subplots(figsize=(12, 6))

    # Group by capacity and algorithm
    algo_stats = df.groupby(['total_capacity', 'algorithm'])['crash_rate'].mean().unstack()
    algo_stats = algo_stats.sort_index()

    # Plot curves
    markers = {'A2C': 'o', 'PPO': 's', 'TD7': '^'}
    colors = {'A2C': 'steelblue', 'PPO': 'coral', 'TD7': 'forestgreen'}

    for algo in ['A2C', 'PPO', 'TD7']:
        if algo in algo_stats.columns:
            ax.plot(algo_stats.index, algo_stats[algo]*100,
                   marker=markers[algo], linewidth=2, markersize=8,
                   color=colors[algo], label=algo, alpha=0.8)

    # Annotate TD7 zero-crash region
    td7_zero = algo_stats[algo_stats['TD7'] == 0.0].index
    if len(td7_zero) > 0:
        ax.fill_between(td7_zero, 0, 5, color='green', alpha=0.1,
                        label='TD7 Zero-Crash Region')

    ax.set_xlabel('Total Capacity', fontsize=12, weight='bold')
    ax.set_ylabel('Crash Rate (%)', fontsize=12, weight='bold')
    ax.set_title('Algorithm Robustness Comparison - Crash Rate vs Capacity', fontsize=14, weight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)

    plt.tight_layout()
    plt.savefig(output_dir / 'figure3_algorithm_robustness.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figure3_algorithm_robustness.pdf', bbox_inches='tight')
    print(f"Saved: figure3_algorithm_robustness.png/pdf")
    plt.close()


# ============================================================================
# Figure 4: Algorithm Comprehensive Performance Comparison (Radar Chart)
# ============================================================================

def plot_algorithm_radar():
    """Algorithm comprehensive performance radar chart"""

    from matplotlib.patches import Circle, RegularPolygon
    from matplotlib.path import Path
    from matplotlib.projections.polar import PolarAxes
    from matplotlib.projections import register_projection
    from matplotlib.spines import Spine
    from matplotlib.transforms import Affine2D

    # Calculate algorithm average metrics (only viable configs, capacity <= 25)
    df_viable = df[df['total_capacity'] <= 25].copy()

    algo_metrics = df_viable.groupby('algorithm').agg({
        'mean_reward': 'mean',
        'crash_rate': 'mean',
        'completion_rate': 'mean',
        'mean_episode_length': 'mean'
    }).reset_index()

    # Normalize metrics to 0-1 (for radar chart)
    # Reward: positive metric, higher is better
    # Crash rate: negative metric, lower is better -> convert to (1 - crash_rate)
    # Completion rate: positive metric
    # Episode length: positive metric (longer means more stable)

    # Note: TD7 episode length is much larger than A2C/PPO (different protocol), need separate normalization
    df_onpolicy_viable = df_viable[df_viable['algorithm'].isin(['A2C', 'PPO'])]
    df_td7_viable = df_viable[df_viable['algorithm'] == 'TD7']

    # Only compare A2C and PPO (same evaluation protocol)
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

    # Radar chart
    categories = ['Reward', 'Stability\n(1-Crash Rate)', 'Completion Rate', 'Episode Length']
    N = len(categories)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

    colors_radar = {'A2C': 'steelblue', 'PPO': 'coral'}

    for idx, row in algo_metrics_onpolicy.iterrows():
        algo = row['algorithm']
        values = [
            row['reward_norm'],
            row['stability_norm'],
            row['completion_norm'],
            row['length_norm']
        ]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=algo,
               color=colors_radar[algo], alpha=0.7)
        ax.fill(angles, values, alpha=0.15, color=colors_radar[algo])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title('A2C vs PPO Comprehensive Performance Comparison\n(Only viable configs, capacity≤25)',
                fontsize=14, weight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / 'figure4_algorithm_radar.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figure4_algorithm_radar.pdf', bbox_inches='tight')
    print(f"Saved: figure4_algorithm_radar.png/pdf")
    plt.close()


# ============================================================================
# Figure 5: Detailed Experiment Heatmap
# ============================================================================

def plot_heatmap():
    """Experiment results heatmap - Config × Algorithm"""

    # Prepare data - crash rate heatmap
    df_plot = df.copy()
    df_plot['config_display'] = df_plot['config_type'] + '\n' + df_plot['total_capacity'].astype(str)

    pivot = df_plot.pivot_table(
        values='crash_rate',
        index='config_display',
        columns='algorithm'
    ) * 100

    # Sort by total capacity
    capacity_order = df_plot.groupby('config_display')['total_capacity'].first().sort_values()
    pivot = pivot.reindex(capacity_order.index)

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=100)

    # Set ticks
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, fontsize=11)
    ax.set_yticklabels(pivot.index, fontsize=10)

    # Add value annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.values[i, j]
            color = 'white' if value > 50 else 'black'
            text = ax.text(j, i, f'{value:.0f}%',
                          ha="center", va="center", color=color, fontsize=10)

    ax.set_title('Crash Rate Heatmap - Config × Algorithm', fontsize=14, weight='bold', pad=15)
    ax.set_xlabel('Algorithm', fontsize=12, weight='bold')
    ax.set_ylabel('Configuration (Type\nCapacity)', fontsize=12, weight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Crash Rate (%)', fontsize=11, weight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'figure5_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figure5_heatmap.pdf', bbox_inches='tight')
    print(f"Saved: figure5_heatmap.png/pdf")
    plt.close()


# ============================================================================
# Generate all figures
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("Starting to generate paper figures...")
    print("="*80 + "\n")

    plot_capacity_performance()
    plot_structure_comparison()
    plot_algorithm_robustness()
    plot_algorithm_radar()
    plot_heatmap()

    print("\n" + "="*80)
    print(f"All figures generated and saved to: {output_dir}")
    print("="*80 + "\n")

    # List generated files
    figures = list(output_dir.glob('*.png'))
    print("Generated figure files:")
    for fig in sorted(figures):
        print(f"  - {fig.name}")
