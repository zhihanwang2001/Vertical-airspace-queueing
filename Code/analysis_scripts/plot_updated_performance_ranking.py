"""
Update Performance Ranking Figure with A2C-v3 Champion Results
"""

import matplotlib.pyplot as plt
import numpy as np

# Set Chinese font
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

# Updated algorithm data (including A2C-v3, removed original high-variance SAC)
algorithms = [
    'A2C-v3',      # New champion！
    'PPO',
    'TD7',
    'R2D2',
    'SAC-v2',
    'TD3',
    'Heuristic',
    'Rainbow DQN v2',
    'Priority',
    'FCFS',
    'SJF',
    'IMPALA v2',
    'DDPG',
    'Random'
]

mean_rewards = [
    4437.86,  # A2C-v3 🔥
    4419.98,  # PPO
    4392.52,  # TD7
    4289.22,  # R2D2
    4282.94,  # SAC-v2
    3972.69,  # TD3
    2860.69,  # Heuristic
    2360.53,  # Rainbow DQN v2 (Optimized)
    2040.04,  # Priority
    2024.75,  # FCFS
    2011.16,  # SJF
    1682.19,  # IMPALA v2 (Optimized)
    1490.48,  # DDPG (Abandoned)
    294.75    # Random
]

std_rewards = [
    128.41,   # A2C-v3
    135.71,   # PPO
    84.60,    # TD7
    82.23,    # R2D2
    80.70,    # SAC-v2
    168.56,   # TD3
    87.96,    # Heuristic
    45.50,    # Rainbow DQN v2
    67.63,    # Priority
    66.64,    # FCFS
    66.58,    # SJF
    73.85,    # IMPALA v2
    102.20,   # DDPG
    308.75    # Random
]

# Algorithm type classification (for color coding)
algorithm_types = [
    'A2C-v3 (Optimized)',      # Gold - Champion
    'Policy-Based RL',   # Deep blue
    'Off-Policy RL',     # Purple
    'Value-Based RL',    # Orange
    'Off-Policy RL',     # Purple
    'Off-Policy RL',     # Purple
    'Traditional',       # Green
    'Value-Based RL (Optimized)',  # Orange
    'Traditional',       # Green
    'Traditional',       # Green
    'Traditional',       # Green
    'Distributed RL (Optimized)',  # Red
    'Off-Policy RL (Abandoned)',   # Gray
    'Baseline'           # Black
]

# Color mapping
color_map = {
    'A2C-v3 (Optimized)': '#FFD700',        # Gold - Champion
    'Policy-Based RL': '#1f77b4',      # Deep blue
    'Off-Policy RL': '#9467bd',        # Purple
    'Value-Based RL': '#ff7f0e',       # Orange
    'Value-Based RL (Optimized)': '#ff9f4a',  # Light orange
    'Traditional': '#2ca02c',          # Green
    'Distributed RL (Optimized)': '#d62728',  # Red
    'Off-Policy RL (Abandoned)': '#7f7f7f',   # Gray
    'Baseline': '#000000'              # Black
}

colors = [color_map[t] for t in algorithm_types]

# Create figure
fig, ax = plt.subplots(figsize=(16, 10))

# Draw horizontal bar chart (sorted from high to low)
y_pos = np.arange(len(algorithms))
bars = ax.barh(y_pos, mean_rewards, xerr=std_rewards,
               color=colors, alpha=0.8, edgecolor='black', linewidth=1.5,
               error_kw={'elinewidth': 2, 'capsize': 5, 'alpha': 0.7})

# Set y-axis labels
ax.set_yticks(y_pos)
ax.set_yticklabels(algorithms, fontsize=12)
ax.invert_yaxis()  # highest score at top

# Set x-axis
ax.set_xlabel('Average Reward', fontsize=14, fontweight='bold')
ax.set_xlim([0, 5000])
ax.grid(axis='x', alpha=0.3, linestyle='--')

# Add title
ax.set_title('Algorithm Performance Ranking Overview\nUpdate: A2C-v3 Delayed Cosine Annealing Optimization Tops',
             fontsize=16, fontweight='bold', pad=20)

# Add value labels on each bar
for i, (mean, std) in enumerate(zip(mean_rewards, std_rewards)):
    # Main values
    label = f'{mean:.1f}±{std:.1f}'
    x_pos = mean + std + 150

    # If A2C-v3, add special marker
    if i == 0:
        label = f'🔥 {label} 🏆'
        ax.text(x_pos, i, label, va='center', fontsize=11,
                fontweight='bold', color='darkred')
    else:
        ax.text(x_pos, i, label, va='center', fontsize=10)

# Add performance tier lines
ax.axvline(x=4200, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Top tier (>4200)')
ax.axvline(x=2000, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Mid tier (2000-4000)')

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#FFD700', edgecolor='black', label='A2C-v3 (Delayed Cosine Annealing) 🏆'),
    Patch(facecolor='#1f77b4', edgecolor='black', label='Policy-Based RL'),
    Patch(facecolor='#9467bd', edgecolor='black', label='Off-Policy RL'),
    Patch(facecolor='#ff7f0e', edgecolor='black', label='Value-Based RL'),
    Patch(facecolor='#2ca02c', edgecolor='black', label='Traditional Schedulers'),
    Patch(facecolor='#d62728', edgecolor='black', label='Distributed RL (Optimized)'),
    Patch(facecolor='#7f7f7f', edgecolor='black', label='DDPG (Abandoned)'),
    Patch(facecolor='#000000', edgecolor='black', label='Random Baseline')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=11, framealpha=0.9)

plt.tight_layout()
plt.savefig('../../Figures/analysis/figure1_performance_ranking.png', dpi=300, bbox_inches='tight')
print("✅ Performance ranking figure updated and saved: figure1_performance_ranking.png")

# ================================
# Extra: Plot Top tier algorithms detailed comparison
# ================================
fig2, ax2 = plt.subplots(figsize=(12, 8))

top_algorithms = ['A2C-v3', 'PPO', 'TD7', 'R2D2', 'SAC-v2']
top_means = [4437.86, 4419.98, 4392.52, 4289.22, 4282.94]
top_stds = [128.41, 135.71, 84.60, 82.23, 80.70]
top_colors = ['#FFD700', '#1f77b4', '#9467bd', '#ff7f0e', '#9467bd']

x_pos = np.arange(len(top_algorithms))
bars = ax2.bar(x_pos, top_means, yerr=top_stds, color=top_colors,
               alpha=0.8, edgecolor='black', linewidth=2,
               error_kw={'elinewidth': 2.5, 'capsize': 8, 'alpha': 0.8})

# Set labels
ax2.set_xticks(x_pos)
ax2.set_xticklabels(top_algorithms, fontsize=13, fontweight='bold')
ax2.set_ylabel('Average Reward', fontsize=14, fontweight='bold')
ax2.set_title('Top Tier Algorithms Detailed Comparison\nA2C-v3 vs PPO vs TD7 vs R2D2 vs SAC-v2',
              fontsize=15, fontweight='bold', pad=20)

# Add value labels
for i, (mean, std) in enumerate(zip(top_means, top_stds)):
    label = f'{mean:.1f}\n±{std:.1f}'
    ax2.text(i, mean + std + 50, label, ha='center', va='bottom',
             fontsize=11, fontweight='bold')

# Add horizontal reference lines
ax2.axhline(y=4400, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label='4400 threshold')
ax2.axhline(y=4300, color='orange', linestyle='--', linewidth=1.5, alpha=0.6, label='4300 threshold')

ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_ylim([4200, 4600])
ax2.legend(fontsize=11)

# Add training time comparison annotations
train_times = ['5.4 min', '30.8 min', '382.4 min', '115.7 min', '287.0 min']
for i, time in enumerate(train_times):
    ax2.text(i, 4220, f'Training:\n{time}', ha='center', fontsize=9,
             style='italic', color='darkblue')

plt.tight_layout()
plt.savefig('../../Figures/analysis/figure1_top_tier_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Top tier comparison figure saved: figure1_top_tier_comparison.png")

print("\n📊 Plots generated:")
print("  1. figure1_performance_ranking.png - Complete performance ranking (15 algorithms)")
print("  2. figure1_top_tier_comparison.png - Top tier detailed comparison (5 algorithms)")
