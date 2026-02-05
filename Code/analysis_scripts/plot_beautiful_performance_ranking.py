"""
Beautiful Performance Ranking Figure
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

# Updated algorithm data (removed version numbers, SAC-v2 kept as algorithm name)
algorithms = [
    'A2C',
    'PPO',
    'TD7',
    'R2D2',
    'SAC-v2',
    'TD3',
    'Heuristic',
    'Rainbow DQN',
    'Priority',
    'FCFS',
    'SJF',
    'IMPALA',
    'DDPG',
    'Random'
]

mean_rewards = [
    4437.86, 4419.98, 4351.84, 4289.22, 4282.94,
    3972.69, 2860.69, 2360.53, 2040.04, 2024.75,
    2011.16, 1682.19, 1490.48, 294.75
]

std_rewards = [
    128.41, 135.71, 51.07, 82.23, 80.70,
    168.56, 87.96, 45.50, 67.63, 66.64,
    66.58, 73.85, 102.20, 308.75
]

# Carefully designed color scheme (using gradient colors)
colors = [
    '#D4AF37',  # Gold - A2C
    '#4169E1',  # Royal Blue - PPO
    '#8A2BE2',  # Blue Violet - TD7
    '#FF6347',  # Tomato Red - R2D2
    '#9370DB',  # Medium Purple - SAC-v2
    '#20B2AA',  # Light Sea Blue - TD3
    '#32CD32',  # Lime Green - Heuristic
    '#FF8C00',  # Dark Orange - Rainbow DQN
    '#48D1CC',  # Medium Turquoise - Priority
    '#66CDAA',  # Medium Aquamarine - FCFS
    '#5F9EA0',  # Cadet Blue - SJF
    '#DC143C',  # Crimson - IMPALA
    '#A9A9A9',  # Dark Gray - DDPG
    '#2F4F4F'   # Dark Slate Gray - Random
]

# Create figure
fig, ax = plt.subplots(figsize=(14, 10))

# Draw horizontal bar chart
y_pos = np.arange(len(algorithms))
bars = ax.barh(y_pos, mean_rewards, xerr=std_rewards,
               color=colors, alpha=0.85, edgecolor='#2C3E50', linewidth=1.8,
               error_kw={'elinewidth': 2.2, 'capsize': 6, 'alpha': 0.75,
                        'ecolor': '#34495E', 'capthick': 2})

# Add special border for champion
bars[0].set_edgecolor('#B8860B')
bars[0].set_linewidth(3.5)

# Set y-axis
ax.set_yticks(y_pos)
ax.set_yticklabels(algorithms, fontsize=13, fontweight='medium')
ax.invert_yaxis()

# Set x-axis
ax.set_xlabel('Average Reward (Mean ± Std)', fontsize=15, fontweight='bold', color='#2C3E50')
ax.set_xlim([0, 5200])
ax.grid(axis='x', alpha=0.25, linestyle='--', linewidth=1.2, color='#7F8C8D')

# Title
ax.set_title('Algorithm Performance Ranking Overview\nVertical Stratified Queue Control for UAV Airspace Management',
             fontsize=17, fontweight='bold', pad=25, color='#2C3E50')

# Add value labels
for i, (mean, std, algo) in enumerate(zip(mean_rewards, std_rewards, algorithms)):
    label = f'{mean:.1f}±{std:.1f}'
    x_pos = mean + std + 180

    if i < 5:  # Top tier
        ax.text(x_pos, i, label, va='center', fontsize=11.5, fontweight='semibold', color='#2C3E50')
    else:
        ax.text(x_pos, i, label, va='center', fontsize=10.5, color='#34495E')

# Add performance tier regions
ax.axvspan(4200, 5200, alpha=0.08, color='#27AE60')
ax.axvspan(2000, 4200, alpha=0.06, color='#F39C12')
ax.axvspan(0, 2000, alpha=0.04, color='#E74C3C')

# Add tier separation lines
ax.axvline(x=4200, color='#27AE60', linestyle='--', linewidth=2.5, alpha=0.6)
ax.axvline(x=2000, color='#E67E22', linestyle='--', linewidth=2.5, alpha=0.6)

# Create legend (use Patch to correctly display tier regions)
legend_elements = [
    Patch(facecolor='#27AE60', alpha=0.3, label='Top Tier (>4200)'),
    Patch(facecolor='#F39C12', alpha=0.3, label='Mid Tier (2000-4200)'),
    Patch(facecolor='#E74C3C', alpha=0.3, label='Low Tier (<2000)')
]

# Legend
ax.legend(handles=legend_elements, loc='lower right', fontsize=12, framealpha=0.95,
          edgecolor='#2C3E50', fancybox=True, shadow=True)

# Beautify borders
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['left'].set_color('#2C3E50')
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['bottom'].set_color('#2C3E50')

plt.tight_layout()
plt.savefig('../../Figures/analysis/figure1_performance_ranking.png', dpi=400, bbox_inches='tight', facecolor='white')
print("✅ Beautiful Performance Ranking Figure Saved: figure1_performance_ranking.png")
plt.close()
