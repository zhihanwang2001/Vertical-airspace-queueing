"""
Generate comprehensive figures for capacity paradox analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10

# Create output directory
output_dir = Path("Analysis/figures")
output_dir.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv("Data/summary/capacity_scan_results_complete.csv")
df_rl = df[df['family'] == 'RL'].copy()

print("Generating comprehensive figures...")

# Figure 1: Capacity Paradox - Reward vs Load
print("\n1. Creating reward vs load comparison...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

loads = sorted(df_rl['load_multiplier'].unique())
k10_rewards = []
k30_rewards = []
k10_std = []
k30_std = []

for load in loads:
    k10_data = df_rl[(df_rl['load_multiplier'] == load) & (df_rl['total_capacity'] == 10)]
    k30_data = df_rl[(df_rl['load_multiplier'] == load) & (df_rl['total_capacity'] == 30)]

    k10_rewards.append(k10_data['mean_reward'].mean())
    k30_rewards.append(k30_data['mean_reward'].mean())
    k10_std.append(k10_data['mean_reward'].std())
    k30_std.append(k30_data['mean_reward'].std())

# Plot rewards
ax1.plot(loads, k10_rewards, 'o-', linewidth=2.5, markersize=8,
         label='K=10', color='#2E7D32', alpha=0.9)
ax1.plot(loads, k30_rewards, 's-', linewidth=2.5, markersize=8,
         label='K=30', color='#C62828', alpha=0.9)
ax1.fill_between(loads,
                  np.array(k10_rewards) - np.array(k10_std),
                  np.array(k10_rewards) + np.array(k10_std),
                  alpha=0.2, color='#2E7D32')
ax1.fill_between(loads,
                  np.array(k30_rewards) - np.array(k30_std),
                  np.array(k30_rewards) + np.array(k30_std),
                  alpha=0.2, color='#C62828')

ax1.axvline(x=5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Transition ~5×')
ax1.set_xlabel('Load Multiplier', fontweight='bold')
ax1.set_ylabel('Mean Reward', fontweight='bold')
ax1.set_title('Capacity Paradox: Reward vs Load', fontweight='bold', fontsize=14)
ax1.legend(loc='best', framealpha=0.9)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(loads)

# Plot crash rates
k10_crash = []
k30_crash = []

for load in loads:
    k10_data = df_rl[(df_rl['load_multiplier'] == load) & (df_rl['total_capacity'] == 10)]
    k30_data = df_rl[(df_rl['load_multiplier'] == load) & (df_rl['total_capacity'] == 30)]

    k10_crash.append(k10_data['crash_rate'].mean() * 100)
    k30_crash.append(k30_data['crash_rate'].mean() * 100)

ax2.plot(loads, k10_crash, 'o-', linewidth=2.5, markersize=8,
         label='K=10', color='#2E7D32', alpha=0.9)
ax2.plot(loads, k30_crash, 's-', linewidth=2.5, markersize=8,
         label='K=30', color='#C62828', alpha=0.9)
ax2.axvline(x=5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
ax2.set_xlabel('Load Multiplier', fontweight='bold')
ax2.set_ylabel('Crash Rate (%)', fontweight='bold')
ax2.set_title('System Stability vs Load', fontweight='bold', fontsize=14)
ax2.legend(loc='best', framealpha=0.9)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(loads)
ax2.set_ylim(-5, 105)

plt.tight_layout()
fig_path = output_dir / 'capacity_paradox_comprehensive.png'
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"   ✅ Saved: {fig_path}")
plt.close()

print("\n✅ All figures generated successfully!")
print(f"📁 Output directory: {output_dir}")
