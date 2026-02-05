"""
Plot Training Curves for Optimized Algorithms
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set Chinese font
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

# Read CSV data
def load_training_data(csv_path):
    """Read training data"""
    df = pd.read_csv(csv_path)
    return df['Step'].values, df['Value'].values

# Smoothed curve
def smooth_curve(values, weight=0.9):
    """Exponential moving average smoothing"""
    smoothed = []
    last = values[0]
    for point in values:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Optimized Algorithms Training Curves Comparison',
             fontsize=16, fontweight='bold')

# ================================
# Subplot 1: A2C v3 Delayed Cosine Annealing (Unorthodox Method)
# ================================
ax1 = axes[0, 0]
steps, rewards = load_training_data('result_excel/SB3_A2C.csv')
rewards_smooth = smooth_curve(rewards, weight=0.95)

ax1.plot(steps, rewards, alpha=0.2, color='#FF6B6B', linewidth=0.5, label='Raw data')
ax1.plot(steps, rewards_smooth, color='#FF6B6B', linewidth=2.5, label='Smoothed curve')

# Mark 300k step boundary (delayed cosine annealing start point)
ax1.axvline(x=300000, color='green', linestyle='--', linewidth=2, alpha=0.7,
            label='Cosine annealing starts (300k steps)')

# Add text annotations
ax1.text(150000, 4000, 'First 300k steps:\nFixed lr=7e-4\nFull exploration',
         fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax1.text(400000, 4000, 'Last 200k steps:\nCosine annealing to 1e-5\nStable convergence',
         fontsize=10, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

ax1.set_xlabel('Training Steps', fontsize=12)
ax1.set_ylabel('Average Reward', fontsize=12)
ax1.set_title('A2C v3 - Delayed Cosine Annealing 🔥\nFinal: 4437.86±128.41 (Rank 1)',
              fontsize=13, fontweight='bold')
ax1.legend(loc='lower right', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([-100, 5000])

# ================================
# Subplot 2: Rainbow DQN v2 Stability Optimization
# ================================
ax2 = axes[0, 1]
steps, rewards = load_training_data('result_excel/Rainbow_DQN.csv')
rewards_smooth = smooth_curve(rewards, weight=0.95)

ax2.plot(steps, rewards, alpha=0.2, color='#4ECDC4', linewidth=0.5, label='Raw data')
ax2.plot(steps, rewards_smooth, color='#4ECDC4', linewidth=2.5, label='Smoothed curve')

# Add performance interval band
ax2.axhspan(2337, 2498, alpha=0.2, color='green', label='Stable interval (2337-2498)')

ax2.text(250000, 3500, 'Optimization strategy:\n• lr: 1e-4→6.25e-5\n• Target network: 8000→2000 steps\n• Buffer: 1M→200k\n• Multi-steps: 3→10',
         fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax2.set_xlabel('Training Steps', fontsize=12)
ax2.set_ylabel('Average Reward', fontsize=12)
ax2.set_title('Rainbow DQN v2 - Stability Optimization\nFinal: 2360.53±45.50 (Variance -73%)',
              fontsize=13, fontweight='bold')
ax2.legend(loc='upper left', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 4500])

# ================================
# Subplot 3: IMPALA v2 Conservative V-trace
# ================================
ax3 = axes[1, 0]
steps, rewards = load_training_data('result_excel/IMPALA.csv')
rewards_smooth = smooth_curve(rewards, weight=0.95)

ax3.plot(steps, rewards, alpha=0.2, color='#95E1D3', linewidth=0.5, label='Raw data')
ax3.plot(steps, rewards_smooth, color='#95E1D3', linewidth=2.5, label='Smoothed curve')

# Mark stable convergence region
ax3.axhspan(1600, 1800, alpha=0.2, color='green', label='Stable interval')

ax3.text(250000, 2500, 'Conservative optimization v2:\n• lr: 5e-5→3e-5\n• V-trace ρ/c: 0.9→0.7\n• Buffer: 50k→30k\n• Sequence length: 20→10',
         fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax3.set_xlabel('Training Steps', fontsize=12)
ax3.set_ylabel('Average Reward', fontsize=12)
ax3.set_title('IMPALA v2 - Conservative V-trace Strategy\nFinal: 1682.19±73.85 (Eliminate crashes)',
              fontsize=13, fontweight='bold')
ax3.legend(loc='upper left', fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_ylim([0, 3500])

# ================================
# Subplot 4: Three Algorithm Comparison
# ================================
ax4 = axes[1, 1]

# A2C v3
steps_a2c, rewards_a2c = load_training_data('result_excel/SB3_A2C.csv')
rewards_a2c_smooth = smooth_curve(rewards_a2c, weight=0.95)
ax4.plot(steps_a2c, rewards_a2c_smooth, color='#FF6B6B', linewidth=2.5,
         label='A2C v3 (4437.86)', marker='o', markersize=3, markevery=10000)

# Rainbow DQN v2
steps_rainbow, rewards_rainbow = load_training_data('result_excel/Rainbow_DQN.csv')
rewards_rainbow_smooth = smooth_curve(rewards_rainbow, weight=0.95)
ax4.plot(steps_rainbow, rewards_rainbow_smooth, color='#4ECDC4', linewidth=2.5,
         label='Rainbow DQN v2 (2360.53)', marker='s', markersize=3, markevery=10000)

# IMPALA v2
steps_impala, rewards_impala = load_training_data('result_excel/IMPALA.csv')
rewards_impala_smooth = smooth_curve(rewards_impala, weight=0.95)
ax4.plot(steps_impala, rewards_impala_smooth, color='#95E1D3', linewidth=2.5,
         label='IMPALA v2 (1682.19)', marker='^', markersize=3, markevery=10000)

# Mark 300k steps boundary
ax4.axvline(x=300000, color='green', linestyle='--', linewidth=1.5, alpha=0.5)
ax4.text(300000, 4500, '← A2C Cosine annealing starts', fontsize=9, color='green')

ax4.set_xlabel('Training Steps', fontsize=12)
ax4.set_ylabel('Average Reward', fontsize=12)
ax4.set_title('Optimized Algorithm Performance Comparison',
              fontsize=13, fontweight='bold')
ax4.legend(loc='lower right', fontsize=11)
ax4.grid(True, alpha=0.3)
ax4.set_ylim([-100, 5000])

plt.tight_layout()
plt.savefig('../../Figures/analysis/optimization_training_curves.png', dpi=300, bbox_inches='tight')
print("✅ Training curve plot saved: optimization_training_curves.png")

# ================================
# Extra: Plot A2C v3 detailed analysis (learning rate change)
# ================================
fig2, (ax_reward, ax_lr) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
fig2.suptitle('A2C v3 Delayed Cosine Annealing Detailed Analysis',
              fontsize=16, fontweight='bold')

# Top: Reward curve
steps, rewards = load_training_data('result_excel/SB3_A2C.csv')
rewards_smooth = smooth_curve(rewards, weight=0.95)

ax_reward.plot(steps, rewards, alpha=0.15, color='gray', linewidth=0.5, label='Raw data')
ax_reward.plot(steps, rewards_smooth, color='#FF6B6B', linewidth=3, label='Smoothed curve')
ax_reward.axvline(x=300000, color='green', linestyle='--', linewidth=2, alpha=0.7)
ax_reward.fill_between([0, 300000], -100, 5000, alpha=0.1, color='orange', label='Fixed lr phase')
ax_reward.fill_between([300000, 500000], -100, 5000, alpha=0.1, color='blue', label='Cosine annealing phase')
ax_reward.set_ylabel('Average Reward', fontsize=12)
ax_reward.set_title('Training Reward Change', fontsize=13)
ax_reward.legend(loc='lower right', fontsize=11)
ax_reward.grid(True, alpha=0.3)
ax_reward.set_ylim([-100, 5000])

# Bottom: Learning rate change
import math

def delayed_cosine_annealing(step, warmup=300000, total=500000, initial=7e-4, minimum=1e-5):
    """Calculate Delayed Cosine Annealing Learning Rate"""
    if step < warmup:
        return initial
    progress = (step - warmup) / (total - warmup)
    cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
    return minimum + (initial - minimum) * cosine_factor

lr_values = [delayed_cosine_annealing(s) for s in steps]

ax_lr.plot(steps, lr_values, color='#4ECDC4', linewidth=3, label='Learning rate schedule')
ax_lr.axvline(x=300000, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Annealing start point')
ax_lr.axhline(y=7e-4, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label='Initial lr (7e-4)')
ax_lr.axhline(y=1e-5, color='blue', linestyle=':', linewidth=1.5, alpha=0.7, label='Final lr (1e-5)')
ax_lr.fill_between([0, 300000], 0, 8e-4, alpha=0.1, color='orange')
ax_lr.fill_between([300000, 500000], 0, 8e-4, alpha=0.1, color='blue')

ax_lr.set_xlabel('Training Steps', fontsize=12)
ax_lr.set_ylabel('Learning Rate', fontsize=12)
ax_lr.set_title('Learning Rate Schedule Strategy', fontsize=13)
ax_lr.legend(loc='upper right', fontsize=11)
ax_lr.grid(True, alpha=0.3)
ax_lr.set_ylim([0, 8e-4])
ax_lr.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))

plt.tight_layout()
plt.savefig('../../Figures/analysis/a2c_v3_detailed_analysis.png', dpi=300, bbox_inches='tight')
print("✅ A2C v3Detailed analysis plot saved: a2c_v3_detailed_analysis.png")

print("\n📊 Plots generated:")
print("  1. optimization_training_curves.png - 四算法Comparison图")
print("  2. a2c_v3_detailed_analysis.png - A2C v3Learning Rate分析图")
