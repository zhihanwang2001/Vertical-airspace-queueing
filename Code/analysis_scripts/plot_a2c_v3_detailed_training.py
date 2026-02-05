"""
Plot A2C-v3 Detailed Training Curves with 300k Delayed Cosine Annealing
Plot A2C-v3 Detailed Training Curves with 300k Delayed Cosine Annealing
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set Chinese font
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

# Read A2C-v3 training data
data = pd.read_csv('result_excel/SB3_A2C.csv')

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# ====== Subplot 1: Complete training curve ======
ax1 = axes[0, 0]
ax1.plot(data['Step'], data['Value'], linewidth=1.5, alpha=0.8, color='#1f77b4')
ax1.axvline(x=300000, color='red', linestyle='--', linewidth=2, alpha=0.7, label='300k步: Learning rate annealing starts')
ax1.axhline(y=4437.86, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Final evaluation: 4437.86±128.41')
ax1.set_xlabel('Training Steps (Training Steps)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Reward (Reward)', fontsize=12, fontweight='bold')
ax1.set_title('A2C-v3 Complete Training Curve\nDelayed Cosine Annealing Learning Rate Schedule (300kstep turning point)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10, loc='lower right')
ax1.set_xlim([0, 500000])

# Add phase annotations
ax1.text(50000, 4000, 'Phase1\n0-100k\nFast Learning', fontsize=9, ha='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax1.text(150000, 4000, 'Phase2\n100k-200k\nPolicy Refinement', fontsize=9, ha='center',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
ax1.text(250000, 4000, 'Phase3\n200k-300k\nPerformance Jump', fontsize=9, ha='center',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
ax1.text(350000, 4000, 'Phase4\n300k-400k\nAnnealing Stabilization', fontsize=9, ha='center',
         bbox=dict(boxstyle='round', facecolor='orange', alpha=0.5))
ax1.text(450000, 4000, 'Phase5\n400k-500k\nDeep Annealing', fontsize=9, ha='center',
         bbox=dict(boxstyle='round', facecolor='pink', alpha=0.5))

# ====== 子图2: 300kstep turning point放大 ======
ax2 = axes[0, 1]
mask = (data['Step'] >= 250000) & (data['Step'] <= 350000)
zoomed_data = data[mask]
ax2.plot(zoomed_data['Step'], zoomed_data['Value'], linewidth=2, alpha=0.8, color='#ff7f0e')
ax2.axvline(x=300000, color='red', linestyle='--', linewidth=2.5, label='300k步: Cosine annealing starts')
ax2.fill_between([250000, 300000], 0, 5000, alpha=0.2, color='blue', label='Fixed lr=7e-4')
ax2.fill_between([300000, 350000], 0, 5000, alpha=0.2, color='orange', label='Cosine annealing 7e-4→1e-5')
ax2.set_xlabel('Training Steps (Training Steps)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Reward (Reward)', fontsize=12, fontweight='bold')
ax2.set_title('300kstep turning point放大图\nLearning rate schedule transition moment', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)
ax2.set_xlim([250000, 350000])

# Mark key points
pre_300k_mean = zoomed_data[zoomed_data['Step'] < 300000]['Value'].mean()
post_300k_mean = zoomed_data[zoomed_data['Step'] >= 300000]['Value'].mean()
ax2.text(275000, pre_300k_mean + 100, f'300kPre mean:\n{pre_300k_mean:.0f}',
         fontsize=9, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
ax2.text(325000, post_300k_mean - 100, f'300kPost mean:\n{post_300k_mean:.0f}',
         fontsize=9, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# ====== Subplot 3: Segmented variance analysis ======
ax3 = axes[1, 0]
segments = [
    (0, 100000, 'Phase1\n0-100k'),
    (100000, 200000, 'Phase2\n100k-200k'),
    (200000, 300000, 'Phase3\n200k-300k'),
    (300000, 400000, 'Phase4\n300k-400k\n(退火)'),
    (400000, 500000, 'Phase5\n400k-500k\n(Deep Annealing)')
]

means = []
stds = []
labels = []
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for start, end, label in segments:
    mask = (data['Step'] >= start) & (data['Step'] < end)
    segment_data = data[mask]['Value']
    means.append(segment_data.mean())
    stds.append(segment_data.std())
    labels.append(label)

x_pos = np.arange(len(labels))
bars = ax3.bar(x_pos, means, yerr=stds, capsize=8, alpha=0.8, color=colors,
               edgecolor='black', linewidth=1.5)

# 添加数值标签
for i, (mean, std) in enumerate(zip(means, stds)):
    ax3.text(i, mean + std + 200, f'{mean:.0f}\n±{std:.0f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

ax3.set_xticks(x_pos)
ax3.set_xticklabels(labels, fontsize=10)
ax3.set_ylabel('平均Reward (Mean Reward)', fontsize=12, fontweight='bold')
ax3.set_title('Segmented Performance Statistics\nVariance Reduction Verification', fontsize=14, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)
ax3.axhline(y=4437.86, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Final evaluation')
ax3.legend(fontsize=10)

# ====== Subplot 4: Rolling window variance change ======
ax4 = axes[1, 1]
window_size = 20
rolling_mean = data['Value'].rolling(window=window_size).mean()
rolling_std = data['Value'].rolling(window=window_size).std()

# 绘制滑动Standard Deviation
ax4.plot(data['Step'], rolling_std, linewidth=2, color='#d62728', label='20point rolling standard deviation')
ax4.axvline(x=300000, color='red', linestyle='--', linewidth=2, alpha=0.7, label='300k步: Annealing starts')
ax4.fill_between([0, 300000], 0, 1000, alpha=0.15, color='blue', label='Fixed lr区域')
ax4.fill_between([300000, 500000], 0, 1000, alpha=0.15, color='orange', label='Annealing region')

ax4.set_xlabel('Training Steps (Training Steps)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Standard Deviation (Standard Deviation)', fontsize=12, fontweight='bold')
ax4.set_title('Training Stability Analysis\nRolling Window Standard Deviation Change', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=10)
ax4.set_xlim([0, 500000])

# Mark key observations
pre_std = rolling_std[(data['Step'] >= 280000) & (data['Step'] < 300000)].mean()
post_std = rolling_std[(data['Step'] >= 300000) & (data['Step'] < 320000)].mean()
ax4.text(150000, 800, f'300k前\nStandard Deviation: {pre_std:.1f}', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
ax4.text(400000, 800, f'300k后\nStandard Deviation: {post_std:.1f}\nReduced{(1-post_std/pre_std)*100:.1f}%', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Overall title
fig.suptitle('A2C-v3 延迟Cosine annealing训练详细分析\n🏆 Champion Algorithm: 4437.86±128.41 | Training Time: 6.9minutes | Efficiency Improvement71times',
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('../../Figures/analysis/a2c_v3_detailed_training_curves.png', dpi=300, bbox_inches='tight')
print("✅ A2C-v3Detailed training curve plot saved: a2c_v3_detailed_training_curves.png")

plt.show()
