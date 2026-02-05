"""
TD7 Jump Learning Phenomenon Analysis (Based on Latest Training Data)
TD7 Jump Learning Phenomenon Analysis with Latest Training Data
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set font for international characters
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

# Read latest TD7 training data
data = pd.read_csv('result_excel/TD7.csv')

# Create 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# ====== Subplot 1: Complete training curve ======
ax1 = axes[0, 0]
ax1.plot(data['Step'], data['Value'], linewidth=1.5, alpha=0.8, color='#9467bd', label='TD7 Training Curve')

# Mark two key jump points
jump1_step = 25589
jump1_before = 214.9
jump1_after = 1321.1
jump2_step = 26989
jump2_before = 3085.8
jump2_after = 4309.1

ax1.scatter([jump1_step], [jump1_after], color='red', s=200, zorder=5, marker='^',
            label=f'Jump 1: {jump1_before:.0f}→{jump1_after:.0f} (+515%)')
ax1.scatter([jump2_step], [jump2_after], color='orange', s=200, zorder=5, marker='^',
            label=f'Jump 2: {jump2_before:.0f}→{jump2_after:.0f} (+40%)')

ax1.axhline(y=4360.88, color='green', linestyle='--', linewidth=1.5, alpha=0.7,
            label='Final Performance: 4360.88')
ax1.axvline(x=75000, color='red', linestyle='--', linewidth=2, alpha=0.5,
            label='75k steps: Learning rate decay starts')

ax1.set_xlabel('Training Steps', fontsize=12, fontweight='bold')
ax1.set_ylabel('Reward', fontsize=12, fontweight='bold')
ax1.set_title('TD7 Complete Training Curve\nSALE Representation Learning + Staged Learning Rate Schedule (75k step turning point)',
              fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9, loc='lower right')
ax1.set_xlim([0, 110000])

# ====== Subplot 2: First jump zoom (25k steps) ======
ax2 = axes[0, 1]
mask1 = (data['Step'] >= 24000) & (data['Step'] <= 27000)
zoom1_data = data[mask1]
ax2.plot(zoom1_data['Step'], zoom1_data['Value'], linewidth=2.5, alpha=0.9,
         color='#ff7f0e', marker='o', markersize=3)
ax2.axvline(x=jump1_step, color='red', linestyle='--', linewidth=2.5, alpha=0.7)
ax2.axhline(y=jump1_before, color='blue', linestyle=':', linewidth=1.5, alpha=0.5)
ax2.axhline(y=jump1_after, color='red', linestyle=':', linewidth=1.5, alpha=0.5)

# Annotations
ax2.annotate(f'Before Jump\n{jump1_before:.1f}',
             xy=(jump1_step-500, jump1_before),
             xytext=(24500, 500),
             fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
             arrowprops=dict(arrowstyle='->', color='blue', lw=2))
ax2.annotate(f'After Jump\n{jump1_after:.1f}',
             xy=(jump1_step+500, jump1_after),
             xytext=(26000, 1800),
             fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='orange', alpha=0.8),
             arrowprops=dict(arrowstyle='->', color='red', lw=2))

ax2.set_xlabel('Training Steps', fontsize=11, fontweight='bold')
ax2.set_ylabel('Reward', fontsize=11, fontweight='bold')
ax2.set_title(f'First Jump @ {jump1_step} steps\nPerformance Boost 515% (214.9→1321.1)',
              fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xlim([24000, 27000])

# ====== Subplot 3: Second jump zoom (27k steps) ======
ax3 = axes[1, 0]
mask2 = (data['Step'] >= 26000) & (data['Step'] <= 29000)
zoom2_data = data[mask2]
ax3.plot(zoom2_data['Step'], zoom2_data['Value'], linewidth=2.5, alpha=0.9,
         color='#2ca02c', marker='o', markersize=3)
ax3.axvline(x=jump2_step, color='orange', linestyle='--', linewidth=2.5, alpha=0.7)
ax3.axhline(y=jump2_before, color='blue', linestyle=':', linewidth=1.5, alpha=0.5)
ax3.axhline(y=jump2_after, color='orange', linestyle=':', linewidth=1.5, alpha=0.5)

# Annotations
ax3.annotate(f'Before Jump\n{jump2_before:.1f}',
             xy=(jump2_step-500, jump2_before),
             xytext=(26500, 2500),
             fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
             arrowprops=dict(arrowstyle='->', color='blue', lw=2))
ax3.annotate(f'After Jump\n{jump2_after:.1f}',
             xy=(jump2_step+500, jump2_after),
             xytext=(27500, 4500),
             fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
             arrowprops=dict(arrowstyle='->', color='orange', lw=2))

ax3.set_xlabel('Training Steps', fontsize=11, fontweight='bold')
ax3.set_ylabel('Reward', fontsize=11, fontweight='bold')
ax3.set_title(f'Second Jump @ {jump2_step} steps\nPerformance Boost 40% (3085.8→4309.1)',
              fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_xlim([26000, 29000])

# ====== Subplot 4: Jump learning theoretical explanation ======
ax4 = axes[1, 1]
ax4.axis('off')

# Theoretical explanation text
explanation = """
TD7 Jump Learning Phenomenon Theoretical Analysis
Jump Learning Phenomenon Theoretical Analysis

[SALE Representation Learning Critical Threshold]

1. First Jump (25,589 steps: 215→1321, +515%)
   • State embedding space reaches initial understanding threshold
   • SALE encoder begins capturing queue pressure patterns
   • Policy network suddenly identifies high-reward actions

2. Second Jump (26,989 steps: 3086→4309, +40%)
   • Embedding representation fully aligns with optimal policy
   • Learned representation suddenly captures sufficient structure
   • Achieves qualitative leap in optimal policy reasoning

[Jump Mechanism]
- SALE loss minimization forces encoder to learn action-predictable representations
- When "predictability" in embedding dimensions exceeds critical threshold
- Policy gradients propagate rapidly, causing sudden performance jumps

[Statistical Characteristics]
- Jump magnitude: First 515%, Second 40%
- Jump interval: Only 1,400 steps (~20 episodes)
- Subsequent stability: Highly stable in 4300-4475 range

[Synergy with Learning Rate Schedule]
- First 75k steps fixed lr=3e-4: Allows rapid exploration and jumps
- After 75k steps lr decay: Stabilizes convergence, prevents collapse
- Final performance: 4360.88 (top-tier, ranked 3rd)

[Theoretical Significance]
Critical role of representation learning in reinforcement learning:
When embedding space captures sufficient structural information,
agents can achieve qualitative leaps rather than gradual improvements.
"""

ax4.text(0.05, 0.95, explanation, transform=ax4.transAxes,
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6, pad=1),
         family='monospace')

# Overall title
fig.suptitle('TD7 Jump Learning Phenomenon Detailed Analysis\n'
             'SALE Representation Learning Reaches Critical Understanding Threshold | Two Qualitative Performance Leaps',
             fontsize=15, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('../../Figures/analysis/figure7_td7_jump_learning.png', dpi=300, bbox_inches='tight')
print("✅ TD7 jump learning analysis figure updated: figure7_td7_jump_learning.png")

plt.show()
