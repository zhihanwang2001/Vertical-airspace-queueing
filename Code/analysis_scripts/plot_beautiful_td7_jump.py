"""
Beautiful TD7 Jump Learning Phenomenon Figure
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

# Read TD7 training data
data = pd.read_csv('result_excel/TD7.csv')

# Jump point data (based on actual CSV data)
jump1_step = 25589
jump1_before = 138.05  # Actual value at step 24760
jump1_after = 1321.09
jump2_step = 26989
jump2_before = 2208.82  # Actual value at step 25989
jump2_after = 4309.09

# ====================================
# Figure 1: Complete training curve (single large beautiful plot)
# ====================================
fig1, ax1 = plt.subplots(figsize=(16, 7))

# Draw main curve
ax1.plot(data['Step'], data['Value'], linewidth=2.5, alpha=0.9,
         color='#8A2BE2', label='TD7 Training Curve', zorder=2)

# Mark two jump points (adjusted to smaller triangle markers)
ax1.scatter([jump1_step], [jump1_after], color='#FF4500', s=180, zorder=10,
            marker='^', edgecolor='#8B0000', linewidth=2,
            label=f'Jump 1: {jump1_before:.0f}→{jump1_after:.0f} (+857%)')
ax1.scatter([jump2_step], [jump2_after], color='#FFD700', s=180, zorder=10,
            marker='^', edgecolor='#B8860B', linewidth=2,
            label=f'Jump 2: {jump2_before:.0f}→{jump2_after:.0f} (+95%)')

# Add jump region highlight
ax1.axvspan(24000, 28000, alpha=0.12, color='#FF6347', zorder=1)

# Final performance line (average of last 200 episodes)
ax1.axhline(y=4351.84, color='#32CD32', linestyle='--', linewidth=2.5,
            alpha=0.75, label='Final Performance: 4351.84', zorder=3)

# Add text annotations
ax1.annotate('SALE Representation\nLearning Threshold',
             xy=(jump1_step, jump1_after), xytext=(jump1_step + 5000, 2500),
             fontsize=13, fontweight='bold', color='#8B0000',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#FFE4E1',
                      edgecolor='#FF4500', linewidth=2.5, alpha=0.95),
             arrowprops=dict(arrowstyle='->', lw=3, color='#FF4500',
                           connectionstyle='arc3,rad=0.3'))

ax1.annotate('Policy Optimization\nConvergence',
             xy=(jump2_step, jump2_after), xytext=(jump2_step + 8000, 3400),
             fontsize=13, fontweight='bold', color='#B8860B',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#FFFACD',
                      edgecolor='#FFD700', linewidth=2.5, alpha=0.95),
             arrowprops=dict(arrowstyle='->', lw=3, color='#FFD700',
                           connectionstyle='arc3,rad=-0.2'))

# Axis labels and title
ax1.set_xlabel('Training Steps', fontsize=16, fontweight='bold', color='#2C3E50')
ax1.set_ylabel('Reward', fontsize=16, fontweight='bold', color='#2C3E50')
ax1.set_title('TD7 Jump Learning Phenomenon: SALE Representation Learning Achieves Critical Threshold\n'
              'Two Qualitative Performance Jumps Drive Algorithm to Top-Tier Performance',
              fontsize=18, fontweight='bold', pad=25, color='#2C3E50')

# Grid and legend
ax1.grid(True, alpha=0.25, linestyle='--', linewidth=1.2, color='#7F8C8D')
ax1.legend(fontsize=13, loc='lower right', framealpha=0.95,
          edgecolor='#2C3E50', fancybox=True, shadow=True)

# Set ranges
ax1.set_xlim([0, 110000])
ax1.set_ylim([0, 4800])

# Beautify borders
for spine in ['top', 'right']:
    ax1.spines[spine].set_visible(False)
ax1.spines['left'].set_linewidth(2)
ax1.spines['left'].set_color('#2C3E50')
ax1.spines['bottom'].set_linewidth(2)
ax1.spines['bottom'].set_color('#2C3E50')

plt.tight_layout()
plt.savefig('../../Figures/analysis/figure7_td7_jump_learning.png', dpi=400, bbox_inches='tight', facecolor='white')
print("✅ Beautiful TD7 Jump Learning Figure Saved: figure7_td7_jump_learning.png")
plt.show()


# ====================================
# Figure 2: Jump zoom plot (optional, save separately)
# ====================================
fig2, ax2 = plt.subplots(figsize=(14, 7))

# Filter zoom region data
mask = (data['Step'] >= 24000) & (data['Step'] <= 29000)
zoom_data = data[mask]

# Draw zoom curve
ax2.plot(zoom_data['Step'], zoom_data['Value'], linewidth=3.5, alpha=0.95,
         color='#FF8C00', marker='o', markersize=5, markevery=3,
         markeredgecolor='#8B4513', markeredgewidth=1.5, label='TD7 Training (Zoom)')

# Jump point vertical lines
ax2.axvline(x=jump1_step, color='#FF4500', linestyle='--', linewidth=3, alpha=0.8,
            label=f'Jump 1 @ {jump1_step} steps')
ax2.axvline(x=jump2_step, color='#FFD700', linestyle='--', linewidth=3, alpha=0.8,
            label=f'Jump 2 @ {jump2_step} steps')

# Horizontal reference lines
ax2.axhline(y=jump1_before, color='#4682B4', linestyle=':', linewidth=2.5, alpha=0.7)
ax2.axhline(y=jump1_after, color='#FF6347', linestyle=':', linewidth=2.5, alpha=0.7)
ax2.axhline(y=jump2_after, color='#32CD32', linestyle=':', linewidth=2.5, alpha=0.7)

# Annotate performance gains
ax2.annotate(f'857% Boost\n{jump1_before:.0f}→{jump1_after:.0f}',
             xy=(jump1_step, (jump1_before + jump1_after)/2),
             xytext=(jump1_step - 1200, 2200),
             fontsize=12, fontweight='bold', color='#FF4500',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFE4E1',
                      edgecolor='#FF4500', linewidth=2, alpha=0.9),
             arrowprops=dict(arrowstyle='->', lw=2.5, color='#FF4500'))

ax2.annotate(f'95% Boost\n{jump2_before:.0f}→{jump2_after:.0f}',
             xy=(jump2_step, (jump2_before + jump2_after)/2),
             xytext=(jump2_step + 700, 3600),
             fontsize=12, fontweight='bold', color='#FFD700',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFFACD',
                      edgecolor='#DAA520', linewidth=2, alpha=0.9),
             arrowprops=dict(arrowstyle='->', lw=2.5, color='#FFD700'))

# Axis labels and title
ax2.set_xlabel('Training Steps', fontsize=15, fontweight='bold', color='#2C3E50')
ax2.set_ylabel('Reward', fontsize=15, fontweight='bold', color='#2C3E50')
ax2.set_title('TD7 Jump Learning: Detailed View of Critical Performance Transitions\n'
              'Two Sequential Jumps within 1,400 Steps (~20 Episodes)',
              fontsize=17, fontweight='bold', pad=20, color='#2C3E50')

# Grid and legend
ax2.grid(True, alpha=0.25, linestyle='--', linewidth=1.2, color='#7F8C8D')
ax2.legend(fontsize=12, loc='upper left', framealpha=0.95,
          edgecolor='#2C3E50', fancybox=True, shadow=True)

# 设置范围
ax2.set_xlim([24000, 29000])
ax2.set_ylim([0, 4600])

# 美化边框
for spine in ['top', 'right']:
    ax2.spines[spine].set_visible(False)
ax2.spines['left'].set_linewidth(2)
ax2.spines['left'].set_color('#2C3E50')
ax2.spines['bottom'].set_linewidth(2)
ax2.spines['bottom'].set_color('#2C3E50')

plt.tight_layout()
plt.savefig('../../Figures/analysis/figure7_td7_jump_learning_zoom.png', dpi=400, bbox_inches='tight', facecolor='white')
print("✅ TD7 Jump Learning Zoom Figure Saved: figure7_td7_jump_learning_zoom.png")
plt.show()
