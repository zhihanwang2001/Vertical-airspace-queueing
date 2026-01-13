"""
TD7跳跃学习现象分析图（简化版 - 仅1-2个关键子图）
TD7 Jump Learning Phenomenon Analysis - Simplified Version
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

# 读取最新TD7训练数据
data = pd.read_csv('result_excel/TD7.csv')

# 创建1x2子图（或者单图）
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# ====== 子图1: 完整训练曲线 ======
ax1 = axes[0]
ax1.plot(data['Step'], data['Value'], linewidth=2, alpha=0.9, color='#9467bd', label='TD7 Training Curve')

# 标注两个关键跳跃点
jump1_step = 25589
jump1_before = 214.9
jump1_after = 1321.1
jump2_step = 26989
jump2_before = 3085.8
jump2_after = 4309.1

ax1.scatter([jump1_step], [jump1_after], color='red', s=250, zorder=5, marker='^',
            label=f'Jump 1: {jump1_before:.0f}→{jump1_after:.0f} (+515%)')
ax1.scatter([jump2_step], [jump2_after], color='orange', s=250, zorder=5, marker='^',
            label=f'Jump 2: {jump2_before:.0f}→{jump2_after:.0f} (+40%)')

ax1.axhline(y=4360.88, color='green', linestyle='--', linewidth=2, alpha=0.7,
            label='Final Performance: 4360.88')

ax1.set_xlabel('Training Steps', fontsize=14, fontweight='bold')
ax1.set_ylabel('Reward', fontsize=14, fontweight='bold')
ax1.set_title('TD7 Complete Training Curve with Jump Learning',
              fontsize=15, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(fontsize=11, loc='lower right', framealpha=0.9)
ax1.set_xlim([0, 110000])
ax1.set_ylim([0, 4600])

# ====== 子图2: 第一次跳跃放大 (25k步) ======
ax2 = axes[1]
mask1 = (data['Step'] >= 24000) & (data['Step'] <= 28000)
zoom1_data = data[mask1]
ax2.plot(zoom1_data['Step'], zoom1_data['Value'], linewidth=3, alpha=0.95,
         color='#ff7f0e', marker='o', markersize=4, markevery=5)
ax2.axvline(x=jump1_step, color='red', linestyle='--', linewidth=2.5, alpha=0.8)
ax2.axhline(y=jump1_before, color='blue', linestyle=':', linewidth=2, alpha=0.6, label=f'Before: {jump1_before:.1f}')
ax2.axhline(y=jump1_after, color='red', linestyle=':', linewidth=2, alpha=0.6, label=f'After: {jump1_after:.1f}')

# 标注
ax2.annotate(f'Jump Point\nStep {jump1_step}',
             xy=(jump1_step, (jump1_before + jump1_after)/2),
             xytext=(jump1_step + 800, 2000),
             fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7, edgecolor='red', linewidth=2),
             arrowprops=dict(arrowstyle='->', color='red', lw=2.5, connectionstyle='arc3,rad=0.3'))

ax2.set_xlabel('Training Steps', fontsize=14, fontweight='bold')
ax2.set_ylabel('Reward', fontsize=14, fontweight='bold')
ax2.set_title(f'First Jump @ Step {jump1_step}\nPerformance Boost: +515% (215→1321)',
              fontsize=15, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(fontsize=11, loc='upper left', framealpha=0.9)
ax2.set_xlim([24000, 28000])

# 总标题
fig.suptitle('TD7 Jump Learning Phenomenon: SALE Representation Learning Threshold',
             fontsize=17, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('../../Figures/analysis/figure7_td7_jump_learning.png', dpi=300, bbox_inches='tight')
print("✅ TD7 Jump Learning Figure Updated (Simplified): figure7_td7_jump_learning.png")

plt.show()
