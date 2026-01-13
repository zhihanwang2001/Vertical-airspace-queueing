"""
TD7跳跃学习现象分析图（基于最新训练数据）
TD7 Jump Learning Phenomenon Analysis with Latest Training Data
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

# 读取最新TD7训练数据
data = pd.read_csv('result_excel/TD7.csv')

# 创建2x2子图
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# ====== 子图1: 完整训练曲线 ======
ax1 = axes[0, 0]
ax1.plot(data['Step'], data['Value'], linewidth=1.5, alpha=0.8, color='#9467bd', label='TD7训练曲线')

# 标注两个关键跳跃点
jump1_step = 25589
jump1_before = 214.9
jump1_after = 1321.1
jump2_step = 26989
jump2_before = 3085.8
jump2_after = 4309.1

ax1.scatter([jump1_step], [jump1_after], color='red', s=200, zorder=5, marker='^',
            label=f'跳跃1: {jump1_before:.0f}→{jump1_after:.0f} (+515%)')
ax1.scatter([jump2_step], [jump2_after], color='orange', s=200, zorder=5, marker='^',
            label=f'跳跃2: {jump2_before:.0f}→{jump2_after:.0f} (+40%)')

ax1.axhline(y=4360.88, color='green', linestyle='--', linewidth=1.5, alpha=0.7,
            label='最终性能: 4360.88')
ax1.axvline(x=75000, color='red', linestyle='--', linewidth=2, alpha=0.5,
            label='75k步: 学习率衰减开始')

ax1.set_xlabel('训练步数 (Training Steps)', fontsize=12, fontweight='bold')
ax1.set_ylabel('奖励 (Reward)', fontsize=12, fontweight='bold')
ax1.set_title('TD7完整训练曲线\nSALE表示学习 + 阶段性学习率调度 (75k步转折)',
              fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9, loc='lower right')
ax1.set_xlim([0, 110000])

# ====== 子图2: 第一次跳跃放大 (25k步) ======
ax2 = axes[0, 1]
mask1 = (data['Step'] >= 24000) & (data['Step'] <= 27000)
zoom1_data = data[mask1]
ax2.plot(zoom1_data['Step'], zoom1_data['Value'], linewidth=2.5, alpha=0.9,
         color='#ff7f0e', marker='o', markersize=3)
ax2.axvline(x=jump1_step, color='red', linestyle='--', linewidth=2.5, alpha=0.7)
ax2.axhline(y=jump1_before, color='blue', linestyle=':', linewidth=1.5, alpha=0.5)
ax2.axhline(y=jump1_after, color='red', linestyle=':', linewidth=1.5, alpha=0.5)

# 标注
ax2.annotate(f'跳跃前\n{jump1_before:.1f}',
             xy=(jump1_step-500, jump1_before),
             xytext=(24500, 500),
             fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
             arrowprops=dict(arrowstyle='->', color='blue', lw=2))
ax2.annotate(f'跳跃后\n{jump1_after:.1f}',
             xy=(jump1_step+500, jump1_after),
             xytext=(26000, 1800),
             fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='orange', alpha=0.8),
             arrowprops=dict(arrowstyle='->', color='red', lw=2))

ax2.set_xlabel('训练步数 (Training Steps)', fontsize=11, fontweight='bold')
ax2.set_ylabel('奖励 (Reward)', fontsize=11, fontweight='bold')
ax2.set_title(f'第一次跳跃 @ {jump1_step}步\n性能提升515% (214.9→1321.1)',
              fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xlim([24000, 27000])

# ====== 子图3: 第二次跳跃放大 (27k步) ======
ax3 = axes[1, 0]
mask2 = (data['Step'] >= 26000) & (data['Step'] <= 29000)
zoom2_data = data[mask2]
ax3.plot(zoom2_data['Step'], zoom2_data['Value'], linewidth=2.5, alpha=0.9,
         color='#2ca02c', marker='o', markersize=3)
ax3.axvline(x=jump2_step, color='orange', linestyle='--', linewidth=2.5, alpha=0.7)
ax3.axhline(y=jump2_before, color='blue', linestyle=':', linewidth=1.5, alpha=0.5)
ax3.axhline(y=jump2_after, color='orange', linestyle=':', linewidth=1.5, alpha=0.5)

# 标注
ax3.annotate(f'跳跃前\n{jump2_before:.1f}',
             xy=(jump2_step-500, jump2_before),
             xytext=(26500, 2500),
             fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
             arrowprops=dict(arrowstyle='->', color='blue', lw=2))
ax3.annotate(f'跳跃后\n{jump2_after:.1f}',
             xy=(jump2_step+500, jump2_after),
             xytext=(27500, 4500),
             fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
             arrowprops=dict(arrowstyle='->', color='orange', lw=2))

ax3.set_xlabel('训练步数 (Training Steps)', fontsize=11, fontweight='bold')
ax3.set_ylabel('奖励 (Reward)', fontsize=11, fontweight='bold')
ax3.set_title(f'第二次跳跃 @ {jump2_step}步\n性能提升40% (3085.8→4309.1)',
              fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_xlim([26000, 29000])

# ====== 子图4: 跳跃学习理论解释 ======
ax4 = axes[1, 1]
ax4.axis('off')

# 理论解释文本
explanation = """
TD7 跳跃学习现象理论解释
Jump Learning Phenomenon Theoretical Analysis

【SALE表示学习的关键阈值】

1. 第一次跳跃 (25,589步: 215→1321, +515%)
   • 状态嵌入空间达到初步理解阈值
   • SALE编码器开始捕获队列压力模式
   • 策略网络突然能够识别高奖励动作

2. 第二次跳跃 (26,989步: 3086→4309, +40%)
   • 嵌入表示与最优策略完全对齐
   • 学习的表示突然捕获足够结构
   • 实现最优策略推理的质变

【跳跃机制】
- SALE损失最小化迫使编码器学习可预测动作的表示
- 当嵌入维度中的"可预测性"超过临界阈值时
- 策略梯度能够快速传播，导致突然的性能跃升

【统计特征】
- 跳跃幅度: 第一次 515%, 第二次 40%
- 跳跃间隔: 仅1,400步 (~20 episodes)
- 后续稳定: 4300-4475区间高度稳定

【与学习率调度的协同】
- 前75k步固定lr=3e-4: 允许快速探索和跳跃
- 75k步后学习率衰减: 稳定收敛，防止崩溃
- 最终性能: 4360.88 (顶级层，排名第3)

【理论意义】
表示学习在强化学习中的关键作用：
当嵌入空间捕获足够的结构信息时，
智能体可以实现质的飞跃，而非渐进式改进。
"""

ax4.text(0.05, 0.95, explanation, transform=ax4.transAxes,
         fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6, pad=1),
         family='monospace')

# 总标题
fig.suptitle('TD7 跳跃学习现象详细分析 (TD7 Jump Learning Phenomenon)\n'
             'SALE表示学习达到关键理解阈值 | 两次质变式性能跃升',
             fontsize=15, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('../../Figures/analysis/figure7_td7_jump_learning.png', dpi=300, bbox_inches='tight')
print("✅ TD7跳跃学习分析图已更新: figure7_td7_jump_learning.png")

plt.show()
