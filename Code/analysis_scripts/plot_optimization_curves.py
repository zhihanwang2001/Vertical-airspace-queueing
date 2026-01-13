"""
绘制优化算法训练曲线对比图
Plot Training Curves for Optimized Algorithms
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

# 读取CSV数据
def load_training_data(csv_path):
    """读取训练数据"""
    df = pd.read_csv(csv_path)
    return df['Step'].values, df['Value'].values

# 平滑曲线
def smooth_curve(values, weight=0.9):
    """指数移动平均平滑"""
    smoothed = []
    last = values[0]
    for point in values:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)

# 创建图表
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('优化算法训练曲线对比 (Optimized Algorithms Training Curves)',
             fontsize=16, fontweight='bold')

# ================================
# 子图1: A2C v3 延迟余弦退火 (邪修秘法)
# ================================
ax1 = axes[0, 0]
steps, rewards = load_training_data('result_excel/SB3_A2C.csv')
rewards_smooth = smooth_curve(rewards, weight=0.95)

ax1.plot(steps, rewards, alpha=0.2, color='#FF6B6B', linewidth=0.5, label='原始数据')
ax1.plot(steps, rewards_smooth, color='#FF6B6B', linewidth=2.5, label='平滑曲线')

# 标注300k步分界线（延迟余弦退火启动点）
ax1.axvline(x=300000, color='green', linestyle='--', linewidth=2, alpha=0.7,
            label='余弦退火启动 (300k步)')

# 添加文本注释
ax1.text(150000, 4000, '前300k步:\n固定lr=7e-4\n充分探索',
         fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax1.text(400000, 4000, '后200k步:\n余弦退火至1e-5\n稳定收敛',
         fontsize=10, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

ax1.set_xlabel('训练步数 (Training Steps)', fontsize=12)
ax1.set_ylabel('平均奖励 (Average Reward)', fontsize=12)
ax1.set_title('A2C v3 - 延迟余弦退火 🔥\n最终: 4437.86±128.41 (第1名)',
              fontsize=13, fontweight='bold')
ax1.legend(loc='lower right', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim([-100, 5000])

# ================================
# 子图2: Rainbow DQN v2 稳定性优化
# ================================
ax2 = axes[0, 1]
steps, rewards = load_training_data('result_excel/Rainbow_DQN.csv')
rewards_smooth = smooth_curve(rewards, weight=0.95)

ax2.plot(steps, rewards, alpha=0.2, color='#4ECDC4', linewidth=0.5, label='原始数据')
ax2.plot(steps, rewards_smooth, color='#4ECDC4', linewidth=2.5, label='平滑曲线')

# 添加性能区间带
ax2.axhspan(2337, 2498, alpha=0.2, color='green', label='稳定区间 (2337-2498)')

ax2.text(250000, 3500, '优化策略:\n• lr: 1e-4→6.25e-5\n• 目标网络: 8000→2000步\n• 缓冲区: 1M→200k\n• 多步: 3→10',
         fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax2.set_xlabel('训练步数 (Training Steps)', fontsize=12)
ax2.set_ylabel('平均奖励 (Average Reward)', fontsize=12)
ax2.set_title('Rainbow DQN v2 - 稳定性优化\n最终: 2360.53±45.50 (方差-73%)',
              fontsize=13, fontweight='bold')
ax2.legend(loc='upper left', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 4500])

# ================================
# 子图3: IMPALA v2 保守V-trace
# ================================
ax3 = axes[1, 0]
steps, rewards = load_training_data('result_excel/IMPALA.csv')
rewards_smooth = smooth_curve(rewards, weight=0.95)

ax3.plot(steps, rewards, alpha=0.2, color='#95E1D3', linewidth=0.5, label='原始数据')
ax3.plot(steps, rewards_smooth, color='#95E1D3', linewidth=2.5, label='平滑曲线')

# 标注稳定收敛区域
ax3.axhspan(1600, 1800, alpha=0.2, color='green', label='稳定区间')

ax3.text(250000, 2500, '保守优化 v2:\n• lr: 5e-5→3e-5\n• V-trace ρ/c: 0.9→0.7\n• 缓冲区: 50k→30k\n• 序列长度: 20→10',
         fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax3.set_xlabel('训练步数 (Training Steps)', fontsize=12)
ax3.set_ylabel('平均奖励 (Average Reward)', fontsize=12)
ax3.set_title('IMPALA v2 - 保守V-trace策略\n最终: 1682.19±73.85 (消除崩溃)',
              fontsize=13, fontweight='bold')
ax3.legend(loc='upper left', fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_ylim([0, 3500])

# ================================
# 子图4: 三算法对比
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

# 标注300k步分界线
ax4.axvline(x=300000, color='green', linestyle='--', linewidth=1.5, alpha=0.5)
ax4.text(300000, 4500, '← A2C余弦退火启动', fontsize=9, color='green')

ax4.set_xlabel('训练步数 (Training Steps)', fontsize=12)
ax4.set_ylabel('平均奖励 (Average Reward)', fontsize=12)
ax4.set_title('优化算法性能对比 (Comparison)',
              fontsize=13, fontweight='bold')
ax4.legend(loc='lower right', fontsize=11)
ax4.grid(True, alpha=0.3)
ax4.set_ylim([-100, 5000])

plt.tight_layout()
plt.savefig('../../Figures/analysis/optimization_training_curves.png', dpi=300, bbox_inches='tight')
print("✅ 训练曲线图已保存: optimization_training_curves.png")

# ================================
# 额外：绘制A2C v3详细分析图（学习率变化）
# ================================
fig2, (ax_reward, ax_lr) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
fig2.suptitle('A2C v3 延迟余弦退火详细分析 (Delayed Cosine Annealing Analysis)',
              fontsize=16, fontweight='bold')

# 上图：奖励曲线
steps, rewards = load_training_data('result_excel/SB3_A2C.csv')
rewards_smooth = smooth_curve(rewards, weight=0.95)

ax_reward.plot(steps, rewards, alpha=0.15, color='gray', linewidth=0.5, label='原始数据')
ax_reward.plot(steps, rewards_smooth, color='#FF6B6B', linewidth=3, label='平滑曲线')
ax_reward.axvline(x=300000, color='green', linestyle='--', linewidth=2, alpha=0.7)
ax_reward.fill_between([0, 300000], -100, 5000, alpha=0.1, color='orange', label='固定lr阶段')
ax_reward.fill_between([300000, 500000], -100, 5000, alpha=0.1, color='blue', label='余弦退火阶段')
ax_reward.set_ylabel('平均奖励 (Reward)', fontsize=12)
ax_reward.set_title('训练奖励变化', fontsize=13)
ax_reward.legend(loc='lower right', fontsize=11)
ax_reward.grid(True, alpha=0.3)
ax_reward.set_ylim([-100, 5000])

# 下图：学习率变化
import math

def delayed_cosine_annealing(step, warmup=300000, total=500000, initial=7e-4, minimum=1e-5):
    """计算延迟余弦退火学习率"""
    if step < warmup:
        return initial
    progress = (step - warmup) / (total - warmup)
    cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
    return minimum + (initial - minimum) * cosine_factor

lr_values = [delayed_cosine_annealing(s) for s in steps]

ax_lr.plot(steps, lr_values, color='#4ECDC4', linewidth=3, label='学习率调度')
ax_lr.axvline(x=300000, color='green', linestyle='--', linewidth=2, alpha=0.7, label='退火启动点')
ax_lr.axhline(y=7e-4, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label='初始lr (7e-4)')
ax_lr.axhline(y=1e-5, color='blue', linestyle=':', linewidth=1.5, alpha=0.7, label='最终lr (1e-5)')
ax_lr.fill_between([0, 300000], 0, 8e-4, alpha=0.1, color='orange')
ax_lr.fill_between([300000, 500000], 0, 8e-4, alpha=0.1, color='blue')

ax_lr.set_xlabel('训练步数 (Training Steps)', fontsize=12)
ax_lr.set_ylabel('学习率 (Learning Rate)', fontsize=12)
ax_lr.set_title('学习率调度策略', fontsize=13)
ax_lr.legend(loc='upper right', fontsize=11)
ax_lr.grid(True, alpha=0.3)
ax_lr.set_ylim([0, 8e-4])
ax_lr.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))

plt.tight_layout()
plt.savefig('../../Figures/analysis/a2c_v3_detailed_analysis.png', dpi=300, bbox_inches='tight')
print("✅ A2C v3详细分析图已保存: a2c_v3_detailed_analysis.png")

print("\n📊 图表已生成:")
print("  1. optimization_training_curves.png - 四算法对比图")
print("  2. a2c_v3_detailed_analysis.png - A2C v3学习率分析图")
