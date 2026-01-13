"""
更新性能排名图 (图3) - 加入A2C-v3冠军数据
Update Performance Ranking Figure with A2C-v3 Champion Results
"""

import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

# 更新后的算法数据 (包含A2C-v3，移除原始高方差SAC)
algorithms = [
    'A2C-v3',      # 新冠军！
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
    2360.53,  # Rainbow DQN v2 (优化后)
    2040.04,  # Priority
    2024.75,  # FCFS
    2011.16,  # SJF
    1682.19,  # IMPALA v2 (优化后)
    1490.48,  # DDPG (放弃)
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

# 算法类型分类 (用于颜色编码)
algorithm_types = [
    'A2C-v3 (优化)',      # 金色 - 冠军
    'Policy-Based RL',   # 深蓝
    'Off-Policy RL',     # 紫色
    'Value-Based RL',    # 橙色
    'Off-Policy RL',     # 紫色
    'Off-Policy RL',     # 紫色
    'Traditional',       # 绿色
    'Value-Based RL (优化)',  # 橙色
    'Traditional',       # 绿色
    'Traditional',       # 绿色
    'Traditional',       # 绿色
    'Distributed RL (优化)',  # 红色
    'Off-Policy RL (放弃)',   # 灰色
    'Baseline'           # 黑色
]

# 颜色映射
color_map = {
    'A2C-v3 (优化)': '#FFD700',        # 金色 - 冠军
    'Policy-Based RL': '#1f77b4',      # 深蓝
    'Off-Policy RL': '#9467bd',        # 紫色
    'Value-Based RL': '#ff7f0e',       # 橙色
    'Value-Based RL (优化)': '#ff9f4a',  # 浅橙
    'Traditional': '#2ca02c',          # 绿色
    'Distributed RL (优化)': '#d62728',  # 红色
    'Off-Policy RL (放弃)': '#7f7f7f',   # 灰色
    'Baseline': '#000000'              # 黑色
}

colors = [color_map[t] for t in algorithm_types]

# 创建图表
fig, ax = plt.subplots(figsize=(16, 10))

# 绘制水平条形图 (从高到低排序)
y_pos = np.arange(len(algorithms))
bars = ax.barh(y_pos, mean_rewards, xerr=std_rewards,
               color=colors, alpha=0.8, edgecolor='black', linewidth=1.5,
               error_kw={'elinewidth': 2, 'capsize': 5, 'alpha': 0.7})

# 设置y轴标签
ax.set_yticks(y_pos)
ax.set_yticklabels(algorithms, fontsize=12)
ax.invert_yaxis()  # 最高分在顶部

# 设置x轴
ax.set_xlabel('平均奖励 (Average Reward)', fontsize=14, fontweight='bold')
ax.set_xlim([0, 5000])
ax.grid(axis='x', alpha=0.3, linestyle='--')

# 添加标题
ax.set_title('算法性能排名总览 (Algorithm Performance Ranking)\n更新: A2C-v3延迟余弦退火优化登顶',
             fontsize=16, fontweight='bold', pad=20)

# 在每个条形上添加数值标签
for i, (mean, std) in enumerate(zip(mean_rewards, std_rewards)):
    # 主要数值
    label = f'{mean:.1f}±{std:.1f}'
    x_pos = mean + std + 150

    # 如果是A2C-v3，添加特殊标记
    if i == 0:
        label = f'🔥 {label} 🏆'
        ax.text(x_pos, i, label, va='center', fontsize=11,
                fontweight='bold', color='darkred')
    else:
        ax.text(x_pos, i, label, va='center', fontsize=10)

# 添加性能分层线
ax.axvline(x=4200, color='red', linestyle='--', linewidth=2, alpha=0.5, label='顶级层 (>4200)')
ax.axvline(x=2000, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='中级层 (2000-4000)')

# 添加图例
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#FFD700', edgecolor='black', label='A2C-v3 (延迟余弦退火) 🏆'),
    Patch(facecolor='#1f77b4', edgecolor='black', label='Policy-Based RL'),
    Patch(facecolor='#9467bd', edgecolor='black', label='Off-Policy RL'),
    Patch(facecolor='#ff7f0e', edgecolor='black', label='Value-Based RL'),
    Patch(facecolor='#2ca02c', edgecolor='black', label='Traditional Schedulers'),
    Patch(facecolor='#d62728', edgecolor='black', label='Distributed RL (优化)'),
    Patch(facecolor='#7f7f7f', edgecolor='black', label='DDPG (放弃)'),
    Patch(facecolor='#000000', edgecolor='black', label='Random Baseline')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=11, framealpha=0.9)

plt.tight_layout()
plt.savefig('../../Figures/analysis/figure1_performance_ranking.png', dpi=300, bbox_inches='tight')
print("✅ 性能排名图已更新保存: figure1_performance_ranking.png")

# ================================
# 额外：绘制顶级层算法详细对比图
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

# 设置标签
ax2.set_xticks(x_pos)
ax2.set_xticklabels(top_algorithms, fontsize=13, fontweight='bold')
ax2.set_ylabel('平均奖励 (Average Reward)', fontsize=14, fontweight='bold')
ax2.set_title('顶级层算法详细对比 (Top-Tier Algorithms Comparison)\nA2C-v3 vs PPO vs TD7 vs R2D2 vs SAC-v2',
              fontsize=15, fontweight='bold', pad=20)

# 添加数值标签
for i, (mean, std) in enumerate(zip(top_means, top_stds)):
    label = f'{mean:.1f}\n±{std:.1f}'
    ax2.text(i, mean + std + 50, label, ha='center', va='bottom',
             fontsize=11, fontweight='bold')

# 添加水平参考线
ax2.axhline(y=4400, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label='4400 门槛')
ax2.axhline(y=4300, color='orange', linestyle='--', linewidth=1.5, alpha=0.6, label='4300 门槛')

ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_ylim([4200, 4600])
ax2.legend(fontsize=11)

# 添加训练时间对比注释
train_times = ['5.4 min', '30.8 min', '382.4 min', '115.7 min', '287.0 min']
for i, time in enumerate(train_times):
    ax2.text(i, 4220, f'训练:\n{time}', ha='center', fontsize=9,
             style='italic', color='darkblue')

plt.tight_layout()
plt.savefig('../../Figures/analysis/figure1_top_tier_comparison.png', dpi=300, bbox_inches='tight')
print("✅ 顶级层对比图已保存: figure1_top_tier_comparison.png")

print("\n📊 图表已生成:")
print("  1. figure1_performance_ranking.png - 完整性能排名 (15算法)")
print("  2. figure1_top_tier_comparison.png - 顶级层详细对比 (5算法)")
