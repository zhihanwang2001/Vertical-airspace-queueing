"""
美化版性能排名图 (图3)
Beautiful Performance Ranking Figure
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch

# 设置绘图风格
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

# 更新后的算法数据（去掉版本号，SAC-v2保留因为是算法名称）
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

# 精心设计的颜色方案（使用渐变色）
colors = [
    '#D4AF37',  # 金色 - A2C
    '#4169E1',  # 皇家蓝 - PPO
    '#8A2BE2',  # 蓝紫 - TD7
    '#FF6347',  # 番茄红 - R2D2
    '#9370DB',  # 中紫 - SAC-v2
    '#20B2AA',  # 浅海蓝 - TD3
    '#32CD32',  # 酸橙绿 - Heuristic
    '#FF8C00',  # 深橙 - Rainbow DQN
    '#48D1CC',  # 中绿松石 - Priority
    '#66CDAA',  # 中海蓝 - FCFS
    '#5F9EA0',  # 军蓝 - SJF
    '#DC143C',  # 深红 - IMPALA
    '#A9A9A9',  # 深灰 - DDPG
    '#2F4F4F'   # 暗石板灰 - Random
]

# 创建图表
fig, ax = plt.subplots(figsize=(14, 10))

# 绘制水平条形图
y_pos = np.arange(len(algorithms))
bars = ax.barh(y_pos, mean_rewards, xerr=std_rewards,
               color=colors, alpha=0.85, edgecolor='#2C3E50', linewidth=1.8,
               error_kw={'elinewidth': 2.2, 'capsize': 6, 'alpha': 0.75,
                        'ecolor': '#34495E', 'capthick': 2})

# 为冠军添加特殊边框
bars[0].set_edgecolor('#B8860B')
bars[0].set_linewidth(3.5)

# 设置y轴
ax.set_yticks(y_pos)
ax.set_yticklabels(algorithms, fontsize=13, fontweight='medium')
ax.invert_yaxis()

# 设置x轴
ax.set_xlabel('Average Reward (Mean ± Std)', fontsize=15, fontweight='bold', color='#2C3E50')
ax.set_xlim([0, 5200])
ax.grid(axis='x', alpha=0.25, linestyle='--', linewidth=1.2, color='#7F8C8D')

# 标题
ax.set_title('Algorithm Performance Ranking Overview\nVertical Stratified Queue Control for UAV Airspace Management',
             fontsize=17, fontweight='bold', pad=25, color='#2C3E50')

# 添加数值标签
for i, (mean, std, algo) in enumerate(zip(mean_rewards, std_rewards, algorithms)):
    label = f'{mean:.1f}±{std:.1f}'
    x_pos = mean + std + 180

    if i < 5:  # 顶级层
        ax.text(x_pos, i, label, va='center', fontsize=11.5, fontweight='semibold', color='#2C3E50')
    else:
        ax.text(x_pos, i, label, va='center', fontsize=10.5, color='#34495E')

# 添加性能分层区域
ax.axvspan(4200, 5200, alpha=0.08, color='#27AE60')
ax.axvspan(2000, 4200, alpha=0.06, color='#F39C12')
ax.axvspan(0, 2000, alpha=0.04, color='#E74C3C')

# 添加分层分隔线
ax.axvline(x=4200, color='#27AE60', linestyle='--', linewidth=2.5, alpha=0.6)
ax.axvline(x=2000, color='#E67E22', linestyle='--', linewidth=2.5, alpha=0.6)

# 创建图例（使用Patch来正确显示tier区域）
legend_elements = [
    Patch(facecolor='#27AE60', alpha=0.3, label='Top Tier (>4200)'),
    Patch(facecolor='#F39C12', alpha=0.3, label='Mid Tier (2000-4200)'),
    Patch(facecolor='#E74C3C', alpha=0.3, label='Low Tier (<2000)')
]

# 图例
ax.legend(handles=legend_elements, loc='lower right', fontsize=12, framealpha=0.95,
          edgecolor='#2C3E50', fancybox=True, shadow=True)

# 美化边框
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
