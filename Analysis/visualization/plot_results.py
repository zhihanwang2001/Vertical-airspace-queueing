"""
可视化分析脚本 - 生成论文图表
Visualization Script - Generate Paper Figures
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# 设置中文字体 - 修复中文显示问题
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'

# macOS系统可用的中文字体
import platform
if platform.system() == 'Darwin':  # macOS
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'PingFang SC', 'STHeiti']
else:  # Windows/Linux
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']

plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置图表风格
plt.style.use('seaborn-v0_8-darkgrid')

# 数据路径
project_root = Path(__file__).parent.parent.parent
data_file = project_root / 'Data' / 'summary' / 'comprehensive_experiments_data.json'
output_dir = project_root / 'Analysis' / 'figures'
output_dir.mkdir(parents=True, exist_ok=True)

# 读取数据
with open(data_file, 'r') as f:
    data = json.load(f)

experiments = data['experiments']

# 转换为DataFrame
df = pd.DataFrame(experiments)

print(f"加载了 {len(df)} 条实验数据")
print(f"配置类型: {df['config_type'].unique()}")
print(f"算法: {df['algorithm'].unique()}")


# ============================================================================
# 图1: 容量-性能曲线 (核心贡献)
# ============================================================================

def plot_capacity_performance():
    """容量vs性能曲线 - 展示容量悖论和性能cliff"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 准备数据 - 仅A2C和PPO (相同评估协议)
    df_onpolicy = df[df['algorithm'].isin(['A2C', 'PPO'])].copy()

    # 按容量分组
    capacity_stats = df_onpolicy.groupby('total_capacity').agg({
        'mean_reward': 'mean',
        'crash_rate': 'mean'
    }).reset_index()

    capacity_stats = capacity_stats.sort_values('total_capacity')

    # 左图: 平均奖励 vs 容量
    ax1.plot(capacity_stats['total_capacity'], capacity_stats['mean_reward'],
             'o-', linewidth=2, markersize=8, color='steelblue', label='平均奖励')

    # 标注关键点
    max_idx = capacity_stats['mean_reward'].idxmax()
    max_capacity = capacity_stats.loc[max_idx, 'total_capacity']
    max_reward = capacity_stats.loc[max_idx, 'mean_reward']
    ax1.annotate(f'最优: 容量{int(max_capacity)}\n奖励{max_reward:.0f}',
                xy=(max_capacity, max_reward),
                xytext=(max_capacity+3, max_reward+2000),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, color='red', weight='bold')

    # 标注性能cliff
    cliff_data = capacity_stats[capacity_stats['total_capacity'] >= 25]
    if len(cliff_data) >= 2:
        cliff_start = cliff_data.iloc[0]
        cliff_end = cliff_data.iloc[1]
        ax1.annotate(f'性能cliff\n-99.8%',
                    xy=(cliff_end['total_capacity'], cliff_end['mean_reward']),
                    xytext=(cliff_end['total_capacity']-5, cliff_end['mean_reward']+3000),
                    arrowprops=dict(arrowstyle='->', color='darkred', lw=2),
                    fontsize=10, color='darkred', weight='bold')

    ax1.set_xlabel('总容量', fontsize=12, weight='bold')
    ax1.set_ylabel('平均奖励', fontsize=12, weight='bold')
    ax1.set_title('容量-性能关系 (A2C+PPO平均)', fontsize=14, weight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('symlog')  # 对数刻度处理负值

    # 右图: 崩溃率 vs 容量
    ax2.plot(capacity_stats['total_capacity'], capacity_stats['crash_rate']*100,
             'o-', linewidth=2, markersize=8, color='crimson', label='崩溃率')

    # 标注稳定性边界
    boundary = capacity_stats[capacity_stats['crash_rate'] < 1.0]['total_capacity'].max()
    ax2.axvline(x=boundary, color='green', linestyle='--', linewidth=2, alpha=0.7)
    ax2.text(boundary-2, 50, f'稳定性边界\n容量{int(boundary)}',
             fontsize=10, color='green', weight='bold',
             ha='right', va='center')

    ax2.set_xlabel('总容量', fontsize=12, weight='bold')
    ax2.set_ylabel('崩溃率 (%)', fontsize=12, weight='bold')
    ax2.set_title('容量-崩溃率关系', fontsize=14, weight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-5, 105)

    plt.tight_layout()
    plt.savefig(output_dir / 'figure1_capacity_performance.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figure1_capacity_performance.pdf', bbox_inches='tight')
    print(f"✅ 已保存: figure1_capacity_performance.png/pdf")
    plt.close()


# ============================================================================
# 图2: 容量结构对比 (验证结构优势)
# ============================================================================

def plot_structure_comparison():
    """对比倒金字塔、正金字塔、均匀分布 (容量23)"""

    # 筛选容量23的三种结构
    configs_23 = ['inverted_pyramid', 'reverse_pyramid']
    df_23 = df[df['config_name'].isin(configs_23) & df['algorithm'].isin(['A2C', 'PPO'])].copy()

    # 添加均匀25作为对照
    df_uniform = df[(df['config_name'] == 'uniform') & df['algorithm'].isin(['A2C', 'PPO'])].copy()
    df_comparison = pd.concat([df_23, df_uniform])

    # 映射配置名称
    config_labels = {
        'inverted_pyramid': '倒金字塔\n[8,6,4,3,2]',
        'reverse_pyramid': '正金字塔\n[2,3,4,6,8]',
        'uniform': '均匀\n[5,5,5,5,5]'
    }
    df_comparison['config_label'] = df_comparison['config_name'].map(config_labels)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 左图: 平均奖励对比
    pivot_reward = df_comparison.pivot_table(
        values='mean_reward',
        index='config_label',
        columns='algorithm'
    )

    x = np.arange(len(pivot_reward))
    width = 0.35

    ax1.bar(x - width/2, pivot_reward['A2C'], width, label='A2C', color='steelblue', alpha=0.8)
    ax1.bar(x + width/2, pivot_reward['PPO'], width, label='PPO', color='coral', alpha=0.8)

    ax1.set_xlabel('容量结构', fontsize=12, weight='bold')
    ax1.set_ylabel('平均奖励', fontsize=12, weight='bold')
    ax1.set_title('结构对比 - 平均奖励', fontsize=14, weight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(pivot_reward.index, fontsize=10)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')

    # 右图: 崩溃率对比
    pivot_crash = df_comparison.pivot_table(
        values='crash_rate',
        index='config_label',
        columns='algorithm'
    ) * 100

    ax2.bar(x - width/2, pivot_crash['A2C'], width, label='A2C', color='steelblue', alpha=0.8)
    ax2.bar(x + width/2, pivot_crash['PPO'], width, label='PPO', color='coral', alpha=0.8)

    ax2.set_xlabel('容量结构', fontsize=12, weight='bold')
    ax2.set_ylabel('崩溃率 (%)', fontsize=12, weight='bold')
    ax2.set_title('结构对比 - 崩溃率', fontsize=14, weight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(pivot_crash.index, fontsize=10)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'figure2_structure_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figure2_structure_comparison.pdf', bbox_inches='tight')
    print(f"✅ 已保存: figure2_structure_comparison.png/pdf")
    plt.close()


# ============================================================================
# 图3: 算法鲁棒性对比
# ============================================================================

def plot_algorithm_robustness():
    """对比A2C、PPO、TD7在不同容量下的崩溃率"""

    fig, ax = plt.subplots(figsize=(12, 6))

    # 按容量和算法分组
    algo_stats = df.groupby(['total_capacity', 'algorithm'])['crash_rate'].mean().unstack()
    algo_stats = algo_stats.sort_index()

    # 绘制曲线
    markers = {'A2C': 'o', 'PPO': 's', 'TD7': '^'}
    colors = {'A2C': 'steelblue', 'PPO': 'coral', 'TD7': 'forestgreen'}

    for algo in ['A2C', 'PPO', 'TD7']:
        if algo in algo_stats.columns:
            ax.plot(algo_stats.index, algo_stats[algo]*100,
                   marker=markers[algo], linewidth=2, markersize=8,
                   color=colors[algo], label=algo, alpha=0.8)

    # 标注TD7的零崩溃区域
    td7_zero = algo_stats[algo_stats['TD7'] == 0.0].index
    if len(td7_zero) > 0:
        ax.fill_between(td7_zero, 0, 5, color='green', alpha=0.1,
                        label='TD7零崩溃区域')

    ax.set_xlabel('总容量', fontsize=12, weight='bold')
    ax.set_ylabel('崩溃率 (%)', fontsize=12, weight='bold')
    ax.set_title('算法鲁棒性对比 - 崩溃率vs容量', fontsize=14, weight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)

    plt.tight_layout()
    plt.savefig(output_dir / 'figure3_algorithm_robustness.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figure3_algorithm_robustness.pdf', bbox_inches='tight')
    print(f"✅ 已保存: figure3_algorithm_robustness.png/pdf")
    plt.close()


# ============================================================================
# 图4: 算法性能全面对比 (雷达图)
# ============================================================================

def plot_algorithm_radar():
    """算法综合性能雷达图"""

    from matplotlib.patches import Circle, RegularPolygon
    from matplotlib.path import Path
    from matplotlib.projections.polar import PolarAxes
    from matplotlib.projections import register_projection
    from matplotlib.spines import Spine
    from matplotlib.transforms import Affine2D

    # 计算算法平均指标 (仅包含可行配置, capacity <= 25)
    df_viable = df[df['total_capacity'] <= 25].copy()

    algo_metrics = df_viable.groupby('algorithm').agg({
        'mean_reward': 'mean',
        'crash_rate': 'mean',
        'completion_rate': 'mean',
        'mean_episode_length': 'mean'
    }).reset_index()

    # 归一化指标到0-1 (用于雷达图)
    # 奖励: 正向指标, 越高越好
    # 崩溃率: 反向指标, 越低越好 -> 转换为 (1 - crash_rate)
    # 完成率: 正向指标
    # Episode长度: 正向指标 (能维持更长说明更稳定)

    # 注意: TD7的episode长度远大于A2C/PPO (不同协议), 需要分开归一化
    df_onpolicy_viable = df_viable[df_viable['algorithm'].isin(['A2C', 'PPO'])]
    df_td7_viable = df_viable[df_viable['algorithm'] == 'TD7']

    # 仅对比A2C和PPO (相同评估协议)
    algo_metrics_onpolicy = df_onpolicy_viable.groupby('algorithm').agg({
        'mean_reward': 'mean',
        'crash_rate': 'mean',
        'completion_rate': 'mean',
        'mean_episode_length': 'mean'
    }).reset_index()

    # 归一化
    max_reward = algo_metrics_onpolicy['mean_reward'].max()
    max_length = algo_metrics_onpolicy['mean_episode_length'].max()

    algo_metrics_onpolicy['reward_norm'] = algo_metrics_onpolicy['mean_reward'] / max_reward
    algo_metrics_onpolicy['stability_norm'] = 1 - algo_metrics_onpolicy['crash_rate']
    algo_metrics_onpolicy['completion_norm'] = algo_metrics_onpolicy['completion_rate']
    algo_metrics_onpolicy['length_norm'] = algo_metrics_onpolicy['mean_episode_length'] / max_length

    # 雷达图
    categories = ['奖励', '稳定性\n(1-崩溃率)', '完成率', 'Episode长度']
    N = len(categories)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

    colors_radar = {'A2C': 'steelblue', 'PPO': 'coral'}

    for idx, row in algo_metrics_onpolicy.iterrows():
        algo = row['algorithm']
        values = [
            row['reward_norm'],
            row['stability_norm'],
            row['completion_norm'],
            row['length_norm']
        ]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=algo,
               color=colors_radar[algo], alpha=0.7)
        ax.fill(angles, values, alpha=0.15, color=colors_radar[algo])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title('A2C vs PPO 综合性能对比\n(仅包含可行配置, 容量≤25)',
                fontsize=14, weight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / 'figure4_algorithm_radar.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figure4_algorithm_radar.pdf', bbox_inches='tight')
    print(f"✅ 已保存: figure4_algorithm_radar.png/pdf")
    plt.close()


# ============================================================================
# 图5: 详细实验热图
# ============================================================================

def plot_heatmap():
    """实验结果热图 - 配置×算法"""

    # 准备数据 - 崩溃率热图
    df_plot = df.copy()
    df_plot['config_display'] = df_plot['config_type'] + '\n' + df_plot['total_capacity'].astype(str)

    pivot = df_plot.pivot_table(
        values='crash_rate',
        index='config_display',
        columns='algorithm'
    ) * 100

    # 按总容量排序
    capacity_order = df_plot.groupby('config_display')['total_capacity'].first().sort_values()
    pivot = pivot.reindex(capacity_order.index)

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=100)

    # 设置刻度
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, fontsize=11)
    ax.set_yticklabels(pivot.index, fontsize=10)

    # 添加数值标注
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.values[i, j]
            color = 'white' if value > 50 else 'black'
            text = ax.text(j, i, f'{value:.0f}%',
                          ha="center", va="center", color=color, fontsize=10)

    ax.set_title('崩溃率热图 - 配置×算法', fontsize=14, weight='bold', pad=15)
    ax.set_xlabel('算法', fontsize=12, weight='bold')
    ax.set_ylabel('配置 (类型\\n容量)', fontsize=12, weight='bold')

    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('崩溃率 (%)', fontsize=11, weight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'figure5_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figure5_heatmap.pdf', bbox_inches='tight')
    print(f"✅ 已保存: figure5_heatmap.png/pdf")
    plt.close()


# ============================================================================
# 生成所有图表
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("开始生成论文图表...")
    print("="*80 + "\n")

    plot_capacity_performance()
    plot_structure_comparison()
    plot_algorithm_robustness()
    plot_algorithm_radar()
    plot_heatmap()

    print("\n" + "="*80)
    print(f"✅ 所有图表已生成并保存至: {output_dir}")
    print("="*80 + "\n")

    # 列出生成的文件
    figures = list(output_dir.glob('*.png'))
    print("生成的图表文件:")
    for fig in sorted(figures):
        print(f"  - {fig.name}")
