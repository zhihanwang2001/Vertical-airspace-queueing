#!/usr/bin/env python3
"""
生成Major Revision论文图表
基于n=3, 5× load的结构对比数据
生成日期: 2026-01-08
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# 设置中文字体和样式
mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['figure.titlesize'] = 16

# 数据路径
DATA_DIR = Path("Data/ablation_studies/structural_5x_load")
OUTPUT_DIR = Path("Analysis/figures/major_revision")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_structural_data():
    """加载结构对比数据（n=3, 5× load）"""
    data = {
        'inverted': {'A2C': [], 'PPO': []},
        'normal': {'A2C': [], 'PPO': []}
    }

    structures = ['inverted_pyramid', 'normal_pyramid']
    algorithms = ['A2C', 'PPO']
    seeds = [42, 123, 456]

    for struct in structures:
        struct_key = 'inverted' if 'inverted' in struct else 'normal'
        for algo in algorithms:
            for seed in seeds:
                file_path = DATA_DIR / struct / f"{algo}_seed{seed}_results.json"
                if file_path.exists():
                    with open(file_path) as f:
                        result = json.load(f)
                        # 提取平均奖励
                        mean_reward = result.get('mean_reward', 0)
                        data[struct_key][algo].append(mean_reward)
                else:
                    print(f"Warning: Missing {file_path}")

    return data

def plot_structure_comparison():
    """
    Figure 1: 结构对比图（Major Revision核心图）
    显示inverted vs normal pyramid在5× load下的n=3对比
    """
    data = load_structural_data()

    # 计算统计量
    stats = {}
    for struct in ['inverted', 'normal']:
        stats[struct] = {}
        for algo in ['A2C', 'PPO']:
            rewards = data[struct][algo]
            stats[struct][algo] = {
                'mean': np.mean(rewards),
                'std': np.std(rewards, ddof=1),
                'sem': np.std(rewards, ddof=1) / np.sqrt(len(rewards))
            }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 子图1: 平均奖励对比
    x = np.arange(2)
    width = 0.35

    inverted_means = [stats['inverted']['A2C']['mean'], stats['inverted']['PPO']['mean']]
    normal_means = [stats['normal']['A2C']['mean'], stats['normal']['PPO']['mean']]
    inverted_sems = [stats['inverted']['A2C']['sem'], stats['inverted']['PPO']['sem']]
    normal_sems = [stats['normal']['A2C']['sem'], stats['normal']['PPO']['sem']]

    bars1 = ax1.bar(x - width/2, inverted_means, width, yerr=inverted_sems,
                    label='Inverted [8,6,4,3,2]', color='#4472C4', capsize=5,
                    error_kw={'linewidth': 2})
    bars2 = ax1.bar(x + width/2, normal_means, width, yerr=normal_sems,
                    label='Normal [2,3,4,6,8]', color='#ED7D31', capsize=5,
                    error_kw={'linewidth': 2})

    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax1.set_ylabel('Average Reward', fontweight='bold')
    ax1.set_title('Structure Comparison - Average Reward\n(n=3, 5× Load)', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['A2C', 'PPO'])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(bottom=650000, top=750000)

    # 子图2: 提升百分比
    improvements = []
    for algo in ['A2C', 'PPO']:
        inv_mean = stats['inverted'][algo]['mean']
        norm_mean = stats['normal'][algo]['mean']
        improvement = ((inv_mean - norm_mean) / norm_mean) * 100
        improvements.append(improvement)

    bars3 = ax2.bar(['A2C', 'PPO'], improvements, color=['#70AD47', '#FFC000'], width=0.6)

    for bar, imp in zip(bars3, improvements):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'+{imp:.1f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax2.set_ylabel('Improvement (%)', fontweight='bold')
    ax2.set_title('Inverted vs Normal Pyramid\nPerformance Improvement', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim(0, 12)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

    plt.tight_layout()

    # 保存
    output_file = OUTPUT_DIR / "fig1_structure_comparison_major_revision"
    plt.savefig(f"{output_file}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_file}.png", dpi=300, bbox_inches='tight')
    print(f"✅ Generated: {output_file}.pdf/.png")
    plt.close()

    return stats

def plot_statistical_evidence():
    """
    Figure 2: 统计显著性可视化
    显示Cohen's d效应量和p值
    """
    # 从logs/experiment_a_5x_analysis.log提取的数据
    results = {
        'A2C': {
            'inverted_mean': 723990,
            'normal_mean': 663227,
            'cohens_d': 33.61,
            'p_value': 0.000005,
            'ci_lower': 14.52,
            'ci_upper': 52.69
        },
        'PPO': {
            'inverted_mean': 722401,
            'normal_mean': 659080,
            'cohens_d': 273.60,
            'p_value': 0.000000,
            'ci_lower': 118.79,
            'ci_upper': 428.42
        }
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 子图1: Cohen's d with 95% CI
    algos = ['A2C', 'PPO']
    d_values = [results[a]['cohens_d'] for a in algos]
    ci_lower = [results[a]['ci_lower'] for a in algos]
    ci_upper = [results[a]['ci_upper'] for a in algos]
    errors = [[d - cl for d, cl in zip(d_values, ci_lower)],
              [cu - d for d, cu in zip(d_values, ci_upper)]]

    bars = ax1.barh(algos, d_values, xerr=errors, color=['#4472C4', '#ED7D31'],
                    capsize=8, error_kw={'linewidth': 2.5})

    for i, (algo, d_val) in enumerate(zip(algos, d_values)):
        ax1.text(d_val + 50, i, f"d = {d_val:.1f}",
                va='center', ha='left', fontsize=11, fontweight='bold')

    ax1.set_xlabel("Cohen's d (Effect Size)", fontweight='bold')
    ax1.set_title("Effect Size with 95% Confidence Interval\n(n=3, Welch's t-test)", fontweight='bold')
    ax1.axvline(x=0.8, color='red', linestyle='--', linewidth=2, label='Large effect (d=0.8)')
    ax1.legend()
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.set_xlim(0, 500)

    # 子图2: -log10(p-value)
    p_values = [results[a]['p_value'] for a in algos]
    neg_log_p = [-np.log10(p) for p in p_values]

    bars2 = ax2.bar(algos, neg_log_p, color=['#70AD47', '#FFC000'], width=0.5)

    for bar, val, p in zip(bars2, neg_log_p, p_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'p < 0.001\n-log₁₀(p) = {val:.1f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax2.set_ylabel('-log₁₀(p-value)', fontweight='bold')
    ax2.set_title('Statistical Significance\n(Higher = More Significant)', fontweight='bold')
    ax2.axhline(y=-np.log10(0.001), color='red', linestyle='--', linewidth=2,
                label='p = 0.001')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    output_file = OUTPUT_DIR / "fig2_statistical_evidence_major_revision"
    plt.savefig(f"{output_file}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_file}.png", dpi=300, bbox_inches='tight')
    print(f"✅ Generated: {output_file}.pdf/.png")
    plt.close()

def main():
    print("="*60)
    print("生成Major Revision论文图表")
    print("数据: n=3, 5× load, seeds=[42,123,456]")
    print("="*60)

    # 生成图表
    print("\n[1/2] 生成结构对比图...")
    stats = plot_structure_comparison()

    print("\n[2/2] 生成统计显著性图...")
    plot_statistical_evidence()

    print("\n" + "="*60)
    print("图表生成完成！")
    print(f"输出目录: {OUTPUT_DIR}")
    print("="*60)

    # 打印统计摘要
    print("\n📊 数据摘要:")
    for struct in ['inverted', 'normal']:
        print(f"\n{struct.upper()} PYRAMID:")
        for algo in ['A2C', 'PPO']:
            s = stats[struct][algo]
            print(f"  {algo}: {s['mean']:.1f} ± {s['std']:.1f} (SEM: {s['sem']:.1f})")

if __name__ == "__main__":
    main()
