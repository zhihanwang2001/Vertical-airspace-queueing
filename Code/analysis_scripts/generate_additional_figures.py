#!/usr/bin/env python3
"""
Generate Additional Figures for Major Revision Paper
Based on Table 4 (Capacity Paradox) and Table 7 (Algorithm Robustness)
Generation date: 2026-01-08
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# Set style
mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 14
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['figure.titlesize'] = 16

# Output directory
OUTPUT_DIR = Path("Analysis/figures/major_revision")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def plot_capacity_paradox():
    """
    Figure 3: Capacity Paradox Visualization
    Based on Table 4 data
    """
    # Table 4 data: Capacity Configuration Performance Ranking
    configs = ['Low-10', 'Unif-20', 'Inv-Pyr\n(K=23)', 'Unif-25', 'Norm-Pyr\n(K=23)', 'Unif-30', 'High-40']
    capacities = [10, 20, 23, 25, 23, 30, 40]
    rewards = [11180, 10855, 8844, 7817, 3950, 13, -32]
    crash_rates = [0, 10, 29, 35, 65, 100, 100]

    # For continuous curve plotting, sort by capacity (handle two K=23 configs)
    capacity_plot = [10, 20, 23, 25, 30, 40]
    reward_plot = [11180, 10855, 8844, 7817, 13, -32]  # Use Inv-Pyr for K=23
    crash_plot = [0, 10, 29, 35, 100, 100]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: Average reward vs capacity (showing inverted-U curve)
    ax1.plot(capacity_plot, reward_plot, 'o-', linewidth=2.5, markersize=10,
             color='#4472C4', label='Average Reward')

    # Mark key points
    ax1.plot(10, 11180, 'o', markersize=15, color='#70AD47', label='Optimal (K=10)')
    ax1.plot(30, 13, 'x', markersize=15, markeredgewidth=3, color='#C00000', label='Collapse (K≥30)')

    # Add value labels
    for cap, rew in zip(capacity_plot, reward_plot):
        if rew > 100:  # Only label non-collapse points
            ax1.text(cap, rew + 500, f'{rew:,}', ha='center', fontsize=9, fontweight='bold')

    # Add critical threshold annotation
    ax1.axvline(x=25, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Critical Threshold')
    ax1.text(25, 9000, 'Critical\nThreshold', ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3))

    # Add 99.8% cliff annotation
    ax1.annotate('', xy=(30, 13), xytext=(25, 7817),
                arrowprops=dict(arrowstyle='->', color='red', lw=2.5))
    ax1.text(27.5, 4000, '99.8%\nDrop', ha='center', fontsize=10, color='red', fontweight='bold')

    ax1.set_xlabel('Total Capacity K', fontweight='bold')
    ax1.set_ylabel('Average Reward', fontweight='bold')
    ax1.set_title('Capacity Paradox: Inverted-U Performance Curve', fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(-2000, 12000)
    ax1.set_xticks(capacity_plot)

    # Subplot 2: Crash rate vs capacity
    bars = ax2.bar(capacity_plot, crash_plot, color=['#70AD47', '#92D050', '#FFC000',
                                                       '#ED7D31', '#C00000', '#C00000'],
                   width=3, edgecolor='black', linewidth=1.5)

    # Add percentage labels
    for cap, crash in zip(capacity_plot, crash_plot):
        ax2.text(cap, crash + 3, f'{crash}%', ha='center', fontsize=10, fontweight='bold')

    # Mark viable vs collapse regions
    ax2.axhline(y=100, color='red', linestyle='-', linewidth=2, alpha=0.5)
    ax2.fill_between([9, 26], 0, 100, alpha=0.15, color='green', label='Viable (K≤25)')
    ax2.fill_between([28, 41], 0, 105, alpha=0.15, color='red', label='Collapse (K≥30)')

    ax2.set_xlabel('Total Capacity K', fontweight='bold')
    ax2.set_ylabel('Crash Rate (%)', fontweight='bold')
    ax2.set_title('Training Stability: Collapse at K≥30', fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim(0, 110)
    ax2.set_xticks(capacity_plot)

    plt.tight_layout()

    # Save
    output_file = OUTPUT_DIR / "fig3_capacity_paradox"
    plt.savefig(f"{output_file}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_file}.png", dpi=300, bbox_inches='tight')
    print(f"Generated: {output_file}.pdf/.png")
    plt.close()


def plot_algorithm_robustness():
    """
    Figure 4: Algorithm Robustness Comparison
    Based on Table 7 data
    """
    # Table 7 data: Algorithm Robustness Comparison (Viable Configurations K≤25)
    configs = ['Low-10', 'Unif-20', 'Inv-Pyr\n(K=23)', 'Unif-25', 'Norm-Pyr\n(K=23)']

    # A2C data
    a2c_rewards = [11264, 11063, 9864, 8623, 5326]
    a2c_crash = [0, 10, 20, 20, 50]

    # PPO data
    ppo_rewards = [11097, 10648, 7825, 7012, 2574]
    ppo_crash = [0, 10, 38, 50, 80]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: Average reward comparison
    x = np.arange(len(configs))
    width = 0.35

    bars1 = ax1.bar(x - width/2, a2c_rewards, width, label='A2C',
                    color='#4472C4', edgecolor='black', linewidth=1)
    bars2 = ax1.bar(x + width/2, ppo_rewards, width, label='PPO',
                    color='#ED7D31', edgecolor='black', linewidth=1)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 200,
                    f'{int(height):,}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax1.set_xlabel('Configuration', fontweight='bold')
    ax1.set_ylabel('Average Reward', fontweight='bold')
    ax1.set_title('Algorithm Performance across Viable Configurations', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, 12500)

    # 子图2: 崩溃率对比
    x = np.arange(len(configs))

    bars3 = ax2.bar(x - width/2, a2c_crash, width, label='A2C',
                    color='#70AD47', edgecolor='black', linewidth=1)
    bars4 = ax2.bar(x + width/2, ppo_crash, width, label='PPO',
                    color='#FFC000', edgecolor='black', linewidth=1)

    # 添加百分比标签
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{int(height)}%',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2.set_xlabel('Configuration', fontweight='bold')
    ax2.set_ylabel('Crash Rate (%)', fontweight='bold')
    ax2.set_title('Training Stability Comparison', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim(0, 90)

    plt.tight_layout()

    # 保存
    output_file = OUTPUT_DIR / "fig4_algorithm_robustness"
    plt.savefig(f"{output_file}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_file}.png", dpi=300, bbox_inches='tight')
    print(f"✅ Generated: {output_file}.pdf/.png")
    plt.close()


def main():
    print("="*60)
    print("生成Major Revision额外图表")
    print("基于Table 4和Table 7数据")
    print("="*60)

    print("\n[1/2] 生成容量悖论图...")
    plot_capacity_paradox()

    print("\n[2/2] 生成算法鲁棒性图...")
    plot_algorithm_robustness()

    print("\n" + "="*60)
    print("图表生成完成！")
    print(f"输出目录: {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
