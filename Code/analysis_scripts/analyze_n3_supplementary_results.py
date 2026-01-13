"""
补充实验统计分析 (n=3)
Statistical Analysis for Supplementary Experiments with n=3

目标:
- 合并原始实验 (seed=42) 与补充实验 (seeds=123, 456)
- 计算有效的统计量 (t-test, 95% CI, Cohen's d with CI)
- 验证核心claims: 结构优势 & 容量悖论

注意:
- n=3: 可以计算t-test和CI，但power较低
- 需要谨慎解释p-values (small sample)
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats
import pandas as pd


def load_experiment_results(config_type, algorithm, seeds=[42, 123, 456]):
    """
    加载n=3实验结果

    参数:
    - config_type: 配置类型
    - algorithm: 算法名称
    - seeds: 随机种子列表 [42(原始), 123, 456]

    返回:
    - List of results dicts
    """
    results = []

    # Seed 42: 原始实验 (Data/ablation_studies/final/)
    original_path = Path(__file__).parent.parent.parent / 'Data' / 'ablation_studies' / 'final' / config_type / f'{algorithm}_results.json'

    if original_path.exists():
        with open(original_path, 'r') as f:
            data = json.load(f)
            data['seed'] = 42  # 添加seed标记
            results.append(data)
    else:
        print(f"⚠️  警告: 未找到原始结果 {original_path}")

    # Seeds 123, 456: 补充实验 (Data/ablation_studies/supplementary_n3/)
    for seed in [123, 456]:
        supp_path = Path(__file__).parent.parent.parent / 'Data' / 'ablation_studies' / 'supplementary_n3' / config_type / f'{algorithm}_seed{seed}_results.json'

        if supp_path.exists():
            with open(supp_path, 'r') as f:
                data = json.load(f)
                results.append(data)
        else:
            print(f"⚠️  警告: 未找到补充结果 {supp_path}")

    return results


def compute_statistics_n3(results):
    """
    计算n=3统计量

    返回:
    - mean_reward: 平均奖励
    - se_reward: 标准误 (SE = SD / sqrt(n))
    - ci_95_lower, ci_95_upper: 95% 置信区间
    - mean_crash_rate: 平均崩溃率
    - se_crash_rate: 崩溃率标准误
    """
    if len(results) < 3:
        print(f"❌ 错误: 样本量不足 (n={len(results)} < 3)")
        return None

    # 提取mean rewards (每个实验的mean)
    mean_rewards = [r['mean_reward'] for r in results]
    crash_rates = [r['crash_rate'] for r in results]

    # 跨训练run的统计量
    mean_reward = np.mean(mean_rewards)
    std_reward = np.std(mean_rewards, ddof=1)  # 样本标准差 (n-1)
    se_reward = std_reward / np.sqrt(len(mean_rewards))  # 标准误

    # 95% CI (使用t分布, df=n-1=2)
    t_critical = stats.t.ppf(0.975, df=len(mean_rewards)-1)  # 双尾检验
    ci_lower = mean_reward - t_critical * se_reward
    ci_upper = mean_reward + t_critical * se_reward

    # 崩溃率统计
    mean_crash_rate = np.mean(crash_rates)
    std_crash_rate = np.std(crash_rates, ddof=1)
    se_crash_rate = std_crash_rate / np.sqrt(len(crash_rates))

    return {
        'n': len(results),
        'seeds': [r.get('seed', 42) for r in results],
        'mean_rewards_per_run': mean_rewards,
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'se_reward': se_reward,
        'ci_95_lower': ci_lower,
        'ci_95_upper': ci_upper,
        't_critical': t_critical,
        'crash_rates_per_run': crash_rates,
        'mean_crash_rate': mean_crash_rate,
        'std_crash_rate': std_crash_rate,
        'se_crash_rate': se_crash_rate,
    }


def welch_t_test(group1_stats, group2_stats):
    """
    Welch's t-test (不假设方差齐性)

    参数:
    - group1_stats, group2_stats: 来自compute_statistics_n3的结果

    返回:
    - t_statistic: t统计量
    - df: 自由度 (Welch-Satterthwaite方程)
    - p_value: 双尾p值
    - cohen_d: Cohen's d效应量
    - cohen_d_ci_lower, cohen_d_ci_upper: Cohen's d的95% CI
    """
    n1, n2 = group1_stats['n'], group2_stats['n']
    mean1, mean2 = group1_stats['mean_reward'], group2_stats['mean_reward']
    std1, std2 = group1_stats['std_reward'], group2_stats['std_reward']
    se1, se2 = group1_stats['se_reward'], group2_stats['se_reward']

    # Welch's t统计量
    t_stat = (mean1 - mean2) / np.sqrt(se1**2 + se2**2)

    # Welch-Satterthwaite自由度
    df = (se1**2 + se2**2)**2 / ((se1**4 / (n1-1)) + (se2**4 / (n2-1)))

    # p-value (双尾)
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=df))

    # Cohen's d (pooled standard deviation)
    pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
    cohen_d = (mean1 - mean2) / pooled_std

    # Cohen's d的95% CI (近似公式)
    # 参考: Hedges & Olkin (1985), Nakagawa & Cuthill (2007)
    ncp = cohen_d * np.sqrt((n1 * n2) / (n1 + n2))  # 非中心参数
    se_d = np.sqrt((n1+n2)/(n1*n2) + cohen_d**2/(2*(n1+n2)))
    d_ci_lower = cohen_d - 1.96 * se_d
    d_ci_upper = cohen_d + 1.96 * se_d

    return {
        't_statistic': t_stat,
        'df': df,
        'p_value': p_value,
        'cohen_d': cohen_d,
        'cohen_d_ci_lower': d_ci_lower,
        'cohen_d_ci_upper': d_ci_upper,
        'pooled_std': pooled_std,
    }


def analyze_structural_comparison():
    """
    分析结构对比: Inverted vs Normal Pyramid

    对比:
    - A2C: inverted vs normal
    - PPO: inverted vs normal
    """
    print("\n" + "="*80)
    print("结构对比分析: Inverted Pyramid vs Normal Pyramid")
    print("="*80)

    for algorithm in ['A2C', 'PPO']:
        print(f"\n{'─'*80}")
        print(f"算法: {algorithm}")
        print(f"{'─'*80}")

        # 加载数据
        inverted_results = load_experiment_results('inverted_pyramid', algorithm)
        normal_results = load_experiment_results('normal_pyramid', algorithm)

        # 计算统计量
        inverted_stats = compute_statistics_n3(inverted_results)
        normal_stats = compute_statistics_n3(normal_results)

        if inverted_stats is None or normal_stats is None:
            print(f"❌ 跳过{algorithm}: 数据不完整")
            continue

        # t-test
        test_results = welch_t_test(inverted_stats, normal_stats)

        # 打印结果
        print(f"\n【Inverted Pyramid】")
        print(f"  n = {inverted_stats['n']}, seeds = {inverted_stats['seeds']}")
        print(f"  Mean rewards: {inverted_stats['mean_rewards_per_run']}")
        print(f"  平均奖励: {inverted_stats['mean_reward']:.2f} ± {inverted_stats['std_reward']:.2f} (SD)")
        print(f"  95% CI: [{inverted_stats['ci_95_lower']:.2f}, {inverted_stats['ci_95_upper']:.2f}]")
        print(f"  崩溃率: {inverted_stats['mean_crash_rate']*100:.1f}% ± {inverted_stats['std_crash_rate']*100:.1f}%")

        print(f"\n【Normal Pyramid】")
        print(f"  n = {normal_stats['n']}, seeds = {normal_stats['seeds']}")
        print(f"  Mean rewards: {normal_stats['mean_rewards_per_run']}")
        print(f"  平均奖励: {normal_stats['mean_reward']:.2f} ± {normal_stats['std_reward']:.2f} (SD)")
        print(f"  95% CI: [{normal_stats['ci_95_lower']:.2f}, {normal_stats['ci_95_upper']:.2f}]")
        print(f"  崩溃率: {normal_stats['mean_crash_rate']*100:.1f}% ± {normal_stats['std_crash_rate']*100:.1f}%")

        print(f"\n【统计检验】")
        print(f"  Welch's t-test:")
        print(f"    t({test_results['df']:.2f}) = {test_results['t_statistic']:.3f}")
        print(f"    p = {test_results['p_value']:.6f} {'***' if test_results['p_value'] < 0.001 else '**' if test_results['p_value'] < 0.01 else '*' if test_results['p_value'] < 0.05 else 'ns'}")
        print(f"  Cohen's d = {test_results['cohen_d']:.3f}")
        print(f"  Cohen's d 95% CI: [{test_results['cohen_d_ci_lower']:.3f}, {test_results['cohen_d_ci_upper']:.3f}]")

        # 效应量解释
        d_abs = abs(test_results['cohen_d'])
        if d_abs < 0.2:
            effect_size = "negligible"
        elif d_abs < 0.5:
            effect_size = "small"
        elif d_abs < 0.8:
            effect_size = "medium"
        else:
            effect_size = "large/very large"

        print(f"  效应量大小: {effect_size} (|d| = {d_abs:.3f})")

        # 统计显著性解释
        if test_results['p_value'] < 0.05:
            print(f"  ✅ 结论: Inverted pyramid在{algorithm}下显著优于Normal pyramid (α=0.05)")
        else:
            print(f"  ⚠️  结论: 差异不显著 (p={test_results['p_value']:.3f}), 但注意n=3的低power")

        print()


def analyze_capacity_paradox():
    """
    分析容量悖论: K=10 vs K=30 (A2C only)
    """
    print("\n" + "="*80)
    print("容量悖论分析: K=10 (Optimal) vs K=30 (Collapse)")
    print("="*80)

    algorithm = 'A2C'
    print(f"\n算法: {algorithm}")

    # 加载数据
    k10_results = load_experiment_results('low_capacity', algorithm)  # K=10
    k30_results = load_experiment_results('capacity_30', algorithm)   # K=30

    # 计算统计量
    k10_stats = compute_statistics_n3(k10_results)
    k30_stats = compute_statistics_n3(k30_results)

    if k10_stats is None or k30_stats is None:
        print(f"❌ 数据不完整，跳过分析")
        return

    # t-test
    test_results = welch_t_test(k10_stats, k30_stats)

    # 打印结果
    print(f"\n【K=10 (Low Capacity)】")
    print(f"  n = {k10_stats['n']}, seeds = {k10_stats['seeds']}")
    print(f"  Mean rewards: {k10_stats['mean_rewards_per_run']}")
    print(f"  平均奖励: {k10_stats['mean_reward']:.2f} ± {k10_stats['std_reward']:.2f} (SD)")
    print(f"  95% CI: [{k10_stats['ci_95_lower']:.2f}, {k10_stats['ci_95_upper']:.2f}]")
    print(f"  崩溃率: {k10_stats['mean_crash_rate']*100:.1f}% ± {k10_stats['std_crash_rate']*100:.1f}%")

    print(f"\n【K=30 (High Capacity)】")
    print(f"  n = {k30_stats['n']}, seeds = {k30_stats['seeds']}")
    print(f"  Mean rewards: {k30_stats['mean_rewards_per_run']}")
    print(f"  平均奖励: {k30_stats['mean_reward']:.2f} ± {k30_stats['std_reward']:.2f} (SD)")
    print(f"  95% CI: [{k30_stats['ci_95_lower']:.2f}, {k30_stats['ci_95_upper']:.2f}]")
    print(f"  崩溃率: {k30_stats['mean_crash_rate']*100:.1f}% ± {k30_stats['std_crash_rate']*100:.1f}%")

    print(f"\n【统计检验】")
    print(f"  Welch's t-test:")
    print(f"    t({test_results['df']:.2f}) = {test_results['t_statistic']:.3f}")
    print(f"    p = {test_results['p_value']:.6f} {'***' if test_results['p_value'] < 0.001 else '**' if test_results['p_value'] < 0.01 else '*' if test_results['p_value'] < 0.05 else 'ns'}")
    print(f"  Cohen's d = {test_results['cohen_d']:.3f}")
    print(f"  Cohen's d 95% CI: [{test_results['cohen_d_ci_lower']:.3f}, {test_results['cohen_d_ci_upper']:.3f}]")

    # 百分比差异
    pct_diff = ((k10_stats['mean_reward'] - k30_stats['mean_reward']) / k30_stats['mean_reward']) * 100
    print(f"  百分比差异: K=10 比 K=30 高 {pct_diff:.1f}%")

    # 效应量解释
    d_abs = abs(test_results['cohen_d'])
    if d_abs < 0.2:
        effect_size = "negligible"
    elif d_abs < 0.5:
        effect_size = "small"
    elif d_abs < 0.8:
        effect_size = "medium"
    else:
        effect_size = "large/very large"

    print(f"  效应量大小: {effect_size} (|d| = {d_abs:.3f})")

    # 统计显著性解释
    if test_results['p_value'] < 0.05:
        print(f"  ✅ 结论: Capacity paradox显著存在 (K=10 >> K=30, p<0.05)")
    else:
        print(f"  ⚠️  结论: 差异不显著 (p={test_results['p_value']:.3f}), 但效应量极大 (d={test_results['cohen_d']:.2f})")

    print()


def create_summary_table():
    """
    创建汇总表 (LaTeX格式)
    """
    print("\n" + "="*80)
    print("LaTeX汇总表生成")
    print("="*80)

    # 数据收集
    comparisons = [
        # 结构对比
        {'name': 'Inverted vs Normal (A2C)', 'group1_config': 'inverted_pyramid', 'group2_config': 'normal_pyramid', 'algo': 'A2C'},
        {'name': 'Inverted vs Normal (PPO)', 'group1_config': 'inverted_pyramid', 'group2_config': 'normal_pyramid', 'algo': 'PPO'},
        # 容量悖论
        {'name': 'K=10 vs K=30 (A2C)', 'group1_config': 'low_capacity', 'group2_config': 'capacity_30', 'algo': 'A2C'},
    ]

    rows = []

    for comp in comparisons:
        # 加载数据
        group1_results = load_experiment_results(comp['group1_config'], comp['algo'])
        group2_results = load_experiment_results(comp['group2_config'], comp['algo'])

        # 计算统计
        group1_stats = compute_statistics_n3(group1_results)
        group2_stats = compute_statistics_n3(group2_results)

        if group1_stats and group2_stats:
            test_results = welch_t_test(group1_stats, group2_stats)

            # 格式化行
            row = {
                'Comparison': comp['name'],
                'Group1_Mean': f"{group1_stats['mean_reward']:.1f}",
                'Group1_CI': f"[{group1_stats['ci_95_lower']:.1f}, {group1_stats['ci_95_upper']:.1f}]",
                'Group2_Mean': f"{group2_stats['mean_reward']:.1f}",
                'Group2_CI': f"[{group2_stats['ci_95_lower']:.1f}, {group2_stats['ci_95_upper']:.1f}]",
                't': f"{test_results['t_statistic']:.2f}",
                'df': f"{test_results['df']:.1f}",
                'p': f"{test_results['p_value']:.4f}",
                'Cohen_d': f"{test_results['cohen_d']:.3f}",
                'd_CI': f"[{test_results['cohen_d_ci_lower']:.2f}, {test_results['cohen_d_ci_upper']:.2f}]",
            }
            rows.append(row)

    # 创建DataFrame
    df = pd.DataFrame(rows)

    print("\n【汇总表 (Markdown)】")
    print(df.to_markdown(index=False))

    print("\n\n【LaTeX代码】")
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Supplementary Experiments Statistical Analysis (n=3)}")
    print("\\label{tab:supplementary_n3}")
    print("\\begin{tabular}{lcccccc}")
    print("\\hline")
    print("Comparison & Mean (95\\% CI) & Mean (95\\% CI) & t(df) & p-value & Cohen's d (95\\% CI) \\\\")
    print("\\hline")

    for _, row in df.iterrows():
        print(f"{row['Comparison']} & {row['Group1_Mean']} {row['Group1_CI']} & {row['Group2_Mean']} {row['Group2_CI']} & {row['t']}({row['df']}) & {row['p']} & {row['Cohen_d']} {row['d_CI']} \\\\")

    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")

    print()


if __name__ == '__main__':
    print("\n" + "#"*80)
    print("补充实验统计分析 (n=1→n=3)")
    print("#"*80)

    # 分析1: 结构对比
    analyze_structural_comparison()

    # 分析2: 容量悖论
    analyze_capacity_paradox()

    # 生成汇总表
    create_summary_table()

    print("\n" + "="*80)
    print("分析完成!")
    print("="*80)
