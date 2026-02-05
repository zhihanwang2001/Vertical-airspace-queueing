"""
Statistical Analysis for Experiment A: 5× Load Structural Comparison (n=3)

Objective:
- Analyze structural comparison at 5× load (Inverted vs Normal Pyramid)
- Calculate statistics (Welch's t-test, 95% CI, Cohen's d)
- Generate LaTeX tables for paper

Note:
- Data path: Data/ablation_studies/structural_5x_load/
- Seeds: 42, 123, 456 (n=3)
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats
import pandas as pd


def load_experiment_results_5x(config_type, algorithm, seeds=[42, 123, 456]):
    """
    Load Experiment A (5× load) n=3 results

    Args:
        config_type: Configuration type (inverted_pyramid or normal_pyramid)
        algorithm: Algorithm name (A2C or PPO)
        seeds: Random seed list [42, 123, 456]

    Returns:
        List of results dicts
    """
    results = []

    # All seeds come from structural_5x_load directory
    for seed in seeds:
        result_path = Path(__file__).parent.parent.parent / 'Data' / 'ablation_studies' / 'structural_5x_load' / config_type / f'{algorithm}_seed{seed}_results.json'

        if result_path.exists():
            with open(result_path, 'r') as f:
                data = json.load(f)
                results.append(data)
        else:
            print(f"Warning: Results not found {result_path}")

    return results


def compute_statistics_n3(results):
    """
    Calculate n=3 statistics

    Returns:
        - mean_reward: Mean reward
        - se_reward: Standard error (SE = SD / sqrt(n))
        - ci_95_lower, ci_95_upper: 95% confidence interval
        - mean_crash_rate: Mean crash rate
        - se_crash_rate: Crash rate standard error
    """
    if len(results) < 3:
        print(f"Error: Insufficient sample size (n={len(results)} < 3)")
        return None

    # Extract mean rewards (mean of each experiment)
    mean_rewards = [r['mean_reward'] for r in results]
    crash_rates = [r['crash_rate'] for r in results]

    # Statistics across training runs
    mean_reward = np.mean(mean_rewards)
    std_reward = np.std(mean_rewards, ddof=1)  # Sample standard deviation (n-1)
    se_reward = std_reward / np.sqrt(len(mean_rewards))  # Standard error

    # 95% CI (using t-distribution, df=n-1=2)
    t_critical = stats.t.ppf(0.975, df=len(mean_rewards)-1)  # Two-tailed test
    ci_lower = mean_reward - t_critical * se_reward
    ci_upper = mean_reward + t_critical * se_reward

    # Crash rate statistics
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
    Welch's t-test (does not assume equal variances)

    Args:
        group1_stats, group2_stats: Results from compute_statistics_n3

    Returns:
        - t_statistic: t-statistic
        - df: Degrees of freedom (Welch-Satterthwaite equation)
        - p_value: Two-tailed p-value
        - cohen_d: Cohen's d effect size
        - cohen_d_ci_lower, cohen_d_ci_upper: 95% CI for Cohen's d
    """
    n1, n2 = group1_stats['n'], group2_stats['n']
    mean1, mean2 = group1_stats['mean_reward'], group2_stats['mean_reward']
    std1, std2 = group1_stats['std_reward'], group2_stats['std_reward']
    se1, se2 = group1_stats['se_reward'], group2_stats['se_reward']

    # Welch's t-statistic
    t_stat = (mean1 - mean2) / np.sqrt(se1**2 + se2**2)

    # Welch-Satterthwaite degrees of freedom
    df = (se1**2 + se2**2)**2 / ((se1**4 / (n1-1)) + (se2**4 / (n2-1)))

    # p-value (two-tailed)
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=df))

    # Cohen's d (pooled standard deviation)
    pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
    cohen_d = (mean1 - mean2) / pooled_std

    # 95% CI for Cohen's d (approximate formula)
    # Reference: Hedges & Olkin (1985), Nakagawa & Cuthill (2007)
    ncp = cohen_d * np.sqrt((n1 * n2) / (n1 + n2))  # Non-centrality parameter
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


def analyze_structural_comparison_5x():
    """
    分析实验A: 5× Load结构对比
    Inverted vs Normal Pyramid

    对比:
    - A2C: inverted vs normal @ 5× load
    - PPO: inverted vs normal @ 5× load
    """
    print("\n" + "="*80)
    print("实验A分析: 5× Load结构对比 (Inverted vs Normal Pyramid)")
    print("="*80)

    results_summary = []

    for algorithm in ['A2C', 'PPO']:
        print(f"\n{'─'*80}")
        print(f"算法: {algorithm}")
        print(f"{'─'*80}")

        # 加载数据
        inverted_results = load_experiment_results_5x('inverted_pyramid', algorithm)
        normal_results = load_experiment_results_5x('normal_pyramid', algorithm)

        # 计算统计量
        inverted_stats = compute_statistics_n3(inverted_results)
        normal_stats = compute_statistics_n3(normal_results)

        if inverted_stats is None or normal_stats is None:
            print(f"❌ 跳过{algorithm}: 数据不完整")
            continue

        # t-test
        test_results = welch_t_test(inverted_stats, normal_stats)

        # 打印结果
        print(f"\n【Inverted Pyramid @ 5× Load】")
        print(f"  n = {inverted_stats['n']}, seeds = {inverted_stats['seeds']}")
        print(f"  Mean rewards: {inverted_stats['mean_rewards_per_run']}")
        print(f"  平均奖励: {inverted_stats['mean_reward']:.2f} ± {inverted_stats['std_reward']:.2f} (SD)")
        print(f"  95% CI: [{inverted_stats['ci_95_lower']:.2f}, {inverted_stats['ci_95_upper']:.2f}]")
        print(f"  崩溃率: {inverted_stats['mean_crash_rate']*100:.1f}% ± {inverted_stats['std_crash_rate']*100:.1f}%")

        print(f"\n【Normal Pyramid @ 5× Load】")
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
            verdict = "显著优于"
        else:
            print(f"  ⚠️  结论: 差异不显著 (p={test_results['p_value']:.3f}), 但注意n=3的低power")
            verdict = "差异不显著"

        # 保存结果
        results_summary.append({
            'algorithm': algorithm,
            'inverted_stats': inverted_stats,
            'normal_stats': normal_stats,
            'test_results': test_results,
            'verdict': verdict
        })

        print()

    return results_summary


def create_latex_table_5x(results_summary):
    """
    创建LaTeX汇总表 (实验A: 5× load)
    """
    print("\n" + "="*80)
    print("LaTeX表格生成 (实验A: 5× Load结构对比)")
    print("="*80)

    rows = []

    for result in results_summary:
        algo = result['algorithm']
        inv_stats = result['inverted_stats']
        norm_stats = result['normal_stats']
        test = result['test_results']

        row = {
            'Algorithm': algo,
            'Inverted_Mean': f"{inv_stats['mean_reward']:.0f}",
            'Inverted_CI': f"[{inv_stats['ci_95_lower']:.0f}, {inv_stats['ci_95_upper']:.0f}]",
            'Inverted_Crash': f"{inv_stats['mean_crash_rate']*100:.1f}\\%",
            'Normal_Mean': f"{norm_stats['mean_reward']:.0f}",
            'Normal_CI': f"[{norm_stats['ci_95_lower']:.0f}, {norm_stats['ci_95_upper']:.0f}]",
            'Normal_Crash': f"{norm_stats['mean_crash_rate']*100:.1f}\\%",
            't_stat': f"{test['t_statistic']:.2f}",
            'df': f"{test['df']:.1f}",
            'p_value': f"{test['p_value']:.4f}",
            'cohen_d': f"{test['cohen_d']:.2f}",
            'd_CI': f"[{test['cohen_d_ci_lower']:.2f}, {test['cohen_d_ci_upper']:.2f}]",
        }
        rows.append(row)

    # Markdown表格
    df = pd.DataFrame(rows)
    print("\n【Markdown表格】")
    print(df.to_markdown(index=False))

    # LaTeX代码
    print("\n\n【LaTeX代码】")
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Experiment A: Structural Comparison at 5$\\times$ Load (n=3)}")
    print("\\label{tab:experiment_a_5x_structural}")
    print("\\begin{tabular}{lcccccc}")
    print("\\hline")
    print("Algorithm & \\multicolumn{2}{c}{Inverted Pyramid} & \\multicolumn{2}{c}{Normal Pyramid} & Statistical Test \\\\")
    print("          & Mean (95\\% CI) & Crash & Mean (95\\% CI) & Crash & t(df), p, d \\\\")
    print("\\hline")

    for _, row in df.iterrows():
        print(f"{row['Algorithm']} & {row['Inverted_Mean']} {row['Inverted_CI']} & {row['Inverted_Crash']} & "
              f"{row['Normal_Mean']} {row['Normal_CI']} & {row['Normal_Crash']} & "
              f"t({row['df']})={row['t_stat']}, p={row['p_value']}, d={row['cohen_d']} \\\\")

    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")

    # 生成简化版（用于主文）
    print("\n\n【简化LaTeX表格 (用于论文正文)】")
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Structural Comparison: Inverted vs Normal Pyramid at 5$\\times$ Load}")
    print("\\label{tab:structural_comparison_5x}")
    print("\\begin{tabular}{lccccc}")
    print("\\hline")
    print("Algorithm & Inverted (95\\% CI) & Normal (95\\% CI) & p-value & Cohen's d \\\\")
    print("\\hline")

    for _, row in df.iterrows():
        sig_marker = "***" if float(row['p_value']) < 0.001 else "**" if float(row['p_value']) < 0.01 else "*" if float(row['p_value']) < 0.05 else ""
        print(f"{row['Algorithm']} & {row['Inverted_Mean']} {row['Inverted_CI']} & "
              f"{row['Normal_Mean']} {row['Normal_CI']} & "
              f"{row['p_value']}{sig_marker} & {row['cohen_d']} {row['d_CI']} \\\\")

    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")

    print()


def save_results_json_5x(results_summary):
    """
    保存分析结果到JSON
    """
    output_path = Path(__file__).parent.parent.parent / 'Data' / 'ablation_studies' / 'structural_5x_load' / 'STATISTICAL_ANALYSIS.json'

    output_data = []
    for result in results_summary:
        output_data.append({
            'algorithm': result['algorithm'],
            'inverted_pyramid': {
                'n': result['inverted_stats']['n'],
                'mean_reward': float(result['inverted_stats']['mean_reward']),
                'std_reward': float(result['inverted_stats']['std_reward']),
                'ci_95': [float(result['inverted_stats']['ci_95_lower']),
                         float(result['inverted_stats']['ci_95_upper'])],
                'mean_crash_rate': float(result['inverted_stats']['mean_crash_rate']),
            },
            'normal_pyramid': {
                'n': result['normal_stats']['n'],
                'mean_reward': float(result['normal_stats']['mean_reward']),
                'std_reward': float(result['normal_stats']['std_reward']),
                'ci_95': [float(result['normal_stats']['ci_95_lower']),
                         float(result['normal_stats']['ci_95_upper'])],
                'mean_crash_rate': float(result['normal_stats']['mean_crash_rate']),
            },
            'statistical_test': {
                't_statistic': float(result['test_results']['t_statistic']),
                'df': float(result['test_results']['df']),
                'p_value': float(result['test_results']['p_value']),
                'cohen_d': float(result['test_results']['cohen_d']),
                'cohen_d_ci': [float(result['test_results']['cohen_d_ci_lower']),
                              float(result['test_results']['cohen_d_ci_upper'])],
            },
            'verdict': result['verdict']
        })

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✅ 分析结果已保存至: {output_path}")


if __name__ == '__main__':
    print("\n" + "#"*80)
    print("实验A统计分析: 5× Load结构对比 (n=3)")
    print("#"*80)

    # 分析结构对比
    results_summary = analyze_structural_comparison_5x()

    if len(results_summary) > 0:
        # 生成LaTeX表格
        create_latex_table_5x(results_summary)

        # 保存结果
        save_results_json_5x(results_summary)

    print("\n" + "="*80)
    print("分析完成!")
    print("="*80)
