"""
Analyze HCA2C Final Comparison Results (45 experiments)

Analyzes: Data/hca2c_final_comparison/
- 3 algorithms (HCA2C, A2C, PPO)
- 5 seeds (42, 43, 44, 45, 46)
- 3 loads (3.0, 5.0, 7.0)
= 45 experiments total

Outputs:
- Statistical summary tables
- Comparison figures
- LaTeX-formatted results
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / 'Data' / 'hca2c_final_comparison'
OUT_DIR = PROJECT_ROOT / 'Analysis' / 'statistical_reports'
FIG_DIR = PROJECT_ROOT / 'Analysis' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def collect_data() -> pd.DataFrame:
    """Collect all 45 experiment results into a DataFrame."""
    rows: List[Dict] = []

    for json_file in sorted(DATA_DIR.glob('*.json')):
        try:
            data = json.loads(json_file.read_text())

            rows.append({
                'algorithm': data['algorithm'],
                'seed': data['seed'],
                'load': data['load_multiplier'],
                'mean_reward': data['mean_reward'],
                'std_reward': data['std_reward'],
                'crash_rate': data['crash_rate'],
                'train_time': data['train_time'],
                'timesteps': data['timesteps'],
            })
        except Exception as e:
            print(f"Warning: Failed to load {json_file.name}: {e}")
            continue

    df = pd.DataFrame(rows)
    print(f"\n✓ Loaded {len(df)} experiments")
    print(f"  Algorithms: {sorted(df['algorithm'].unique())}")
    print(f"  Seeds: {sorted(df['seed'].unique())}")
    print(f"  Loads: {sorted(df['load'].unique())}")

    return df


def compute_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute descriptive statistics for each algorithm-load combination."""

    stats_df = df.groupby(['algorithm', 'load']).agg(
        n=('mean_reward', 'size'),
        mean=('mean_reward', 'mean'),
        std=('mean_reward', 'std'),
        min=('mean_reward', 'min'),
        max=('mean_reward', 'max'),
        median=('mean_reward', 'median'),
        cv=('mean_reward', lambda x: np.std(x) / np.abs(np.mean(x)) * 100 if np.mean(x) != 0 else np.nan),
        crash_rate_mean=('crash_rate', 'mean'),
        train_time_mean=('train_time', 'mean'),
    ).reset_index()

    # Format for display
    stats_df['mean_std'] = stats_df.apply(
        lambda row: f"{row['mean']:.1f} ± {row['std']:.1f}", axis=1
    )

    return stats_df


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0.0


def pairwise_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    """Perform pairwise statistical comparisons between algorithms."""

    results = []

    for load in sorted(df['load'].unique()):
        df_load = df[df['load'] == load]

        # HCA2C vs A2C
        hca2c = df_load[df_load['algorithm'] == 'HCA2C']['mean_reward'].values
        a2c = df_load[df_load['algorithm'] == 'A2C']['mean_reward'].values

        if len(hca2c) > 0 and len(a2c) > 0:
            t_stat, p_val = stats.ttest_ind(hca2c, a2c, equal_var=False)
            d = cohens_d(hca2c, a2c)

            results.append({
                'load': load,
                'comparison': 'HCA2C vs A2C',
                'n1': len(hca2c),
                'n2': len(a2c),
                'mean1': np.mean(hca2c),
                'mean2': np.mean(a2c),
                'diff': np.mean(hca2c) - np.mean(a2c),
                't_stat': t_stat,
                'p_value': p_val,
                'cohens_d': d,
                'significant': 'Yes' if p_val < 0.05 else 'No',
            })

        # HCA2C vs PPO
        ppo = df_load[df_load['algorithm'] == 'PPO']['mean_reward'].values

        if len(hca2c) > 0 and len(ppo) > 0:
            t_stat, p_val = stats.ttest_ind(hca2c, ppo, equal_var=False)
            d = cohens_d(hca2c, ppo)

            results.append({
                'load': load,
                'comparison': 'HCA2C vs PPO',
                'n1': len(hca2c),
                'n2': len(ppo),
                'mean1': np.mean(hca2c),
                'mean2': np.mean(ppo),
                'diff': np.mean(hca2c) - np.mean(ppo),
                't_stat': t_stat,
                'p_value': p_val,
                'cohens_d': d,
                'significant': 'Yes' if p_val < 0.05 else 'No',
            })

        # A2C vs PPO
        if len(a2c) > 0 and len(ppo) > 0:
            t_stat, p_val = stats.ttest_ind(a2c, ppo, equal_var=False)
            d = cohens_d(a2c, ppo)

            results.append({
                'load': load,
                'comparison': 'A2C vs PPO',
                'n1': len(a2c),
                'n2': len(ppo),
                'mean1': np.mean(a2c),
                'mean2': np.mean(ppo),
                'diff': np.mean(a2c) - np.mean(a2c),
                't_stat': t_stat,
                'p_value': p_val,
                'cohens_d': d,
                'significant': 'Yes' if p_val < 0.05 else 'No',
            })

    return pd.DataFrame(results)


def create_summary_figure(df: pd.DataFrame, stats_df: pd.DataFrame):
    """Create comprehensive summary figure."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('HCA2C Final Comparison: 45 Experiments Analysis', fontsize=16, fontweight='bold')

    # 1. Bar plot with error bars
    ax1 = axes[0, 0]
    algorithms = sorted(df['algorithm'].unique())
    loads = sorted(df['load'].unique())
    x = np.arange(len(loads))
    width = 0.25

    for i, algo in enumerate(algorithms):
        algo_stats = stats_df[stats_df['algorithm'] == algo]
        means = [algo_stats[algo_stats['load'] == load]['mean'].values[0] for load in loads]
        stds = [algo_stats[algo_stats['load'] == load]['std'].values[0] for load in loads]
        ax1.bar(x + i * width, means, width, label=algo, yerr=stds, capsize=5)

    ax1.set_xlabel('Load Multiplier', fontsize=12)
    ax1.set_ylabel('Mean Reward', fontsize=12)
    ax1.set_title('Performance Across Loads', fontsize=13, fontweight='bold')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels([f'{load}×' for load in loads])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # 2. Box plot
    ax2 = axes[0, 1]
    data_for_box = []
    labels_for_box = []
    for algo in algorithms:
        for load in loads:
            data = df[(df['algorithm'] == algo) & (df['load'] == load)]['mean_reward'].values
            if len(data) > 0:
                data_for_box.append(data)
                labels_for_box.append(f'{algo}\n{load}×')

    bp = ax2.boxplot(data_for_box, labels=labels_for_box, patch_artist=True)
    for patch, i in zip(bp['boxes'], range(len(data_for_box))):
        algo_idx = i // len(loads)
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        patch.set_facecolor(colors[algo_idx % len(colors)])
        patch.set_alpha(0.7)

    ax2.set_ylabel('Mean Reward', fontsize=12)
    ax2.set_title('Distribution Comparison', fontsize=13, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45, labelsize=9)
    ax2.grid(axis='y', alpha=0.3)

    # 3. Coefficient of Variation (Stability)
    ax3 = axes[1, 0]
    for algo in algorithms:
        algo_stats = stats_df[stats_df['algorithm'] == algo]
        cvs = [algo_stats[algo_stats['load'] == load]['cv'].values[0] for load in loads]
        ax3.plot(loads, cvs, marker='o', linewidth=2, markersize=8, label=algo)

    ax3.set_xlabel('Load Multiplier', fontsize=12)
    ax3.set_ylabel('Coefficient of Variation (%)', fontsize=12)
    ax3.set_title('Stability Analysis (Lower is Better)', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)

    # 4. Training time comparison
    ax4 = axes[1, 1]
    for algo in algorithms:
        algo_stats = stats_df[stats_df['algorithm'] == algo]
        times = [algo_stats[algo_stats['load'] == load]['train_time_mean'].values[0] / 60 for load in loads]
        ax4.plot(loads, times, marker='s', linewidth=2, markersize=8, label=algo)

    ax4.set_xlabel('Load Multiplier', fontsize=12)
    ax4.set_ylabel('Training Time (minutes)', fontsize=12)
    ax4.set_title('Computational Efficiency', fontsize=13, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)

    plt.tight_layout()

    # Save figure
    fig_path = FIG_DIR / 'hca2c_ablation_comprehensive.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved figure: {fig_path}")

    plt.close()


def save_latex_table(stats_df: pd.DataFrame):
    """Generate LaTeX table for manuscript."""

    latex_lines = []
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{HCA2C Final Comparison Results}")
    latex_lines.append("\\label{tab:hca2c_ablation}")
    latex_lines.append("\\begin{tabular}{llrrrrr}")
    latex_lines.append("\\toprule")
    latex_lines.append("Algorithm & Load & n & Mean $\\pm$ SD & CV (\\%) & Crash Rate & Time (min) \\\\")
    latex_lines.append("\\midrule")

    for _, row in stats_df.iterrows():
        latex_lines.append(
            f"{row['algorithm']} & {row['load']:.1f}× & {int(row['n'])} & "
            f"{row['mean']:.1f} $\\pm$ {row['std']:.1f} & "
            f"{row['cv']:.2f} & {row['crash_rate_mean']:.3f} & "
            f"{row['train_time_mean']/60:.1f} \\\\"
        )

    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")

    latex_path = OUT_DIR / 'hca2c_ablation_table.tex'
    latex_path.write_text('\n'.join(latex_lines))
    print(f"✓ Saved LaTeX table: {latex_path}")


def generate_markdown_report(df: pd.DataFrame, stats_df: pd.DataFrame, comp_df: pd.DataFrame):
    """Generate comprehensive markdown report."""

    lines = []
    lines.append("# HCA2C Final Comparison Analysis Report")
    lines.append(f"\n**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"\n**Total Experiments:** {len(df)}")
    lines.append(f"- Algorithms: {', '.join(sorted(df['algorithm'].unique()))}")
    lines.append(f"- Seeds: {', '.join(map(str, sorted(df['seed'].unique())))}")
    lines.append(f"- Loads: {', '.join(map(str, sorted(df['load'].unique())))}")

    lines.append("\n## Descriptive Statistics\n")
    lines.append("| Algorithm | Load | n | Mean ± SD | CV (%) | Crash Rate | Time (min) |")
    lines.append("|-----------|------|---|-----------|--------|------------|------------|")

    for _, row in stats_df.iterrows():
        lines.append(
            f"| {row['algorithm']} | {row['load']:.1f}× | {int(row['n'])} | "
            f"{row['mean']:.1f} ± {row['std']:.1f} | {row['cv']:.2f} | "
            f"{row['crash_rate_mean']:.3f} | {row['train_time_mean']/60:.1f} |"
        )

    lines.append("\n## Pairwise Comparisons\n")
    lines.append("| Load | Comparison | Mean Diff | t-stat | p-value | Cohen's d | Significant |")
    lines.append("|------|------------|-----------|--------|---------|-----------|-------------|")

    for _, row in comp_df.iterrows():
        lines.append(
            f"| {row['load']:.1f}× | {row['comparison']} | "
            f"{row['diff']:.1f} | {row['t_stat']:.3f} | "
            f"{row['p_value']:.4f} | {row['cohens_d']:.3f} | {row['significant']} |"
        )

    lines.append("\n## Key Findings\n")

    # Find best algorithm per load
    for load in sorted(df['load'].unique()):
        load_stats = stats_df[stats_df['load'] == load]
        best = load_stats.loc[load_stats['mean'].idxmax()]
        lines.append(f"- **Load {load:.1f}×**: {best['algorithm']} achieves highest mean reward "
                    f"({best['mean']:.1f} ± {best['std']:.1f})")

    # Stability analysis
    lines.append("\n### Stability (Coefficient of Variation)")
    for algo in sorted(df['algorithm'].unique()):
        algo_stats = stats_df[stats_df['algorithm'] == algo]
        avg_cv = algo_stats['cv'].mean()
        lines.append(f"- **{algo}**: Average CV = {avg_cv:.2f}% "
                    f"({'Most stable' if avg_cv == stats_df.groupby('algorithm')['cv'].mean().min() else 'Less stable'})")

    report_path = OUT_DIR / 'hca2c_ablation_report.md'
    report_path.write_text('\n'.join(lines))
    print(f"✓ Saved report: {report_path}")


def main():
    """Main analysis pipeline."""

    print("\n" + "="*60)
    print("HCA2C FINAL COMPARISON ANALYSIS")
    print("="*60)

    # 1. Collect data
    print("\n[1/6] Collecting data...")
    df = collect_data()

    # 2. Descriptive statistics
    print("\n[2/6] Computing descriptive statistics...")
    stats_df = compute_descriptive_stats(df)
    stats_csv = OUT_DIR / 'hca2c_ablation_stats.csv'
    stats_df.to_csv(stats_csv, index=False)
    print(f"✓ Saved: {stats_csv}")

    # 3. Pairwise comparisons
    print("\n[3/6] Performing pairwise comparisons...")
    comp_df = pairwise_comparisons(df)
    comp_csv = OUT_DIR / 'hca2c_ablation_comparisons.csv'
    comp_df.to_csv(comp_csv, index=False)
    print(f"✓ Saved: {comp_csv}")

    # 4. Create figures
    print("\n[4/6] Creating summary figure...")
    create_summary_figure(df, stats_df)

    # 5. Generate LaTeX table
    print("\n[5/6] Generating LaTeX table...")
    save_latex_table(stats_df)

    # 6. Generate markdown report
    print("\n[6/6] Generating markdown report...")
    generate_markdown_report(df, stats_df, comp_df)

    print("\n" + "="*60)
    print("✓ ANALYSIS COMPLETE")
    print("="*60)
    print(f"\nOutputs saved to:")
    print(f"  - {OUT_DIR}/")
    print(f"  - {FIG_DIR}/")
    print()


if __name__ == '__main__':
    main()
