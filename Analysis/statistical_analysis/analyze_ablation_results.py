"""
Analyze Ablation Study Results

Computes statistics and generates comparison tables for ablation experiments.
"""

import pandas as pd
import numpy as np
from scipy import stats
import os
import sys


def analyze_ablation_results(results_file: str):
    """
    Analyze ablation study results

    Args:
        results_file: Path to ablation_results.csv
    """
    # Read results
    df = pd.read_csv(results_file)

    print("="*80)
    print("ABLATION STUDY ANALYSIS")
    print("="*80)

    # Group by variant
    grouped = df.groupby('variant')

    # Compute statistics
    stats_df = grouped.agg({
        'mean_reward': ['mean', 'std', 'count'],
        'crash_rate': 'mean',
        'training_time_minutes': 'mean'
    }).round(2)

    print("\n1. Summary Statistics:")
    print(stats_df)

    # Get baseline (HCA2C-Full)
    if 'hca2c_full' not in df['variant'].values:
        print("\n[ERROR] HCA2C-Full baseline not found in results!")
        return

    full_rewards = df[df['variant'] == 'hca2c_full']['mean_reward']
    baseline_mean = full_rewards.mean()

    print(f"\n2. Performance Relative to HCA2C-Full (baseline={baseline_mean:.2f}):")
    print("-"*80)

    results_summary = []

    for variant in df['variant'].unique():
        variant_rewards = df[df['variant'] == variant]['mean_reward']
        variant_mean = variant_rewards.mean()
        variant_std = variant_rewards.std()

        # Relative performance
        rel_performance = (variant_mean / baseline_mean - 1) * 100

        # Statistical test vs Full
        if variant != 'hca2c_full' and len(variant_rewards) > 1:
            t_stat, p_value = stats.ttest_ind(full_rewards, variant_rewards)
            cohens_d = (baseline_mean - variant_mean) / np.sqrt(
                (full_rewards.std()**2 + variant_std**2) / 2
            )
        else:
            t_stat, p_value, cohens_d = 0, 1.0, 0

        results_summary.append({
            'Variant': variant,
            'Mean': f"{variant_mean:.0f}",
            'Std': f"±{variant_std:.0f}",
            'vs Full': f"{rel_performance:+.1f}%",
            'p-value': f"{p_value:.4f}",
            "Cohen's d": f"{cohens_d:.3f}"
        })

        print(f"{variant:20s}: {variant_mean:8.0f} ± {variant_std:5.0f}  "
              f"({rel_performance:+6.1f}%)  p={p_value:.4f}  d={cohens_d:.3f}")

    # Component contributions
    print(f"\n3. Component Contributions:")
    print("-"*80)

    contributions = {}

    if 'hca2c_single' in df['variant'].values:
        single_mean = df[df['variant'] == 'hca2c_single']['mean_reward'].mean()
        contributions['Hierarchical Decomposition'] = baseline_mean - single_mean

    if 'hca2c_flat' in df['variant'].values:
        flat_mean = df[df['variant'] == 'hca2c_flat']['mean_reward'].mean()
        contributions['Neighbor-Aware Features'] = baseline_mean - flat_mean

    if 'hca2c_wide' in df['variant'].values:
        wide_mean = df[df['variant'] == 'hca2c_wide']['mean_reward'].mean()
        contributions['Capacity-Aware Clipping'] = baseline_mean - wide_mean

    for component, contribution in contributions.items():
        pct = (contribution / baseline_mean) * 100
        print(f"{component:30s}: {contribution:8.0f} ({pct:5.1f}%)")

    # A2C comparison
    if 'a2c_enhanced' in df['variant'].values:
        a2c_enhanced_mean = df[df['variant'] == 'a2c_enhanced']['mean_reward'].mean()
        a2c_improvement = ((a2c_enhanced_mean / 85650) - 1) * 100  # vs baseline A2C
        hca2c_vs_enhanced = ((baseline_mean / a2c_enhanced_mean) - 1) * 100

        print(f"\n4. Network Capacity Analysis:")
        print("-"*80)
        print(f"A2C-Enhanced (459K params): {a2c_enhanced_mean:.0f}")
        print(f"  vs A2C-Baseline (13K params): +{a2c_improvement:.1f}%")
        print(f"HCA2C-Full vs A2C-Enhanced: +{hca2c_vs_enhanced:.1f}%")
        print(f"\nConclusion: Simply increasing parameters gives only {a2c_improvement:.1f}% improvement,")
        print(f"            while HCA2C architecture gives {hca2c_vs_enhanced:.1f}% additional improvement.")

    # Save summary table
    summary_df = pd.DataFrame(results_summary)
    output_dir = os.path.dirname(results_file)
    summary_file = os.path.join(output_dir, 'ablation_summary.csv')
    summary_df.to_csv(summary_file, index=False)

    print(f"\n5. Summary saved to: {summary_file}")
    print("="*80)

    # Generate LaTeX table
    print(f"\n6. LaTeX Table:")
    print("-"*80)
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Ablation Study Results (Load 3.0×)}")
    print("\\begin{tabular}{lcccc}")
    print("\\hline")
    print("Variant & Mean Reward & Std & vs Full & Component \\\\")
    print("\\hline")

    for _, row in summary_df.iterrows():
        variant_name = row['Variant'].replace('_', '\\_')
        print(f"{variant_name} & {row['Mean']} & {row['Std']} & {row['vs Full']} & \\\\")

    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        results_file = sys.argv[1]
    else:
        results_file = 'Data/ablation_studies/ablation_results.csv'

    if os.path.exists(results_file):
        analyze_ablation_results(results_file)
    else:
        print(f"Error: Results file not found: {results_file}")
        print(f"\nUsage: python {sys.argv[0]} [results_file.csv]")
        print(f"Default: {results_file}")
