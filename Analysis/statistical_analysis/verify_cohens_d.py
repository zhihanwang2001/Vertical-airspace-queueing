"""
Verify Cohen's d Calculations

This script addresses the critical reviewer concern:
"Cohen's d values (48.452, 15.678) are extraordinarily large and require investigation"

Recalculates effect sizes using correct formula:
Cohen's d = (μ₁ - μ₂) / σ_pooled

where σ_pooled = sqrt(((n₁-1)*σ₁² + (n₂-1)*σ₂²) / (n₁ + n₂ - 2))
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict


def calculate_cohens_d(group1_data: np.ndarray, group2_data: np.ndarray) -> Dict:
    """
    Calculate Cohen's d with pooled standard deviation.

    Returns dict with:
    - cohens_d: Effect size
    - mean1, mean2: Group means
    - std1, std2: Group standard deviations
    - n1, n2: Sample sizes
    - pooled_std: Pooled standard deviation
    """
    n1 = len(group1_data)
    n2 = len(group2_data)

    mean1 = np.mean(group1_data)
    mean2 = np.mean(group2_data)

    std1 = np.std(group1_data, ddof=1)  # Sample std (n-1)
    std2 = np.std(group2_data, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))

    # Cohen's d
    cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0

    return {
        'cohens_d': cohens_d,
        'mean1': mean1,
        'mean2': mean2,
        'std1': std1,
        'std2': std2,
        'n1': n1,
        'n2': n2,
        'pooled_std': pooled_std,
        'mean_diff': mean1 - mean2,
        'percent_diff': ((mean1 - mean2) / mean2 * 100) if mean2 != 0 else 0
    }


def verify_structural_comparison():
    """Verify structural comparison (inverted vs reverse pyramid)."""
    print("=" * 80)
    print("STRUCTURAL COMPARISON: Inverted vs Reverse Pyramid")
    print("=" * 80)

    # Load data
    inverted_file = Path('Data/summary/capacity_scan_results_inverted_3_7_10.csv')
    reverse_file = Path('Data/summary/capacity_scan_results_reverse_3_7_10.csv')

    if not inverted_file.exists() or not reverse_file.exists():
        print(f"⚠️  Data files not found:")
        print(f"   {inverted_file}: {inverted_file.exists()}")
        print(f"   {reverse_file}: {reverse_file.exists()}")
        return

    df_inv = pd.read_csv(inverted_file)
    df_rev = pd.read_csv(reverse_file)

    print(f"\n📊 Data loaded:")
    print(f"   Inverted: {len(df_inv)} rows")
    print(f"   Reverse: {len(df_rev)} rows")

    # Filter for RL algorithms only
    df_inv_rl = df_inv[df_inv['family'] == 'RL']
    df_rev_rl = df_rev[df_rev['family'] == 'RL']

    print(f"\n🤖 RL algorithms only:")
    print(f"   Inverted: {len(df_inv_rl)} rows")
    print(f"   Reverse: {len(df_rev_rl)} rows")

    # Analyze by load multiplier
    for load in sorted(df_inv_rl['load_multiplier'].unique()):
        print(f"\n{'─' * 80}")
        print(f"Load Multiplier: {load}×")
        print(f"{'─' * 80}")

        inv_load = df_inv_rl[df_inv_rl['load_multiplier'] == load]['mean_reward'].values
        rev_load = df_rev_rl[df_rev_rl['load_multiplier'] == load]['mean_reward'].values

        if len(inv_load) == 0 or len(rev_load) == 0:
            print(f"⚠️  No data for load {load}×")
            continue

        stats = calculate_cohens_d(inv_load, rev_load)

        print(f"\n📈 Inverted Pyramid:")
        print(f"   n = {stats['n1']}")
        print(f"   Mean = {stats['mean1']:.2f}")
        print(f"   Std = {stats['std1']:.2f}")

        print(f"\n📉 Reverse Pyramid:")
        print(f"   n = {stats['n2']}")
        print(f"   Mean = {stats['mean2']:.2f}")
        print(f"   Std = {stats['std2']:.2f}")

        print(f"\n📊 Effect Size:")
        print(f"   Mean difference = {stats['mean_diff']:.2f}")
        print(f"   Percent difference = {stats['percent_diff']:.2f}%")
        print(f"   Pooled std = {stats['pooled_std']:.2f}")
        print(f"   Cohen's d = {stats['cohens_d']:.4f}")

        # Interpretation
        abs_d = abs(stats['cohens_d'])
        if abs_d < 0.2:
            interpretation = "negligible"
        elif abs_d < 0.5:
            interpretation = "small"
        elif abs_d < 0.8:
            interpretation = "medium"
        elif abs_d < 1.2:
            interpretation = "large"
        else:
            interpretation = "very large" if abs_d < 10 else "EXTRAORDINARILY LARGE"

        print(f"   Interpretation: {interpretation}")

        if abs_d > 10:
            print(f"\n⚠️  WARNING: Effect size > 10 is extremely unusual!")
            print(f"   Possible causes:")
            print(f"   1. Perfect or near-perfect separation between groups")
            print(f"   2. Very small pooled standard deviation")
            print(f"   3. Data quality issues")


def verify_drl_vs_heuristics():
    """Verify DRL vs heuristics comparison."""
    print("\n\n")
    print("=" * 80)
    print("DRL VS HEURISTICS COMPARISON")
    print("=" * 80)

    # Load comprehensive data
    data_file = Path('Data/summary/capacity_scan_results_uniform_3_7_10.csv')

    if not data_file.exists():
        # Try alternative file
        data_file = Path('Data/summary/capacity_scan_results_all.csv')

    if not data_file.exists():
        print(f"⚠️  Data file not found: {data_file}")
        return

    df = pd.read_csv(data_file)

    print(f"\n📊 Data loaded: {len(df)} rows")

    # Separate RL and Heuristics
    df_rl = df[df['family'] == 'RL']
    df_heur = df[df['family'] == 'Heuristic']

    print(f"\n🤖 RL: {len(df_rl)} rows")
    print(f"📏 Heuristics: {len(df_heur)} rows")

    if len(df_rl) == 0 or len(df_heur) == 0:
        print("⚠️  Insufficient data for comparison")
        return

    rl_rewards = df_rl['mean_reward'].values
    heur_rewards = df_heur['mean_reward'].values

    stats = calculate_cohens_d(rl_rewards, heur_rewards)

    print(f"\n📈 RL Algorithms:")
    print(f"   n = {stats['n1']}")
    print(f"   Mean = {stats['mean1']:.2f}")
    print(f"   Std = {stats['std1']:.2f}")

    print(f"\n📉 Heuristics:")
    print(f"   n = {stats['n2']}")
    print(f"   Mean = {stats['mean2']:.2f}")
    print(f"   Std = {stats['std2']:.2f}")

    print(f"\n📊 Effect Size:")
    print(f"   Mean difference = {stats['mean_diff']:.2f}")
    print(f"   Percent difference = {stats['percent_diff']:.2f}%")
    print(f"   Pooled std = {stats['pooled_std']:.2f}")
    print(f"   Cohen's d = {stats['cohens_d']:.4f}")


def main():
    """Run all verifications."""
    print("\n🔍 COHEN'S D VERIFICATION REPORT")
    print("=" * 80)
    print("Addressing reviewer concern about extraordinarily large effect sizes")
    print("=" * 80)

    verify_structural_comparison()
    verify_drl_vs_heuristics()

    print("\n\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("\nThis analysis recalculates Cohen's d using the correct pooled standard")
    print("deviation formula. If effect sizes remain > 10, this indicates:")
    print("  1. Genuine perfect/near-perfect separation between groups")
    print("  2. Very consistent performance within groups (low variance)")
    print("  3. Large mean differences relative to within-group variation")
    print("\nSuch large effect sizes are rare but can occur in computational")
    print("experiments where algorithms have fundamentally different behaviors.")
    print("=" * 80)


if __name__ == '__main__':
    main()
