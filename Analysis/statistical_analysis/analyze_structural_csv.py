"""
Analyze structural comparison from CSV files
Compares inverted vs reverse (normal) pyramid across loads 3×, 7×, 10×
"""

import pandas as pd
import numpy as np
from scipy import stats

print("="*80)
print("STRUCTURAL COMPARISON ANALYSIS")
print("="*80)

# Load data
print("\nLoading data...")
df_inv = pd.read_csv("Data/summary/capacity_scan_results_inverted_3_7_10.csv")
df_rev = pd.read_csv("Data/summary/capacity_scan_results_reverse_3_7_10.csv")

print(f"Inverted: {len(df_inv)} rows")
print(f"Reverse: {len(df_rev)} rows")

# Add structure label
df_inv['structure'] = 'inverted'
df_rev['structure'] = 'reverse'

# Combine
df = pd.concat([df_inv, df_rev], ignore_index=True)

print(f"\nTotal: {len(df)} rows")
print(f"Loads: {sorted(df['load_multiplier'].unique())}")
print(f"Capacities: {sorted(df['total_capacity'].unique())}")

# Analyze each load level
print("\n" + "="*80)
print("COMPARISON BY LOAD LEVEL")
print("="*80)

results = []
for load in sorted(df['load_multiplier'].unique()):
    for capacity in sorted(df['total_capacity'].unique()):
        subset = df[(df['load_multiplier'] == load) & (df['total_capacity'] == capacity)]

        inv_data = subset[subset['structure'] == 'inverted']['mean_reward'].values
        rev_data = subset[subset['structure'] == 'reverse']['mean_reward'].values

        if len(inv_data) == 0 or len(rev_data) == 0:
            continue

        inv_mean = np.mean(inv_data)
        rev_mean = np.mean(rev_data)
        advantage = ((inv_mean - rev_mean) / rev_mean * 100) if rev_mean != 0 else 0

        # t-test
        t_stat, p_value = stats.ttest_ind(inv_data, rev_data)

        # Cohen's d
        pooled_std = np.sqrt((np.std(inv_data, ddof=1)**2 + np.std(rev_data, ddof=1)**2) / 2)
        cohens_d = (inv_mean - rev_mean) / pooled_std if pooled_std > 0 else 0

        results.append({
            'load': load,
            'capacity': capacity,
            'inverted_mean': inv_mean,
            'reverse_mean': rev_mean,
            'advantage_pct': advantage,
            't_stat': t_stat,
            'p_value': p_value,
            'cohens_d': cohens_d
        })

        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        winner = "INV" if advantage > 0 else "REV"

        print(f"\nLoad {load}×, K={capacity}:")
        print(f"  Inverted: {inv_mean:>12,.1f}")
        print(f"  Reverse:  {rev_mean:>12,.1f}")
        print(f"  Advantage: {advantage:>+6.2f}% → {winner}")
        print(f"  t={t_stat:.3f}, p={p_value:.6f} {sig}, d={cohens_d:.3f}")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("Analysis/statistical_reports/structural_comparison_results.csv", index=False)
print(f"\n✅ Saved: Analysis/statistical_reports/structural_comparison_results.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
