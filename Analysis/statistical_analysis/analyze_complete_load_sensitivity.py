"""
Complete Load Sensitivity Analysis
Analyzes capacity paradox across all load levels: 3×, 4×, 6×, 7×, 8×, 9×, 10×
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Load complete data
print("Loading complete capacity scan data...")
df = pd.read_csv("Data/summary/capacity_scan_results_complete.csv")

print(f"📊 Total rows: {len(df)}")
print(f"📊 Load levels: {sorted(df['load_multiplier'].unique())}")
print(f"📊 Capacities: {sorted(df['total_capacity'].unique())}")
print(f"📊 Algorithms: {sorted(df['algorithm'].unique())}")

# Filter RL algorithms only for main analysis
df_rl = df[df['family'] == 'RL'].copy()

print("\n" + "="*80)
print("CAPACITY PARADOX ANALYSIS - ALL LOADS")
print("="*80)

# Analyze each load level
results = []
for load in sorted(df_rl['load_multiplier'].unique()):
    load_data = df_rl[df_rl['load_multiplier'] == load]

    k10_data = load_data[load_data['total_capacity'] == 10]
    k30_data = load_data[load_data['total_capacity'] == 30]

    if len(k10_data) == 0 or len(k30_data) == 0:
        continue

    k10_rewards = k10_data['mean_reward'].values
    k30_rewards = k30_data['mean_reward'].values

    k10_mean = np.mean(k10_rewards)
    k30_mean = np.mean(k30_rewards)
    k10_crash = np.mean(k10_data['crash_rate'].values)
    k30_crash = np.mean(k30_data['crash_rate'].values)

    # Statistical test
    t_stat, p_value = stats.ttest_ind(k10_rewards, k30_rewards)

    # Cohen's d
    pooled_std = np.sqrt((np.std(k10_rewards, ddof=1)**2 + np.std(k30_rewards, ddof=1)**2) / 2)
    cohens_d = (k10_mean - k30_mean) / pooled_std if pooled_std > 0 else 0

    winner = "K=10" if k10_mean > k30_mean else "K=30"

    results.append({
        'load': load,
        'k10_mean': k10_mean,
        'k30_mean': k30_mean,
        'k10_crash': k10_crash,
        'k30_crash': k30_crash,
        'difference': k10_mean - k30_mean,
        't_stat': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'winner': winner
    })

    sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
    print(f"\nLoad {load}×:")
    print(f"  K=10: {k10_mean:>12,.1f} (crash: {k10_crash*100:>5.1f}%)")
    print(f"  K=30: {k30_mean:>12,.1f} (crash: {k30_crash*100:>5.1f}%)")
    print(f"  Diff: {k10_mean - k30_mean:>12,.1f} → {winner}")
    print(f"  t={t_stat:.3f}, p={p_value:.6f} {sig}, d={cohens_d:.3f}")

results_df = pd.DataFrame(results)

# Find transition point
print("\n" + "="*80)
print("TRANSITION POINT ANALYSIS")
print("="*80)

transition_load = None
for i in range(len(results_df) - 1):
    if results_df.iloc[i]['winner'] == 'K=30' and results_df.iloc[i+1]['winner'] == 'K=10':
        transition_load = results_df.iloc[i+1]['load']
        print(f"🎯 Capacity paradox emerges between {results_df.iloc[i]['load']}× and {transition_load}×")
        break

if transition_load is None:
    print("⚠️  No clear transition point found in data")

# Save results
results_df.to_csv("Analysis/statistical_reports/complete_load_sensitivity.csv", index=False)
print(f"\n✅ Saved results to: Analysis/statistical_reports/complete_load_sensitivity.csv")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
