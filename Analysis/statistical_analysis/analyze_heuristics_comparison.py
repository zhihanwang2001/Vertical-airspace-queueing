"""
Heuristic Baselines Comparison Analysis
Compares RL algorithms (A2C, PPO) with heuristic baselines (FCFS, SJF, Priority, Heuristic)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")

print("="*80)
print("HEURISTIC BASELINES COMPARISON")
print("="*80)

# Load data
df = pd.read_csv("Data/summary/capacity_scan_results_complete.csv")

# Focus on loads 6× and 7× where we have heuristics
df_67 = df[df['load_multiplier'].isin([6.0, 7.0])].copy()

print(f"\n📊 Analyzing loads 6× and 7× with all algorithms")
print(f"📊 Total rows: {len(df_67)}")
print(f"📊 Algorithms: {sorted(df_67['algorithm'].unique())}")

# Group by algorithm, capacity, and load
summary = df_67.groupby(['algorithm', 'total_capacity', 'load_multiplier']).agg({
    'mean_reward': 'mean',
    'crash_rate': 'mean'
}).reset_index()

print("\n" + "="*80)
print("PERFORMANCE SUMMARY - LOAD 6×")
print("="*80)

load6 = summary[summary['load_multiplier'] == 6.0].sort_values('mean_reward', ascending=False)
print("\nK=10:")
k10_load6 = load6[load6['total_capacity'] == 10]
for _, row in k10_load6.iterrows():
    print(f"  {row['algorithm']:12s}: {row['mean_reward']:>12,.1f} (crash: {row['crash_rate']*100:>5.1f}%)")

print("\nK=30:")
k30_load6 = load6[load6['total_capacity'] == 30]
for _, row in k30_load6.iterrows():
    print(f"  {row['algorithm']:12s}: {row['mean_reward']:>12,.1f} (crash: {row['crash_rate']*100:>5.1f}%)")

print("\n" + "="*80)
print("PERFORMANCE SUMMARY - LOAD 7×")
print("="*80)

load7 = summary[summary['load_multiplier'] == 7.0].sort_values('mean_reward', ascending=False)
print("\nK=10:")
k10_load7 = load7[load7['total_capacity'] == 10]
for _, row in k10_load7.iterrows():
    print(f"  {row['algorithm']:12s}: {row['mean_reward']:>12,.1f} (crash: {row['crash_rate']*100:>5.1f}%)")

print("\nK=30:")
k30_load7 = load7[load7['total_capacity'] == 30]
for _, row in k30_load7.iterrows():
    print(f"  {row['algorithm']:12s}: {row['mean_reward']:>12,.1f} (crash: {row['crash_rate']*100:>5.1f}%)")

# Key findings
print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)

# Best performers at K=10
best_k10_6 = k10_load6.iloc[0]
best_k10_7 = k10_load7.iloc[0]

print(f"\n✅ Best at K=10, Load 6×: {best_k10_6['algorithm']} ({best_k10_6['mean_reward']:,.1f})")
print(f"✅ Best at K=10, Load 7×: {best_k10_7['algorithm']} ({best_k10_7['mean_reward']:,.1f})")

# RL vs best heuristic
rl_algos = ['A2C', 'PPO']
heuristic_algos = ['FCFS', 'SJF', 'Priority', 'Heuristic']

rl_k10_6 = k10_load6[k10_load6['algorithm'].isin(rl_algos)]['mean_reward'].mean()
heur_k10_6 = k10_load6[k10_load6['algorithm'].isin(heuristic_algos)]['mean_reward'].max()

print(f"\n📊 K=10, Load 6×:")
print(f"   RL average: {rl_k10_6:,.1f}")
print(f"   Best heuristic: {heur_k10_6:,.1f}")
print(f"   RL advantage: {((rl_k10_6 - heur_k10_6) / heur_k10_6 * 100):+.2f}%")

# Crash rates at K=30
print(f"\n⚠️  K=30 Crash Rates:")
for algo in sorted(df_67['algorithm'].unique()):
    crash_30 = summary[(summary['algorithm'] == algo) & (summary['total_capacity'] == 30)]['crash_rate'].mean()
    print(f"   {algo:12s}: {crash_30*100:>5.1f}%")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
