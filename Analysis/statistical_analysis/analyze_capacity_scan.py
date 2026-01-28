"""
Analyze Capacity Scan Results (Heuristics only quick check)

Reads Data/summary/capacity_scan_results_all.csv and reports for each shape
and K whether crash rate increases at K=30 vs K=10 under the same load.
Also prints mean rewards to illustrate capacity paradox signs.
"""

import pandas as pd
from pathlib import Path


def main():
    p = Path('Data/summary/capacity_scan_results_all.csv')
    if not p.exists():
        print('No data: run capacity scan first.')
        return
    df = pd.read_csv(p)
    # Aggregate by shape, total_capacity, load
    key = ['shape', 'total_capacity', 'load_multiplier', 'algorithm']
    agg = df.groupby(key).agg(
        n=('mean_reward','size'),
        mean_reward=('mean_reward','mean'),
        crash_rate=('crash_rate','mean')
    ).reset_index()

    for load in sorted(agg['load_multiplier'].unique()):
        print(f"\n=== Load {load}× ===")
        for shape in sorted(agg['shape'].unique()):
            sub = agg[(agg['shape']==shape) & (agg['load_multiplier']==load)]
            k10 = sub[sub['total_capacity']==10]
            k30 = sub[sub['total_capacity']==30]
            if k10.empty or k30.empty:
                continue
            # Focus on algorithms that appear in both K
            common_algos = set(k10['algorithm']) & set(k30['algorithm'])
            if not common_algos:
                continue
            print(f"-- {shape} --")
            for algo in sorted(common_algos):
                a10 = k10[k10['algorithm']==algo].iloc[0]
                a30 = k30[k30['algorithm']==algo].iloc[0]
                delta_reward = a30['mean_reward'] - a10['mean_reward']
                delta_crash = a30['crash_rate'] - a10['crash_rate']
                print(f"{algo:9s}: K10 r={a10['mean_reward']:.1f}, c={a10['crash_rate']:.2f} | "
                      f"K30 r={a30['mean_reward']:.1f}, c={a30['crash_rate']:.2f} | "
                      f"Δr={delta_reward:.1f}, Δc={delta_crash:.2f}")


if __name__ == '__main__':
    main()

