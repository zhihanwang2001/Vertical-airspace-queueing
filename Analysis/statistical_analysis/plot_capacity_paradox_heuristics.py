"""
Plot capacity paradox from heuristic baselines.

Reads Data/summary/capacity_scan_results_all.csv and for a given load (default 5×)
plots mean_reward and crash_rate vs total_capacity K for selected heuristics
(FCFS, SJF, Priority) across shapes (inverted/reverse/uniform).

Outputs:
 - Analysis/figures/capacity_paradox_reward_heuristics.png
 - Analysis/figures/capacity_paradox_crash_heuristics.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def main(load=5.0):
    p = Path('Data/summary/capacity_scan_results_all.csv')
    if not p.exists():
        print('No combined results found. Run capacity scan script first.')
        return
    df = pd.read_csv(p)
    df = df[(df['family']=='Heuristic') & (df['load_multiplier'].round(1)==round(load,1))]

    # Aggregate
    group_cols = ['shape','total_capacity','algorithm']
    agg = df.groupby(group_cols).agg(
        mean_reward=('mean_reward','mean'),
        crash_rate=('crash_rate','mean'),
        n=('mean_reward','size')
    ).reset_index()

    shapes = ['inverted','reverse','uniform']
    algos = ['FCFS','SJF','Priority']

    # Reward plot
    plt.figure(figsize=(8,5))
    for shape in shapes:
        sub = agg[agg['shape']==shape]
        for algo in algos:
            line = sub[sub['algorithm']==algo].sort_values('total_capacity')
            if line.empty:
                continue
            plt.plot(line['total_capacity'], line['mean_reward'], marker='o', label=f'{shape}-{algo}')
    plt.xlabel('Total Capacity K')
    plt.ylabel('Mean Reward')
    plt.title(f'Heuristic Capacity Paradox (Reward) @ Load {load}×')
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    Path('Analysis/figures').mkdir(parents=True, exist_ok=True)
    out1 = Path('Analysis/figures/capacity_paradox_reward_heuristics.png')
    plt.tight_layout(); plt.savefig(out1, dpi=180)
    print(f'✅ Saved {out1}')

    # Crash plot
    plt.figure(figsize=(8,5))
    for shape in shapes:
        sub = agg[agg['shape']==shape]
        for algo in algos:
            line = sub[sub['algorithm']==algo].sort_values('total_capacity')
            if line.empty:
                continue
            plt.plot(line['total_capacity'], line['crash_rate'], marker='o', label=f'{shape}-{algo}')
    plt.xlabel('Total Capacity K')
    plt.ylabel('Crash Rate')
    plt.title(f'Heuristic Capacity Paradox (Crash) @ Load {load}×')
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    out2 = Path('Analysis/figures/capacity_paradox_crash_heuristics.png')
    plt.tight_layout(); plt.savefig(out2, dpi=180)
    print(f'✅ Saved {out2}')


if __name__ == '__main__':
    main()

