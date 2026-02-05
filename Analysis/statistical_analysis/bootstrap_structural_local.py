"""
Bootstrap CIs (Local Structural 5x)

Computes percentile bootstrap 95% CIs for the mean difference
in mean_reward between Inverted and Normal pyramids per algorithm
and combined across algorithms, using per-seed results synced locally.

Inputs:
 - Analysis/statistical_reports/structural_5x_per_seed.csv

Output:
 - Analysis/statistical_reports/bootstrap_structural_local.md
"""

from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PER_SEED = PROJECT_ROOT / 'Analysis' / 'statistical_reports' / 'structural_5x_per_seed.csv'
OUT = PROJECT_ROOT / 'Analysis' / 'statistical_reports' / 'bootstrap_structural_local.md'


def bootstrap_mean_diff(x: np.ndarray, y: np.ndarray, n_boot: int = 10000, seed: int = 42) -> Tuple[float, Tuple[float, float]]:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    est = float(np.mean(x) - np.mean(y))
    if len(x) == 0 or len(y) == 0:
        return est, (np.nan, np.nan)
    bs = []
    for _ in range(n_boot):
        xb = rng.choice(x, size=len(x), replace=True)
        yb = rng.choice(y, size=len(y), replace=True)
        bs.append(np.mean(xb) - np.mean(yb))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return est, (float(lo), float(hi))


def main():
    df = pd.read_csv(PER_SEED)
    lines = []
    lines.append('# Bootstrap CI (Local Structural 5x)')
    for algo in ['A2C', 'PPO']:
        inv = df[(df['shape']=='inverted_pyramid') & (df['algorithm']==algo)]['mean_reward'].values
        norm = df[(df['shape']=='normal_pyramid') & (df['algorithm']==algo)]['mean_reward'].values
        est, (lo, hi) = bootstrap_mean_diff(inv, norm)
        lines.append(f'## {algo}: Inverted - Normal')
        lines.append(f'point_estimate = {est:.2f}')
        lines.append(f'95% CI = [{lo:.2f}, {hi:.2f}]')
        lines.append('')

    inv_all = df[df['shape']=='inverted_pyramid']['mean_reward'].values
    norm_all = df[df['shape']=='normal_pyramid']['mean_reward'].values
    est, (lo, hi) = bootstrap_mean_diff(inv_all, norm_all)
    lines.append('## Combined (A2C+PPO): Inverted - Normal')
    lines.append(f'point_estimate = {est:.2f}')
    lines.append(f'95% CI = [{lo:.2f}, {hi:.2f}]')

    OUT.write_text('\n'.join(lines) + '\n')
    print('✅ Wrote', OUT)


if __name__ == '__main__':
    main()

