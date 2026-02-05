"""
Bootstrap Confidence Intervals for Key Comparisons
Generates percentile bootstrap CIs for mean differences on
  - Inverted vs Reverse pyramid (same total capacity)
  - A2C vs PPO on viable configurations (total_capacity <= 25)

Outputs a markdown report under Analysis/statistical_reports.
"""

import json
from pathlib import Path
from typing import Tuple, Dict, List
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_FILE = PROJECT_ROOT / 'Data' / 'summary' / 'comprehensive_experiments_data.json'
OUT_DIR = PROJECT_ROOT / 'Analysis' / 'statistical_reports'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / 'bootstrap_ci_results.md'


def bootstrap_mean_diff(x: np.ndarray, y: np.ndarray, n_boot: int = 10000, seed: int = 42) -> Tuple[float, Tuple[float, float]]:
    """Bootstrap the mean difference (x - y).
    Returns (point_estimate, (ci_low, ci_high)).
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    est = float(np.mean(x) - np.mean(y))
    if len(x) == 0 or len(y) == 0:
        return est, (np.nan, np.nan)

    boot_stats = []
    for _ in range(n_boot):
        xb = rng.choice(x, size=len(x), replace=True)
        yb = rng.choice(y, size=len(y), replace=True)
        boot_stats.append(np.mean(xb) - np.mean(yb))
    ci_low, ci_high = np.percentile(boot_stats, [2.5, 97.5])
    return est, (float(ci_low), float(ci_high))


def load_dataframe() -> pd.DataFrame:
    with open(DATA_FILE, 'r') as f:
        data = json.load(f)
    return pd.DataFrame(data['experiments'])


def run_bootstrap(df: pd.DataFrame) -> Dict[str, Dict]:
    results: Dict[str, Dict] = {}

    # 1) Inverted vs Reverse (A2C+PPO), same capacity if available
    inv = df[(df.get('config_name') == 'inverted_pyramid') & df['algorithm'].isin(['A2C', 'PPO'])]
    rev = df[(df.get('config_name') == 'reverse_pyramid') & df['algorithm'].isin(['A2C', 'PPO'])]
    if len(inv) and len(rev):
        est, ci = bootstrap_mean_diff(inv['mean_reward'].values, rev['mean_reward'].values)
        results['inverted_vs_reverse'] = {
            'n_inv': int(len(inv)),
            'n_rev': int(len(rev)),
            'point_estimate': float(est),
            'ci_low': float(ci[0]),
            'ci_high': float(ci[1])
        }

    # 2) A2C vs PPO on viable (total_capacity <= 25)
    viable = df[df['total_capacity'] <= 25]
    a2c = viable[viable['algorithm'] == 'A2C']
    ppo = viable[viable['algorithm'] == 'PPO']
    if len(a2c) and len(ppo):
        est, ci = bootstrap_mean_diff(a2c['mean_reward'].values, ppo['mean_reward'].values)
        results['a2c_vs_ppo_viable'] = {
            'n_a2c': int(len(a2c)),
            'n_ppo': int(len(ppo)),
            'point_estimate': float(est),
            'ci_low': float(ci[0]),
            'ci_high': float(ci[1])
        }

    return results


def to_markdown(results: Dict[str, Dict]) -> str:
    lines: List[str] = []
    lines.append('# Bootstrap CI Results')
    lines.append('')
    lines.append('Confidence level: 95% (percentile)')
    lines.append('Bootstraps: 10,000')
    lines.append('')

    if 'inverted_vs_reverse' in results:
        r = results['inverted_vs_reverse']
        lines.append('## Inverted vs Reverse (A2C+PPO)')
        lines.append(f"n_inv={r['n_inv']}, n_rev={r['n_rev']}")
        lines.append(f"Mean difference (inv - rev): {r['point_estimate']:.2f} \n95% CI: [{r['ci_low']:.2f}, {r['ci_high']:.2f}]")
        lines.append('')

    if 'a2c_vs_ppo_viable' in results:
        r = results['a2c_vs_ppo_viable']
        lines.append('## A2C vs PPO (Viable, total_capacity ≤ 25)')
        lines.append(f"n_A2C={r['n_a2c']}, n_PPO={r['n_ppo']}")
        lines.append(f"Mean difference (A2C - PPO): {r['point_estimate']:.2f} \n95% CI: [{r['ci_low']:.2f}, {r['ci_high']:.2f}]")
        lines.append('')

    if not results:
        lines.append('No comparable groups found in the dataset.')

    return '\n'.join(lines)


if __name__ == '__main__':
    df = load_dataframe()
    results = run_bootstrap(df)
    report = to_markdown(results)
    with open(OUT_FILE, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"✅ Bootstrap report saved to: {OUT_FILE}")

