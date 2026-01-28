"""
Stability Proxy Analysis

Reads structural 5x results JSON files and aggregates stability proxies
added to evaluation output, then produces simple scatter plots:
 - mean_drift_l1 vs crash_rate
 - mean_max_load_rate vs mean_reward

Outputs:
 - Analysis/statistical_reports/stability_proxies_summary.csv
 - Analysis/figures/stability_drift_vs_crash.png
 - Analysis/figures/stability_load_vs_reward.png
"""

import json
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).parent.parent.parent
STRUCT_DIR = PROJECT_ROOT / 'Data' / 'ablation_studies' / 'structural_5x_load'
OUT_DIR = PROJECT_ROOT / 'Analysis' / 'statistical_reports'
FIG_DIR = PROJECT_ROOT / 'Analysis' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def collect_results() -> pd.DataFrame:
    rows: List[Dict] = []
    if not STRUCT_DIR.exists():
        return pd.DataFrame(rows)
    for config_dir in STRUCT_DIR.iterdir():
        if not config_dir.is_dir():
            continue
        for f in config_dir.glob('*.json'):
            try:
                with open(f, 'r') as fp:
                    data = json.load(fp)
                row = {
                    'config_type': data.get('config_type'),
                    'algorithm': data.get('algorithm'),
                    'seed': data.get('seed'),
                    'total_capacity': data.get('total_capacity'),
                    'mean_reward': data.get('mean_reward'),
                    'std_reward': data.get('std_reward'),
                    'crash_rate': data.get('crash_rate'),
                    'mean_drift_l1': data.get('mean_drift_l1', np.nan),
                    'mean_lyapunov': data.get('mean_lyapunov', np.nan),
                    'mean_lyapunov_drift': data.get('mean_lyapunov_drift', np.nan),
                    'mean_safe_ratio': data.get('mean_safe_ratio', np.nan),
                    'mean_max_load_rate': data.get('mean_max_load_rate', np.nan),
                }
                rows.append(row)
            except Exception:
                continue
    return pd.DataFrame(rows)


def plot_scatter(df: pd.DataFrame, x: str, y: str, title: str, path: Path):
    if df.empty:
        print('No data to plot.')
        return
    plt.figure(figsize=(6,4))
    for algo in sorted(df['algorithm'].dropna().unique()):
        sub = df[df['algorithm'] == algo]
        plt.scatter(sub[x], sub[y], label=str(algo), alpha=0.7)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    print(f"✅ Saved figure to {path}")


def main():
    df = collect_results()
    out_csv = OUT_DIR / 'stability_proxies_summary.csv'
    df.to_csv(out_csv, index=False)
    print(f"✅ Saved summary to {out_csv}")

    # Plots
    plot_scatter(df, 'mean_drift_l1', 'crash_rate', 'Drift L1 vs Crash Rate', FIG_DIR / 'stability_drift_vs_crash.png')
    plot_scatter(df, 'mean_max_load_rate', 'mean_reward', 'Max Load Rate vs Mean Reward', FIG_DIR / 'stability_load_vs_reward.png')


if __name__ == '__main__':
    main()

