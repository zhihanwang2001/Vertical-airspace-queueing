"""
Analyze Local Synced Results (2026-01-16)

Inputs (local paths):
 - Results/remote_sync/2026-01-16/structural_5x_load/**/**/*_results.json
 - Results/remote_sync/2026-01-16/capacity_summary/capacity_scan_summary.csv

Outputs:
 - Analysis/statistical_reports/structural_5x_per_seed.csv
 - Analysis/statistical_reports/structural_5x_group_summary.csv
 - Analysis/statistical_reports/structural_5x_stats.md
 - Analysis/statistical_reports/capacity_scan_summary_report.md
 - Analysis/figures/structural_reward_bars.png
 - Analysis/figures/structural_reward_box.png
 - Analysis/figures/structural_stability_scatter.png
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SYNC_ROOT = PROJECT_ROOT / 'Results' / 'remote_sync' / '2026-01-16'
STRUCT_DIR = SYNC_ROOT / 'structural_5x_load'
CAP_SUMMARY = SYNC_ROOT / 'capacity_summary' / 'capacity_scan_summary.csv'
OUT_DIR = PROJECT_ROOT / 'Analysis' / 'statistical_reports'
FIG_DIR = PROJECT_ROOT / 'Analysis' / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _collect_structural() -> pd.DataFrame:
    rows: List[Dict] = []
    for shape_dir in ['inverted_pyramid', 'normal_pyramid']:
        d = STRUCT_DIR / shape_dir
        if not d.exists():
            continue
        for f in sorted(d.glob('*_results.json')):
            try:
                data = json.loads(f.read_text())
            except Exception:
                continue
            rows.append({
                'shape': data.get('config_type', shape_dir),
                'algorithm': data.get('algorithm'),
                'seed': int(data.get('seed', -1)),
                'mean_reward': float(data.get('mean_reward', np.nan)),
                'std_reward': float(data.get('std_reward', np.nan)),
                'crash_rate': float(data.get('crash_rate', np.nan)),
                'mean_episode_length': float(data.get('mean_episode_length', np.nan)),
                'mean_max_load_rate': float(data.get('mean_max_load_rate', np.nan)),
                'mean_lyapunov': float(data.get('mean_lyapunov', np.nan)),
                'mean_drift_l1': float(data.get('mean_drift_l1', np.nan)),
            })
    return pd.DataFrame(rows)


def _save_structural_tables(df: pd.DataFrame) -> Tuple[Path, Path]:
    per_seed_path = OUT_DIR / 'structural_5x_per_seed.csv'
    df.to_csv(per_seed_path, index=False)

    grp = df.groupby(['shape', 'algorithm']).agg(
        n=('mean_reward', 'size'),
        mean_reward_mean=('mean_reward', 'mean'),
        mean_reward_std=('mean_reward', 'std'),
        crash_rate_mean=('crash_rate', 'mean'),
        mean_len_mean=('mean_episode_length', 'mean'),
        max_load_mean=('mean_max_load_rate', 'mean'),
    ).reset_index()
    group_path = OUT_DIR / 'structural_5x_group_summary.csv'
    grp.to_csv(group_path, index=False)
    return per_seed_path, group_path


def _structural_stats(df: pd.DataFrame) -> Path:
    lines: List[str] = []
    lines.append('# Structural 5x Stats')
    lines.append('')
    for algo in ['A2C', 'PPO']:
        inv = df[(df['shape'] == 'inverted_pyramid') & (df['algorithm'] == algo)]['mean_reward'].values
        norm = df[(df['shape'] == 'normal_pyramid') & (df['algorithm'] == algo)]['mean_reward'].values
        if len(inv) and len(norm):
            t, p = stats.ttest_ind(inv, norm, equal_var=False)
            d = _cohens_d(inv, norm)
            lines.append(f'## {algo}: Inverted vs Normal')
            lines.append(f'- n_inv={len(inv)}, n_norm={len(norm)}')
            lines.append(f'- mean_inv={np.mean(inv):.2f} ± {np.std(inv):.2f}')
            lines.append(f'- mean_norm={np.mean(norm):.2f} ± {np.std(norm):.2f}')
            lines.append(f'- diff={np.mean(inv)-np.mean(norm):.2f} (t={t:.3f}, p={p:.4g}, d={d:.3f})')
            lines.append('')

    # combined across algorithms
    inv_all = df[df['shape'] == 'inverted_pyramid']['mean_reward'].values
    norm_all = df[df['shape'] == 'normal_pyramid']['mean_reward'].values
    if len(inv_all) and len(norm_all):
        t, p = stats.ttest_ind(inv_all, norm_all, equal_var=False)
        d = _cohens_d(inv_all, norm_all)
        lines.append('## Combined: Inverted vs Normal (A2C+PPO)')
        lines.append(f'- n_inv={len(inv_all)}, n_norm={len(norm_all)}')
        lines.append(f'- mean_inv={np.mean(inv_all):.2f} ± {np.std(inv_all):.2f}')
        lines.append(f'- mean_norm={np.mean(norm_all):.2f} ± {np.std(norm_all):.2f}')
        lines.append(f'- diff={np.mean(inv_all)-np.mean(norm_all):.2f} (t={t:.3f}, p={p:.4g}, d={d:.3f})')

    # Stability proxies overview
    lines.append('')
    lines.append('## Stability Proxies (means)')
    stab = df.groupby(['shape', 'algorithm']).agg(
        max_load_mean=('mean_max_load_rate', 'mean'),
        drift_l1_mean=('mean_drift_l1', 'mean'),
        lyapunov_mean=('mean_lyapunov', 'mean'),
        crash_mean=('crash_rate', 'mean'),
    ).reset_index()
    lines.append(stab.to_csv(index=False))

    out = OUT_DIR / 'structural_5x_stats.md'
    out.write_text('\n'.join(lines))
    return out


def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    if nx + ny <= 2:
        return float('nan')
    pooled = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled == 0:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / pooled)


def _structural_plots(df: pd.DataFrame) -> Tuple[Path, Path, Path]:
    # Bar plot with error bars
    fig1 = FIG_DIR / 'structural_reward_bars.png'
    plt.figure(figsize=(6,4))
    grp = df.groupby(['shape','algorithm'])['mean_reward'].agg(['mean','std','count']).reset_index()
    xlabels = ['Inv-A2C','Inv-PPO','Norm-A2C','Norm-PPO']
    means = [
        grp[(grp['shape']=='inverted_pyramid') & (grp['algorithm']=='A2C')]['mean'].values[0],
        grp[(grp['shape']=='inverted_pyramid') & (grp['algorithm']=='PPO')]['mean'].values[0],
        grp[(grp['shape']=='normal_pyramid') & (grp['algorithm']=='A2C')]['mean'].values[0],
        grp[(grp['shape']=='normal_pyramid') & (grp['algorithm']=='PPO')]['mean'].values[0],
    ]
    stds = [
        grp[(grp['shape']=='inverted_pyramid') & (grp['algorithm']=='A2C')]['std'].values[0],
        grp[(grp['shape']=='inverted_pyramid') & (grp['algorithm']=='PPO')]['std'].values[0],
        grp[(grp['shape']=='normal_pyramid') & (grp['algorithm']=='A2C')]['std'].values[0],
        grp[(grp['shape']=='normal_pyramid') & (grp['algorithm']=='PPO')]['std'].values[0],
    ]
    xs = np.arange(len(xlabels))
    plt.bar(xs, means, yerr=stds, capsize=3, alpha=0.8)
    plt.xticks(xs, xlabels)
    plt.ylabel('Mean Reward')
    plt.title('Structural 5x: Reward by Shape/Algo (±1 SD)')
    plt.tight_layout()
    plt.grid(True, axis='y', alpha=0.3)
    plt.savefig(fig1, dpi=180)

    # Box plot per group
    fig2 = FIG_DIR / 'structural_reward_box.png'
    plt.figure(figsize=(6,4))
    groups = [
        df[(df['shape']=='inverted_pyramid') & (df['algorithm']=='A2C')]['mean_reward'],
        df[(df['shape']=='inverted_pyramid') & (df['algorithm']=='PPO')]['mean_reward'],
        df[(df['shape']=='normal_pyramid') & (df['algorithm']=='A2C')]['mean_reward'],
        df[(df['shape']=='normal_pyramid') & (df['algorithm']=='PPO')]['mean_reward'],
    ]
    plt.boxplot(groups, labels=xlabels, showmeans=True)
    plt.ylabel('Mean Reward')
    plt.title('Structural 5x: Reward Distribution by Group')
    plt.tight_layout()
    plt.grid(True, axis='y', alpha=0.3)
    plt.savefig(fig2, dpi=180)

    # Stability proxy scatter
    fig3 = FIG_DIR / 'structural_stability_scatter.png'
    plt.figure(figsize=(6,4))
    for algo, marker in [('A2C','o'),('PPO','s')]:
        sub = df[df['algorithm']==algo]
        plt.scatter(sub['mean_max_load_rate'], sub['mean_reward'], label=algo, alpha=0.7, marker=marker)
    plt.xlabel('mean_max_load_rate')
    plt.ylabel('mean_reward')
    plt.title('Max Load Rate vs Reward (Structural 5x)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig3, dpi=180)

    return fig1, fig2, fig3


def _capacity_report() -> Path:
    p = CAP_SUMMARY
    lines: List[str] = []
    lines.append('# Capacity Scan Summary (Local)')
    if not p.exists():
        lines.append('No capacity summary CSV found.')
        out = OUT_DIR / 'capacity_scan_summary_report.md'
        out.write_text('\n'.join(lines))
        return out
    df = pd.read_csv(p)
    def one(shape: str, K: int, algo: str) -> Tuple[float,float,int]:
        sub = df[(df['shape']==shape) & (df['total_capacity']==K) & (df['algorithm']==algo)]
        if sub.empty:
            return float('nan'), float('nan'), 0
        r = float(sub['mean_reward'].iloc[0]); c = float(sub['mean_crash_rate'].iloc[0]); n = int(sub['n'].iloc[0])
        return r, c, n
    # Focus on uniform K=10 vs 30 for both families
    lines.append('## Uniform, K=10 vs K=30, load=5x')
    for fam, algos in [('Heuristic',['FCFS','SJF','Priority','Heuristic']), ('RL',['A2C','PPO'])]:
        lines.append(f'### {fam}')
        for algo in algos:
            r10, c10, n10 = one('uniform', 10, algo if fam=='Heuristic' else algo)
            r30, c30, n30 = one('uniform', 30, algo if fam=='Heuristic' else algo)
            if n10 and n30:
                lines.append(f'- {algo}: K10 r={r10:.1f}, c={c10:.3f}; K30 r={r30:.1f}, c={c30:.3f}; Δr={r30-r10:.1f}, Δc={c30-c10:.3f}')
    out = OUT_DIR / 'capacity_scan_summary_report.md'
    out.write_text('\n'.join(lines))
    return out


def main():
    df = _collect_structural()
    per_seed, group = _save_structural_tables(df)
    stats_md = _structural_stats(df)
    f1, f2, f3 = _structural_plots(df)
    cap_md = _capacity_report()
    print('✅ Wrote:', per_seed)
    print('✅ Wrote:', group)
    print('✅ Wrote:', stats_md)
    print('✅ Wrote:', cap_md)
    print('✅ Saved figures:', f1, f2, f3)


if __name__ == '__main__':
    main()

