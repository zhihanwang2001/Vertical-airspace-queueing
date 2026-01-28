"""
Prepare publication-ready artifacts from local synced results.

Inputs
- Structural per-seed CSV: Analysis/statistical_reports/structural_5x_per_seed.csv
- Structural group CSV:   Analysis/statistical_reports/structural_5x_group_summary.csv
- Capacity summary CSV:   Data/summary/capacity_scan_summary.csv

Outputs
- Figures (PNG+PDF+SVG):
  Analysis/figures/structural_reward_bars.{png,pdf,svg}
  Analysis/figures/structural_reward_box.{png,pdf,svg}
  Analysis/figures/structural_stability_scatter.{png,pdf,svg}
  Analysis/figures/capacity_uniform_k10k30_crash.{png,pdf,svg}
  Analysis/figures/capacity_uniform_k10k30_reward.{png,pdf,svg}
- Tables (CSV + LaTeX):
  Analysis/statistical_reports/structural_group_table.{csv,tex}
  Analysis/statistical_reports/capacity_uniform_k10k30_crash.{csv,tex}
  Analysis/statistical_reports/capacity_uniform_k10k30_reward.{csv,tex}
- Numbers digest: Analysis/statistical_reports/paper_numbers.md
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = PROJECT_ROOT / 'Analysis' / 'figures'
REP_DIR = PROJECT_ROOT / 'Analysis' / 'statistical_reports'
FIG_DIR.mkdir(parents=True, exist_ok=True)
REP_DIR.mkdir(parents=True, exist_ok=True)

STRUCT_PER_SEED = REP_DIR / 'structural_5x_per_seed.csv'
STRUCT_GROUP = REP_DIR / 'structural_5x_group_summary.csv'
CAP_SUMMARY = PROJECT_ROOT / 'Data' / 'summary' / 'capacity_scan_summary.csv'


def save_all_formats(figpath_base: Path):
    for ext in ['png', 'pdf', 'svg']:
        plt.savefig(figpath_base.with_suffix('.'+ext), dpi=180 if ext=='png' else None, bbox_inches='tight')


def plot_structural_bars(df: pd.DataFrame):
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
    plt.bar(xs, means, yerr=stds, capsize=3, alpha=0.85)
    plt.xticks(xs, xlabels)
    plt.ylabel('Mean Reward')
    plt.title('Structural 5x: Reward by Shape/Algo (±1 SD)')
    plt.grid(True, axis='y', alpha=0.3)
    save_all_formats(FIG_DIR / 'structural_reward_bars')


def plot_structural_box(df: pd.DataFrame):
    plt.figure(figsize=(6,4))
    xlabels = ['Inv-A2C','Inv-PPO','Norm-A2C','Norm-PPO']
    groups = [
        df[(df['shape']=='inverted_pyramid') & (df['algorithm']=='A2C')]['mean_reward'],
        df[(df['shape']=='inverted_pyramid') & (df['algorithm']=='PPO')]['mean_reward'],
        df[(df['shape']=='normal_pyramid') & (df['algorithm']=='A2C')]['mean_reward'],
        df[(df['shape']=='normal_pyramid') & (df['algorithm']=='PPO')]['mean_reward'],
    ]
    plt.boxplot(groups, tick_labels=xlabels, showmeans=True)
    plt.ylabel('Mean Reward')
    plt.title('Structural 5x: Reward Distribution by Group')
    plt.grid(True, axis='y', alpha=0.3)
    save_all_formats(FIG_DIR / 'structural_reward_box')


def plot_structural_stability(df: pd.DataFrame):
    plt.figure(figsize=(6,4))
    for algo, marker in [('A2C','o'),('PPO','s')]:
        sub = df[df['algorithm']==algo]
        plt.scatter(sub['mean_max_load_rate'], sub['mean_reward'], label=algo, alpha=0.7, marker=marker)
    plt.xlabel('mean_max_load_rate')
    plt.ylabel('mean_reward')
    plt.title('Max Load Rate vs Reward (Structural 5x)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_all_formats(FIG_DIR / 'structural_stability_scatter')


def tables_structural(group_csv: Path):
    df = pd.read_csv(group_csv)
    # Save normalized/renamed columns for paper
    out_csv = REP_DIR / 'structural_group_table.csv'
    df_out = df[['shape','algorithm','n','mean_reward_mean','mean_reward_std','crash_rate_mean','mean_len_mean','max_load_mean']].copy()
    df_out.rename(columns={
        'shape':'Shape','algorithm':'Algorithm','n':'N','mean_reward_mean':'MeanReward','mean_reward_std':'StdReward',
        'crash_rate_mean':'CrashRate','mean_len_mean':'MeanLen','max_load_mean':'MaxLoad'
    }, inplace=True)
    df_out.to_csv(out_csv, index=False)

    # LaTeX table
    tex = df_out.to_latex(index=False, float_format=lambda x: f"{x:.2f}")
    (REP_DIR / 'structural_group_table.tex').write_text(tex)


def plot_capacity_from_summary(df: pd.DataFrame):
    keep = ['FCFS','SJF','Priority','Heuristic','A2C','PPO']
    sub = df[(df['shape']=='uniform') & (df['load_multiplier']==5.0) & (df['algorithm'].isin(keep))]
    p_crash = sub.pivot_table(index='algorithm', columns='total_capacity', values='mean_crash_rate')
    p_rew = sub.pivot_table(index='algorithm', columns='total_capacity', values='mean_reward')
    xs = np.arange(len(keep))
    barw = 0.35

    plt.figure(figsize=(7,4))
    plt.bar(xs - barw/2, p_crash[10].reindex(keep).values, width=barw, label='K=10')
    plt.bar(xs + barw/2, p_crash[30].reindex(keep).values, width=barw, label='K=30')
    plt.xticks(xs, keep)
    plt.ylabel('Crash Rate')
    plt.title('Uniform, Load 5×: Crash Rate (K=10 vs K=30)')
    plt.ylim(0,1.05)
    plt.grid(True, axis='y', alpha=0.3)
    plt.legend()
    save_all_formats(FIG_DIR / 'capacity_uniform_k10k30_crash')

    plt.figure(figsize=(7,4))
    plt.bar(xs - barw/2, np.clip(p_rew[10].reindex(keep).values, 1e-6, None), width=barw, label='K=10')
    plt.bar(xs + barw/2, np.clip(p_rew[30].reindex(keep).values, 1e-6, None), width=barw, label='K=30')
    plt.xticks(xs, keep)
    plt.ylabel('Mean Reward')
    plt.title('Uniform, Load 5×: Mean Reward (K=10 vs K=30)')
    plt.grid(True, axis='y', alpha=0.3)
    plt.legend()
    save_all_formats(FIG_DIR / 'capacity_uniform_k10k30_reward')

    # Export compact tables
    crash_tbl = sub.pivot_table(index='algorithm', columns='total_capacity', values='mean_crash_rate').reindex(keep)
    crash_tbl.to_csv(REP_DIR / 'capacity_uniform_k10k30_crash.csv')
    (REP_DIR / 'capacity_uniform_k10k30_crash.tex').write_text(crash_tbl.to_latex(float_format=lambda x: f"{x:.3f}"))

    rew_tbl = sub.pivot_table(index='algorithm', columns='total_capacity', values='mean_reward').reindex(keep)
    rew_tbl.to_csv(REP_DIR / 'capacity_uniform_k10k30_reward.csv')
    (REP_DIR / 'capacity_uniform_k10k30_reward.tex').write_text(rew_tbl.to_latex(float_format=lambda x: f"{x:.1f}"))


def write_numbers_digest(per_seed_csv: Path, cap_summary_csv: Path):
    df = pd.read_csv(per_seed_csv)
    inv = df[df['shape']=='inverted_pyramid']['mean_reward'].mean()
    norm = df[df['shape']=='normal_pyramid']['mean_reward'].mean()
    diff = inv - norm
    cap = pd.read_csv(cap_summary_csv)
    get = lambda algo,K: cap[(cap['shape']=='uniform') & (cap['total_capacity']==K) & (cap['algorithm']==algo) & (cap['load_multiplier']==5.0)]
    a2c10 = get('A2C',10); a2c30 = get('A2C',30)
    ppo10 = get('PPO',10); ppo30 = get('PPO',30)
    lines = []
    lines.append('# Paper Numbers Digest')
    lines.append(f'- Structural mean (Inv vs Norm, A2C+PPO combined): inv={inv:.2f}, norm={norm:.2f}, diff={diff:.2f}')
    if not a2c10.empty and not a2c30.empty:
        lines.append(f'- Capacity RL A2C: K10 r={a2c10.iloc[0]["mean_reward"]:.1f}, c={a2c10.iloc[0]["mean_crash_rate"]:.3f}; K30 r={a2c30.iloc[0]["mean_reward"]:.1f}, c={a2c30.iloc[0]["mean_crash_rate"]:.3f}')
    if not ppo10.empty and not ppo30.empty:
        lines.append(f'- Capacity RL PPO: K10 r={ppo10.iloc[0]["mean_reward"]:.1f}, c={ppo10.iloc[0]["mean_crash_rate"]:.3f}; K30 r={ppo30.iloc[0]["mean_reward"]:.1f}, c={ppo30.iloc[0]["mean_crash_rate"]:.3f}')
    (REP_DIR / 'paper_numbers.md').write_text('\n'.join(lines) + '\n')


def main():
    # Structural plots from per-seed data (vector outputs)
    per = pd.read_csv(STRUCT_PER_SEED)
    plot_structural_bars(per)
    plot_structural_box(per)
    plot_structural_stability(per)

    # Tables
    tables_structural(STRUCT_GROUP)

    # Capacity plots/tables from summary
    cap = pd.read_csv(CAP_SUMMARY)
    plot_capacity_from_summary(cap)

    # Numbers digest
    write_numbers_digest(STRUCT_PER_SEED, CAP_SUMMARY)
    print('✅ Artifacts written to', FIG_DIR, 'and', REP_DIR)


if __name__ == '__main__':
    main()

