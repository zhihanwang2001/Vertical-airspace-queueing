"""
Analyze Priority 1: Load Sensitivity Analysis

This script analyzes the capacity paradox emergence across different load levels.
Key questions:
1. At what load level does K=10 begin to outperform K=30?
2. Is the transition gradual or abrupt?
3. What is the effect size at each load level?

Expected data: loads ∈ {3,4,8,9,10} with existing {5,6,7} data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple


class LoadSensitivityAnalyzer:
    """Analyzes capacity paradox emergence across load levels."""

    def __init__(self, data_dir: str = "Data/summary"):
        self.data_dir = Path(data_dir)
        self.results = {}

    def load_all_capacity_data(self) -> pd.DataFrame:
        """Load all capacity scan data including existing and new experiments."""
        print("Loading capacity scan data...")

        # List of expected CSV files
        csv_files = [
            "capacity_scan_results_uniform_3_4.csv",      # New Priority 1
            "capacity_scan_results_uniform_8_9_10.csv",   # New Priority 1
            "capacity_scan_results_uniform.csv",          # Existing 5× data
            "capacity_scan_results_uniform_15_25.csv",    # Existing data if available
        ]

        all_data = []
        for csv_file in csv_files:
            file_path = self.data_dir / csv_file
            if file_path.exists():
                df = pd.read_csv(file_path)
                all_data.append(df)
                print(f"  ✅ Loaded: {csv_file} ({len(df)} rows)")
            else:
                print(f"  ⚠️  Not found: {csv_file}")

        if not all_data:
            raise FileNotFoundError("No capacity scan data files found")

        # Combine all data
        df_combined = pd.concat(all_data, ignore_index=True)
        print(f"\n📊 Total rows: {len(df_combined)}")
        print(f"📊 Load levels: {sorted(df_combined['load_multiplier'].unique())}")
        print(f"📊 Capacities: {sorted(df_combined['total_capacity'].unique())}")

        return df_combined

    def identify_paradox_emergence(self, df: pd.DataFrame) -> Dict:
        """Identify at what load level K=10 begins to outperform K=30."""
        print("\n" + "="*60)
        print("CAPACITY PARADOX EMERGENCE ANALYSIS")
        print("="*60)

        # Filter for K=10 and K=30, DRL algorithms only
        df_filtered = df[df['total_capacity'].isin([10, 30])]
        df_filtered = df_filtered[df_filtered['algorithm'].isin(['A2C', 'PPO'])]

        # Group by load and capacity
        grouped = df_filtered.groupby(['load_multiplier', 'total_capacity']).agg({
            'mean_reward': ['mean', 'std', 'count'],
            'crash_rate': 'mean'
        }).reset_index()

        # Flatten column names
        grouped.columns = ['load', 'capacity', 'reward_mean', 'reward_std', 'n', 'crash_rate']

        # Pivot to compare K=10 vs K=30
        results = []
        for load in sorted(grouped['load'].unique()):
            load_data = grouped[grouped['load'] == load]
            k10 = load_data[load_data['capacity'] == 10]
            k30 = load_data[load_data['capacity'] == 30]

            if k10.empty or k30.empty:
                continue

            k10_reward = k10['reward_mean'].values[0]
            k30_reward = k30['reward_mean'].values[0]
            k10_crash = k10['crash_rate'].values[0]
            k30_crash = k30['crash_rate'].values[0]

            # Calculate difference and percentage
            diff = k10_reward - k30_reward
            pct_diff = (diff / k30_reward * 100) if k30_reward != 0 else 0

            # Determine winner
            winner = "K=10" if k10_reward > k30_reward else "K=30"

            results.append({
                'load': load,
                'k10_reward': k10_reward,
                'k30_reward': k30_reward,
                'k10_crash': k10_crash,
                'k30_crash': k30_crash,
                'difference': diff,
                'pct_difference': pct_diff,
                'winner': winner
            })

        results_df = pd.DataFrame(results)

        # Print results table
        print("\nLoad-by-Load Comparison:")
        print("-" * 80)
        print(f"{'Load':>6} | {'K=10 Reward':>12} | {'K=30 Reward':>12} | {'Difference':>12} | {'Winner':>8}")
        print("-" * 80)
        for _, row in results_df.iterrows():
            print(f"{row['load']:>6.1f} | {row['k10_reward']:>12.1f} | {row['k30_reward']:>12.1f} | "
                  f"{row['difference']:>12.1f} | {row['winner']:>8}")

        # Identify transition point
        transition_loads = results_df[results_df['winner'] == 'K=10']['load'].values
        if len(transition_loads) > 0:
            transition_point = transition_loads.min()
            print(f"\n🎯 Capacity paradox emerges at load ≥ {transition_point}×")
        else:
            print("\n⚠️  Capacity paradox not observed in tested load range")
            transition_point = None

        return {
            'results_df': results_df,
            'transition_point': transition_point
        }

    def perform_statistical_tests(self, df: pd.DataFrame) -> Dict:
        """Perform ANOVA and effect size calculations."""
        print("\n" + "="*60)
        print("STATISTICAL ANALYSIS")
        print("="*60)

        # Filter for K=10 and K=30, DRL algorithms only
        df_filtered = df[df['total_capacity'].isin([10, 30])]
        df_filtered = df_filtered[df_filtered['algorithm'].isin(['A2C', 'PPO'])]

        results = {}
        for load in sorted(df_filtered['load_multiplier'].unique()):
            load_data = df_filtered[df_filtered['load_multiplier'] == load]
            k10_rewards = load_data[load_data['total_capacity'] == 10]['mean_reward'].values
            k30_rewards = load_data[load_data['total_capacity'] == 30]['mean_reward'].values

            if len(k10_rewards) == 0 or len(k30_rewards) == 0:
                continue

            # Perform t-test
            t_stat, p_value = stats.ttest_ind(k10_rewards, k30_rewards)

            # Calculate Cohen's d
            pooled_std = np.sqrt((np.std(k10_rewards, ddof=1)**2 + np.std(k30_rewards, ddof=1)**2) / 2)
            cohens_d = (np.mean(k10_rewards) - np.mean(k30_rewards)) / pooled_std if pooled_std > 0 else 0

            results[load] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'k10_mean': np.mean(k10_rewards),
                'k30_mean': np.mean(k30_rewards),
                'significant': p_value < 0.05
            }

            # Print results
            sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            print(f"\nLoad {load}×:")
            print(f"  K=10 mean: {np.mean(k10_rewards):.1f}")
            print(f"  K=30 mean: {np.mean(k30_rewards):.1f}")
            print(f"  t-statistic: {t_stat:.3f}")
            print(f"  p-value: {p_value:.6f} {sig_marker}")
            print(f"  Cohen's d: {cohens_d:.3f}")

        return results

    def create_visualizations(self, df: pd.DataFrame, output_dir: str = "Analysis/figures"):
        """Generate visualizations for capacity paradox."""
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Filter data
        df_filtered = df[df['total_capacity'].isin([10, 30])]
        df_filtered = df_filtered[df_filtered['algorithm'].isin(['A2C', 'PPO'])]

        # Aggregate by load and capacity
        agg_data = df_filtered.groupby(['load_multiplier', 'total_capacity']).agg({
            'mean_reward': ['mean', 'std'],
            'crash_rate': 'mean'
        }).reset_index()
        agg_data.columns = ['load', 'capacity', 'reward_mean', 'reward_std', 'crash_rate']

        # Figure 1: Reward vs Load
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        for capacity in [10, 30]:
            data = agg_data[agg_data['capacity'] == capacity]
            ax1.plot(data['load'], data['reward_mean'], marker='o', linewidth=2,
                    label=f'K={capacity}', markersize=8)
            ax1.fill_between(data['load'],
                            data['reward_mean'] - data['reward_std'],
                            data['reward_mean'] + data['reward_std'],
                            alpha=0.2)

        ax1.set_xlabel('Load Multiplier', fontsize=12)
        ax1.set_ylabel('Mean Reward', fontsize=12)
        ax1.set_title('Capacity Paradox: Reward vs Load', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)

        # Figure 2: Crash Rate vs Load
        for capacity in [10, 30]:
            data = agg_data[agg_data['capacity'] == capacity]
            ax2.plot(data['load'], data['crash_rate'] * 100, marker='s', linewidth=2,
                    label=f'K={capacity}', markersize=8)

        ax2.set_xlabel('Load Multiplier', fontsize=12)
        ax2.set_ylabel('Crash Rate (%)', fontsize=12)
        ax2.set_title('Crash Rate vs Load', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = output_path / 'capacity_paradox_load_sensitivity.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {fig_path}")
        plt.close()


def main():
    """Main analysis workflow."""
    analyzer = LoadSensitivityAnalyzer()

    # Load data
    df = analyzer.load_all_capacity_data()

    # Identify paradox emergence
    paradox_results = analyzer.identify_paradox_emergence(df)

    # Perform statistical tests
    stats_results = analyzer.perform_statistical_tests(df)

    # Create visualizations
    analyzer.create_visualizations(df)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Transition point: {paradox_results['transition_point']}×")
    print("Check Analysis/figures/ for visualizations")


if __name__ == '__main__':
    main()

