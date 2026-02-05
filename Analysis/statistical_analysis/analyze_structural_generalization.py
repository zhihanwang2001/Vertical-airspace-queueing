"""
Analyze Priority 2: Structural Comparison Generalization

This script analyzes whether the inverted pyramid's 9.5% advantage generalizes
across different load levels.

Key questions:
1. Does the 9.5% advantage persist across all loads?
2. Does effect size vary systematically with load?
3. Are there any load levels where normal pyramid performs better?

Expected data: loads ∈ {3,5,7,10} (5× from existing data, 3×,7×,10× from Priority 2)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple
import json


class StructuralGeneralizationAnalyzer:
    """Analyzes structural comparison across load levels."""

    def __init__(self, data_dir: str = "Data/ablation_studies"):
        self.data_dir = Path(data_dir)
        self.results = {}

    def load_structural_data(self) -> pd.DataFrame:
        """Load structural comparison data from all load levels."""
        print("Loading structural comparison data...")

        all_data = []

        # Load existing 5× data
        existing_5x_path = self.data_dir / "structural_5x_load"
        if existing_5x_path.exists():
            data_5x = self._load_load_level_data(existing_5x_path, load=5.0)
            if data_5x:
                all_data.extend(data_5x)
                print(f"  ✅ Loaded 5× data: {len(data_5x)} runs")

        # Load new Priority 2 data
        priority2_path = self.data_dir / "priority2_structural_generalization"
        if priority2_path.exists():
            for load in [3, 7, 10]:
                for structure in ['inverted', 'normal']:
                    load_dir = priority2_path / f"load_{load}x_{structure}"
                    if load_dir.exists():
                        data = self._load_load_level_data(load_dir, load=float(load), structure=structure)
                        if data:
                            all_data.extend(data)
                            print(f"  ✅ Loaded {load}× {structure}: {len(data)} runs")

        if not all_data:
            raise FileNotFoundError("No structural comparison data found")

        df = pd.DataFrame(all_data)
        print(f"\n📊 Total runs: {len(df)}")
        print(f"📊 Load levels: {sorted(df['load'].unique())}")
        print(f"📊 Structures: {sorted(df['structure'].unique())}")
        print(f"📊 Algorithms: {sorted(df['algorithm'].unique())}")

        return df

    def _load_load_level_data(self, directory: Path, load: float, structure: str = None) -> List[Dict]:
        """Load result files from a specific load level directory."""
        results = []
        result_files = list(directory.glob("*_results.json"))

        for result_file in result_files:
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)

                    # Extract algorithm and seed from filename
                    # Expected format: A2C_seed42_results.json
                    filename = result_file.stem
                    parts = filename.split('_')
                    algorithm = parts[0]
                    seed = int(parts[1].replace('seed', ''))

                    # Infer structure from directory name if not provided
                    if structure is None:
                        if 'inverted' in str(directory):
                            structure = 'inverted'
                        elif 'normal' in str(directory):
                            structure = 'normal'
                        else:
                            structure = 'unknown'

                    results.append({
                        'load': load,
                        'structure': structure,
                        'algorithm': algorithm,
                        'seed': seed,
                        'mean_reward': data.get('mean_reward', 0),
                        'std_reward': data.get('std_reward', 0),
                        'crash_rate': data.get('crash_rate', 0)
                    })
            except Exception as e:
                print(f"  ⚠️  Failed to load {result_file.name}: {str(e)}")

        return results

    def compare_structures_by_load(self, df: pd.DataFrame) -> Dict:
        """Compare inverted vs normal pyramid at each load level."""
        print("\n" + "="*60)
        print("STRUCTURAL COMPARISON BY LOAD LEVEL")
        print("="*60)

        results = {}
        for load in sorted(df['load'].unique()):
            load_data = df[df['load'] == load]
            inverted_rewards = load_data[load_data['structure'] == 'inverted']['mean_reward'].values
            normal_rewards = load_data[load_data['structure'] == 'normal']['mean_reward'].values

            if len(inverted_rewards) == 0 or len(normal_rewards) == 0:
                print(f"\n⚠️  Load {load}×: Insufficient data")
                continue

            # Calculate statistics
            inv_mean = np.mean(inverted_rewards)
            norm_mean = np.mean(normal_rewards)
            advantage = ((inv_mean - norm_mean) / norm_mean * 100) if norm_mean != 0 else 0

            # Perform t-test
            t_stat, p_value = stats.ttest_ind(inverted_rewards, normal_rewards)

            # Calculate Cohen's d
            pooled_std = np.sqrt((np.std(inverted_rewards, ddof=1)**2 + np.std(normal_rewards, ddof=1)**2) / 2)
            cohens_d = (inv_mean - norm_mean) / pooled_std if pooled_std > 0 else 0

            results[load] = {
                'inverted_mean': inv_mean,
                'normal_mean': norm_mean,
                'advantage_pct': advantage,
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'n_inverted': len(inverted_rewards),
                'n_normal': len(normal_rewards)
            }

            # Print results
            sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            print(f"\nLoad {load}×:")
            print(f"  Inverted mean: {inv_mean:.1f}")
            print(f"  Normal mean: {norm_mean:.1f}")
            print(f"  Advantage: {advantage:+.2f}% {'✅' if advantage > 0 else '❌'}")
            print(f"  t-statistic: {t_stat:.3f}")
            print(f"  p-value: {p_value:.6f} {sig_marker}")
            print(f"  Cohen's d: {cohens_d:.3f}")

        return results

    def create_visualizations(self, df: pd.DataFrame, comparison_results: Dict,
                            output_dir: str = "Analysis/figures"):
        """Generate visualizations for structural comparison."""
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Prepare data for plotting
        loads = sorted(comparison_results.keys())
        inverted_means = [comparison_results[load]['inverted_mean'] for load in loads]
        normal_means = [comparison_results[load]['normal_mean'] for load in loads]
        advantages = [comparison_results[load]['advantage_pct'] for load in loads]
        cohens_ds = [comparison_results[load]['cohens_d'] for load in loads]

        # Figure 1: Bar chart comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        x = np.arange(len(loads))
        width = 0.35

        ax1.bar(x - width/2, inverted_means, width, label='Inverted', color='#4CAF50', alpha=0.8)
        ax1.bar(x + width/2, normal_means, width, label='Normal', color='#F44336', alpha=0.8)
        ax1.set_xlabel('Load Multiplier', fontsize=12)
        ax1.set_ylabel('Mean Reward', fontsize=12)
        ax1.set_title('Inverted vs Normal Pyramid Performance', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'{load}×' for load in loads])
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3, axis='y')

        # Figure 2: Advantage percentage
        colors = ['#4CAF50' if adv > 0 else '#F44336' for adv in advantages]
        ax2.bar(x, advantages, color=colors, alpha=0.8)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax2.axhline(y=9.5, color='blue', linestyle='--', linewidth=1.5, label='Original 9.5%')
        ax2.set_xlabel('Load Multiplier', fontsize=12)
        ax2.set_ylabel('Advantage (%)', fontsize=12)
        ax2.set_title('Inverted Pyramid Advantage by Load', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'{load}×' for load in loads])
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        fig_path = output_path / 'structural_generalization_comparison.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {fig_path}")
        plt.close()


def main():
    """Main analysis workflow."""
    analyzer = StructuralGeneralizationAnalyzer()

    # Load data
    df = analyzer.load_structural_data()

    # Compare structures by load
    comparison_results = analyzer.compare_structures_by_load(df)

    # Create visualizations
    analyzer.create_visualizations(df, comparison_results)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("Check Analysis/figures/ for visualizations")


if __name__ == '__main__':
    main()


