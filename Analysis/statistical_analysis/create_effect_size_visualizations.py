#!/usr/bin/env python3
"""
Create visualizations for Cohen's d effect sizes and distribution separation.

This script generates:
1. Distribution plots showing complete separation (7× and 10× loads)
2. Boxplots with variance analysis across all loads
3. CV vs. load relationship graph
4. Variance analysis table

Author: Statistical Analysis
Date: 2026-01-18
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9

# Define paths
DATA_DIR = Path("Data/summary")
OUTPUT_DIR_1 = Path("Analysis/figures")
OUTPUT_DIR_2 = Path("Figures")

# Create output directories if they don't exist
OUTPUT_DIR_1.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR_2.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("Effect Size Visualization Script")
print("=" * 80)

def load_structural_data():
    """Load inverted and reverse pyramid data across multiple loads."""
    print("\n[1/5] Loading structural comparison data...")

    # Load inverted pyramid data
    inv_file = DATA_DIR / "capacity_scan_results_inverted_3_7_10.csv"
    rev_file = DATA_DIR / "capacity_scan_results_reverse_3_7_10.csv"

    if not inv_file.exists() or not rev_file.exists():
        print(f"Error: Data files not found!")
        print(f"  Inverted: {inv_file}")
        print(f"  Reverse: {rev_file}")
        return None, None

    df_inv = pd.read_csv(inv_file)
    df_rev = pd.read_csv(rev_file)

    print(f"  Loaded {len(df_inv)} inverted pyramid records")
    print(f"  Loaded {len(df_rev)} reverse pyramid records")

    return df_inv, df_rev

def calculate_statistics(df, group_col='shape', value_col='mean_reward', load_col='load_multiplier'):
    """Calculate comprehensive statistics for each group and load."""
    print("\n[2/5] Calculating statistics...")

    stats_list = []

    for load in sorted(df[load_col].unique()):
        df_load = df[df[load_col] == load]

        for group in df_load[group_col].unique():
            df_group = df_load[df_load[group_col] == group]
            rewards = df_group[value_col].values

            stats_dict = {
                'load': load,
                'group': group,
                'n': len(rewards),
                'mean': np.mean(rewards),
                'std': np.std(rewards, ddof=1),
                'min': np.min(rewards),
                'max': np.max(rewards),
                'range': np.max(rewards) - np.min(rewards),
                'cv': (np.std(rewards, ddof=1) / np.mean(rewards)) * 100
            }
            stats_list.append(stats_dict)

    stats_df = pd.DataFrame(stats_list)
    print(f"  Calculated statistics for {len(stats_df)} groups")

    return stats_df

def plot_distributions(df, load_levels=[7.0, 10.0]):
    """Create distribution plots showing complete separation."""
    print("\n[3/5] Creating distribution plots...")

    fig, axes = plt.subplots(1, len(load_levels), figsize=(12, 5))
    if len(load_levels) == 1:
        axes = [axes]

    for idx, load in enumerate(load_levels):
        ax = axes[idx]
        df_load = df[df['load_multiplier'] == load]

        # Get data for each group
        inv_data = df_load[df_load['shape'] == 'inverted']['mean_reward'].values
        rev_data = df_load[df_load['shape'] == 'reverse']['mean_reward'].values

        # Plot distributions
        ax.hist(inv_data, bins=15, alpha=0.6, label='Inverted Pyramid', color='#2E86AB', edgecolor='black')
        ax.hist(rev_data, bins=15, alpha=0.6, label='Reverse Pyramid', color='#A23B72', edgecolor='black')

        # Add vertical lines for means
        ax.axvline(np.mean(inv_data), color='#2E86AB', linestyle='--', linewidth=2, label=f'Inv Mean: {np.mean(inv_data):.0f}')
        ax.axvline(np.mean(rev_data), color='#A23B72', linestyle='--', linewidth=2, label=f'Rev Mean: {np.mean(rev_data):.0f}')

        ax.set_xlabel('Mean Reward', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'Load {load:.0f}× (Complete Separation)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save to both directories
    for output_dir in [OUTPUT_DIR_1, OUTPUT_DIR_2]:
        filepath = output_dir / "effect_size_distributions.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  Saved: {filepath}")

    plt.close()

def plot_boxplots(df):
    """Create boxplots across all load levels."""
    print("\n[4/5] Creating boxplots...")

    fig, ax = plt.subplots(figsize=(10, 6))

    # Prepare data for boxplot
    df['load_str'] = df['load_multiplier'].apply(lambda x: f'{x:.0f}×')

    # Create boxplot
    sns.boxplot(data=df, x='load_str', y='mean_reward', hue='shape',
                palette={'inverted': '#2E86AB', 'reverse': '#A23B72'}, ax=ax)

    ax.set_xlabel('Load Multiplier', fontsize=11)
    ax.set_ylabel('Mean Reward', fontsize=11)
    ax.set_title('Reward Distribution Across Load Levels', fontsize=12, fontweight='bold')
    ax.legend(title='Structure', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save to both directories
    for output_dir in [OUTPUT_DIR_1, OUTPUT_DIR_2]:
        filepath = output_dir / "effect_size_boxplots.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  Saved: {filepath}")

    plt.close()

def plot_cv_vs_load(stats_df):
    """Create CV vs load relationship graph."""
    print("\n[5/5] Creating CV vs Load graph...")

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot CV for each group
    for group in stats_df['group'].unique():
        df_group = stats_df[stats_df['group'] == group]
        color = '#2E86AB' if group == 'inverted' else '#A23B72'
        ax.plot(df_group['load'], df_group['cv'], marker='o', linewidth=2,
                markersize=8, label=f'{group.capitalize()} Pyramid', color=color)

    ax.set_xlabel('Load Multiplier', fontsize=11)
    ax.set_ylabel('Coefficient of Variation (%)', fontsize=11)
    ax.set_title('Variance Decreases with Load (CV vs Load)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Log scale to show the dramatic decrease

    plt.tight_layout()

    # Save to both directories
    for output_dir in [OUTPUT_DIR_1, OUTPUT_DIR_2]:
        filepath = output_dir / "effect_size_cv_vs_load.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  Saved: {filepath}")

    plt.close()

def main():
    """Main execution function."""
    # Load data
    df_inv, df_rev = load_structural_data()
    if df_inv is None or df_rev is None:
        print("Error: Could not load data. Exiting.")
        return

    # Combine data
    df_combined = pd.concat([df_inv, df_rev], ignore_index=True)

    # Calculate statistics
    stats_df = calculate_statistics(df_combined)

    # Create visualizations
    plot_distributions(df_combined, load_levels=[7.0, 10.0])
    plot_boxplots(df_combined)
    plot_cv_vs_load(stats_df)

    # Print summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    print(stats_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("Visualization generation complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()

