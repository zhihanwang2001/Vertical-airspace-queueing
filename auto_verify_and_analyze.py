#!/usr/bin/env python3
"""
Auto-verify and analyze when experiments complete
"""

import time
import subprocess
import json
from pathlib import Path

def wait_for_completion():
    """Wait for experiments to complete."""

    print("\n" + "="*60)
    print("WAITING FOR EXPERIMENTS TO COMPLETE")
    print("="*60)

    while True:
        # Check if process is running
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True
        )

        if 'rerun_a2c_ppo_experiments.py' not in result.stdout:
            print(f"\n[{time.strftime('%H:%M:%S')}] ✓ Experiments completed!")
            break

        # Count completed experiments
        try:
            with open('rerun_experiments_correct.log', 'r') as f:
                content = f.read()
                completed = content.count('✓ Completed:')

            print(f"[{time.strftime('%H:%M:%S')}] Progress: {completed}/20 experiments", end='\r')
        except:
            pass

        time.sleep(30)  # Check every 30 seconds

def verify_data():
    """Verify that data is now different across loads."""

    print("\n" + "="*60)
    print("VERIFYING DATA")
    print("="*60)

    data_dir = Path("Data/hca2c_final_comparison")

    issues = []

    for algo in ['A2C', 'PPO']:
        for seed in [42, 43, 44, 45, 46]:
            # Load data
            data_3 = json.loads((data_dir / f"{algo}_seed{seed}_load3.0.json").read_text())
            data_5 = json.loads((data_dir / f"{algo}_seed{seed}_load5.0.json").read_text())
            data_7 = json.loads((data_dir / f"{algo}_seed{seed}_load7.0.json").read_text())

            reward_3 = data_3['mean_reward']
            reward_5 = data_5['mean_reward']
            reward_7 = data_7['mean_reward']

            # Check if data is different
            if abs(reward_3 - reward_5) < 1.0:
                issues.append(f"{algo} seed{seed}: load 3.0 and 5.0 are same ({reward_3:.1f})")

            if abs(reward_5 - reward_7) < 1.0:
                issues.append(f"{algo} seed{seed}: load 5.0 and 7.0 are same ({reward_5:.1f})")

            print(f"{algo} seed{seed}: load3.0={reward_3:.1f}, load5.0={reward_5:.1f}, load7.0={reward_7:.1f}")

    if issues:
        print("\n⚠️  DATA ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("\n✓ All data verified - loads are different!")
        return True

def run_analysis():
    """Run the analysis script."""

    print("\n" + "="*60)
    print("RUNNING ANALYSIS")
    print("="*60)

    result = subprocess.run(
        ['python3', 'Analysis/statistical_analysis/analyze_hca2c_ablation.py'],
        capture_output=True,
        text=True
    )

    print(result.stdout)

    if result.returncode == 0:
        print("\n✓ Analysis completed successfully!")
        return True
    else:
        print("\n✗ Analysis failed!")
        print(result.stderr)
        return False

def main():
    """Main execution."""

    # Wait for experiments to complete
    wait_for_completion()

    # Verify data
    if not verify_data():
        print("\n⚠️  Data verification failed!")
        print("Please check the experiment logs and data files.")
        return

    # Run analysis
    if not run_analysis():
        print("\n⚠️  Analysis failed!")
        return

    # Success!
    print("\n" + "="*60)
    print("✓ ALL TASKS COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nGenerated files:")
    print("  - Analysis/statistical_reports/hca2c_ablation_report.md")
    print("  - Analysis/statistical_reports/hca2c_ablation_stats.csv")
    print("  - Analysis/statistical_reports/hca2c_ablation_comparisons.csv")
    print("  - Analysis/statistical_reports/hca2c_ablation_table.tex")
    print("  - Analysis/figures/hca2c_ablation_comprehensive.png")
    print("\nNext steps:")
    print("  1. Review the analysis report")
    print("  2. Check the comprehensive figure")
    print("  3. Update manuscript with final statistics")
    print()

if __name__ == '__main__':
    main()
