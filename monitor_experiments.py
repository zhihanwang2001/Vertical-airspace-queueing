"""
Monitor experiment progress and automatically analyze when complete
"""

import time
import json
from pathlib import Path
import subprocess


def check_experiment_progress():
    """Check how many experiments have been completed."""

    data_dir = Path("Data/hca2c_final_comparison")

    # Expected experiments to be rerun
    algorithms = ['A2C', 'PPO']
    seeds = [42, 43, 44, 45, 46]
    loads = [3.0, 5.0]

    completed = []
    missing = []

    for algo in algorithms:
        for seed in seeds:
            for load in loads:
                filename = f"{algo}_seed{seed}_load{load}.json"
                filepath = data_dir / filename

                if filepath.exists():
                    # Check if file was recently modified (within last hour)
                    mtime = filepath.stat().st_mtime
                    if time.time() - mtime < 3600:  # Modified within last hour
                        completed.append(filename)
                else:
                    missing.append(filename)

    return completed, missing


def check_process_running():
    """Check if the experiment script is still running."""

    result = subprocess.run(
        ['ps', 'aux'],
        capture_output=True,
        text=True
    )

    return 'rerun_a2c_ppo_experiments.py' in result.stdout


def main():
    """Monitor experiment progress."""

    print("\n" + "="*60)
    print("EXPERIMENT PROGRESS MONITOR")
    print("="*60)

    total_experiments = 20
    check_interval = 60  # Check every 60 seconds

    while True:
        completed, missing = check_experiment_progress()
        is_running = check_process_running()

        print(f"\n[{time.strftime('%H:%M:%S')}] Progress Update:")
        print(f"  Completed: {len(completed)}/{total_experiments}")
        print(f"  Remaining: {len(missing)}")
        print(f"  Process running: {'Yes' if is_running else 'No'}")

        if len(completed) > 0:
            print(f"\n  Recently completed:")
            for exp in completed[-3:]:  # Show last 3
                print(f"    - {exp}")

        # Check if all experiments are done
        if len(completed) == total_experiments:
            print("\n" + "="*60)
            print("✓ ALL EXPERIMENTS COMPLETED!")
            print("="*60)
            print("\nStarting automatic analysis...")

            # Run analysis
            subprocess.run([
                'python3',
                'Analysis/statistical_analysis/analyze_hca2c_ablation.py'
            ])

            print("\n✓ Analysis complete!")
            print("\nNext steps:")
            print("1. Review the updated analysis report")
            print("2. Check the new figures")
            print("3. Update the manuscript with final statistics")
            break

        # Check if process stopped but experiments not complete
        if not is_running and len(completed) < total_experiments:
            print("\n⚠️  WARNING: Process stopped but experiments incomplete!")
            print(f"   Completed: {len(completed)}/{total_experiments}")
            print(f"   Missing: {len(missing)}")
            print("\n   Check rerun_experiments.log for errors")
            break

        # Wait before next check
        print(f"\n  Checking again in {check_interval} seconds...")
        time.sleep(check_interval)


if __name__ == '__main__':
    main()
