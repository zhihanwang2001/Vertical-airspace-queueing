#!/usr/bin/env python3
"""
Monitor ablation study progress in real-time
"""

import os
import sys
import time
import json
from datetime import datetime

def check_progress():
    """Check ablation study progress"""

    print("="*80)
    print(f"Ablation Study Progress Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    # Check if process is running
    import subprocess
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True
        )
        if "run_ablation_studies.py" in result.stdout:
            print("✓ Process is running")
            # Extract CPU and memory usage
            for line in result.stdout.split('\n'):
                if "run_ablation_studies.py" in line:
                    parts = line.split()
                    cpu = parts[2]
                    mem = parts[3]
                    print(f"  CPU: {cpu}% | Memory: {mem}%")
        else:
            print("✗ Process not found")
    except Exception as e:
        print(f"Error checking process: {e}")

    print()

    # Check for result files
    base_dir = "Data/ablation_studies"
    variants = ["hca2c_full", "hca2c_flat", "hca2c_wide", "a2c_enhanced"]
    seeds = [42, 43, 44]

    completed = []
    for variant in variants:
        variant_dir = os.path.join(base_dir, variant)
        if os.path.exists(variant_dir):
            for seed in seeds:
                result_file = os.path.join(variant_dir, f"{variant}_seed{seed}_results.json")
                if os.path.exists(result_file):
                    try:
                        with open(result_file, 'r') as f:
                            data = json.load(f)
                            completed.append({
                                'variant': variant,
                                'seed': seed,
                                'reward': data.get('mean_reward', 0),
                                'time': data.get('training_time_minutes', 0)
                            })
                    except:
                        pass

    print(f"Completed: {len(completed)}/12 runs")
    print()

    if completed:
        print("Completed Runs:")
        print("-"*80)
        for run in completed:
            print(f"  {run['variant']:20s} seed={run['seed']} | "
                  f"Reward: {run['reward']:8.0f} | Time: {run['time']:.1f} min")
        print()

        # Group by variant
        from collections import defaultdict
        by_variant = defaultdict(list)
        for run in completed:
            by_variant[run['variant']].append(run['reward'])

        if by_variant:
            print("Summary by Variant:")
            print("-"*80)
            for variant in variants:
                if variant in by_variant:
                    rewards = by_variant[variant]
                    mean = sum(rewards) / len(rewards)
                    print(f"  {variant:20s}: {len(rewards)}/3 runs | Mean: {mean:8.0f}")
    else:
        print("No completed runs yet. Experiment is still initializing or training...")
        print()
        print("This is normal for the first few minutes.")
        print("First run (HCA2C-Full seed=42) will take ~2.5 hours.")

    print()
    print("="*80)
    print("Commands:")
    print("  Monitor log: tail -f ablation_studies.log")
    print("  Check again: python monitor_ablation.py")
    print("  Stop: kill $(cat ablation_studies.pid)")
    print("="*80)

if __name__ == '__main__':
    check_progress()
