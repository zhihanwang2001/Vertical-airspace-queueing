#!/usr/bin/env python3
"""
Quick progress check for running experiments
"""

import subprocess
import time
from pathlib import Path

def check_progress():
    """Check experiment progress."""

    print("\n" + "="*60)
    print("EXPERIMENT PROGRESS CHECK")
    print("="*60)
    print(f"Time: {time.strftime('%H:%M:%S')}")
    print()

    # Check if process is running
    result = subprocess.run(
        ['ps', 'aux'],
        capture_output=True,
        text=True
    )

    is_running = 'rerun_a2c_ppo_experiments.py' in result.stdout

    if is_running:
        # Get process info
        for line in result.stdout.split('\n'):
            if 'rerun_a2c_ppo_experiments.py' in line:
                parts = line.split()
                cpu = parts[2]
                mem = parts[3]
                print(f"✓ Experiment process is running")
                print(f"  CPU: {cpu}% | Memory: {mem}%")
                break

        # Check log file for progress
        log_file = Path("rerun_experiments_fixed.log")
        if log_file.exists():
            with open(log_file, 'r') as f:
                lines = f.readlines()

            # Find last experiment marker
            for line in reversed(lines[-200:]):
                if line.startswith('['):
                    print(f"  {line.strip()}")
                    break

            # Find last completion
            for line in reversed(lines[-500:]):
                if 'Completed:' in line or 'Mean reward:' in line:
                    print(f"  {line.strip()}")
                    break
    else:
        print("✗ Process not running")

        # Check if experiments completed
        data_dir = Path("Data/hca2c_final_comparison")
        recent_files = []

        for json_file in data_dir.glob("*.json"):
            mtime = json_file.stat().st_mtime
            if time.time() - mtime < 7200:  # Modified in last 2 hours
                recent_files.append(json_file.name)

        if recent_files:
            print(f"\n✓ {len(recent_files)} files updated recently")
            print("\nRecent files:")
            for f in sorted(recent_files)[-5:]:
                print(f"  - {f}")
        else:
            print("\n⚠️  No recent file updates")

    print()

if __name__ == '__main__':
    check_progress()
