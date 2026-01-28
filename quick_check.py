#!/usr/bin/env python3
"""
Quick check script for ablation study progress
Usage: python quick_check.py
"""

import os
import json
import subprocess
from datetime import datetime

def quick_check():
    """Quick progress check"""

    print("
" + "="*60)
    print(f"Time: {datetime.now().strftime('%H:%M:%S')} - Quick Status Check")
    print("="*60)

    # Check process
    try:
        result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
        if "run_ablation_studies.py" in result.stdout:
            print("Experiment running")
        else:
            print("Experiment not running!")
            return
    except:
        pass

    # Count completed runs
    completed = 0
    variants = ["hca2c_full", "hca2c_flat", "hca2c_wide", "a2c_enhanced"]

    for variant in variants:
        variant_dir = f"Data/ablation_studies/{variant}"
        if os.path.exists(variant_dir):
            for seed in [42, 43, 44]:
                result_file = f"{variant_dir}/{variant}_seed{seed}_results.json"
                if os.path.exists(result_file):
                    completed += 1

    print(f"Progress: {completed}/12 runs completed ({completed/12*100:.0f}%)")

    # Estimate remaining time
    if completed > 0:
        remaining = 12 - completed
        est_hours = remaining * 2.5
        print(f"Estimated remaining: ~{est_hours:.1f} hours")
    else:
        print(f"First run in progress (takes ~2.5 hours)")

    print("="*60 + "
")

if __name__ == '__main__':
    quick_check()
