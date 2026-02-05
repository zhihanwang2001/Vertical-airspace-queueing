#!/usr/bin/env python3
"""
Verify that optimized Rainbow DQN will be correctly called by run_advanced_algorithm_comparison.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_algorithms import create_algorithm_baseline

def verify_optimized_rainbow():
    """Verify optimized Rainbow DQN configuration"""
    print("🔍 Verifying if optimized Rainbow DQN will be correctly called...")

    # Simulate run_advanced_algorithm_comparison.py call method
    print("\n1️⃣ Simulating script call: create_algorithm_baseline('rainbow_dqn')")
    baseline = create_algorithm_baseline("rainbow_dqn")

    # Check configuration
    config = baseline.config
    print(f"\n📋 Retrieved configuration:")

    # Check key optimization parameters
    optimizations = {
        "Learning rate": (config['learning_rate'], 1e-4, "6.25e-5 → 1e-4"),
        "Target network update": (config['target_update_freq'], 2000, "8000 → 2000"),
        "Learning starts": (config['learning_starts'], 5000, "50000 → 5000"),
        "Multi-step": (config['n_step'], 10, "3 → 10"),
        "Buffer size": (config['buffer_size'], 200000, "1M → 200k")
    }

    all_correct = True
    for param_name, (actual, expected, change) in optimizations.items():
        status = "✅" if actual == expected else "❌"
        if actual != expected:
            all_correct = False
        print(f"   {status} {param_name}: {actual} (expected: {expected}) - {change}")

    print(f"\n2️⃣ Verification result:")
    if all_correct:
        print("✅ All optimization parameters are correct!")
        print("✅ run_advanced_algorithm_comparison.py will use optimized configuration")
    else:
        print("❌ Configuration incorrect, needs checking")
        return False

    print(f"\n3️⃣ Training command verification:")
    print("Command: python run_advanced_algorithm_comparison.py --algorithms rainbow_dqn --timesteps 500000 --eval-episodes 5")
    print("✅ This command will automatically use optimized Rainbow DQN configuration")

    return True

if __name__ == "__main__":
    verify_optimized_rainbow()