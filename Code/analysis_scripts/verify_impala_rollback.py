#!/usr/bin/env python3
"""
Verify if IMPALA has been successfully rolled back to original configuration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from algorithms.advanced.impala.impala_baseline import IMPALABaseline

def verify_impala_rollback():
    """Verify IMPALA rollback configuration"""
    print("🔄 Verifying if IMPALA has been rolled back to original configuration...")

    # Create IMPALA instance
    baseline = IMPALABaseline()
    config = baseline.config

    print("\n📋 Current configuration:")

    # Check original configuration parameters
    original_config = {
        "Learning rate": (config['learning_rate'], 1e-4),
        "Entropy coefficient": (config['entropy_coeff'], 0.01),
        "Value loss weight": (config['value_loss_coeff'], 0.5),
        "Gradient clip": (config['gradient_clip'], 40.0),
        "V-trace ρ_bar": (config['rho_bar'], 1.0),
        "V-trace c_bar": (config['c_bar'], 1.0),
        "Buffer size": (config['buffer_size'], 10000),
        "Sequence length": (config['sequence_length'], 20),
        "Batch size": (config['batch_size'], 16),
        "Learning starts": (config['learning_starts'], 1000),
        "Train frequency": (config['train_freq'], 4),
        "Update frequency": (config['update_freq'], 100)
    }

    all_correct = True
    for param_name, (actual, expected) in original_config.items():
        status = "✅" if actual == expected else "❌"
        if actual != expected:
            all_correct = False
        print(f"   {status} {param_name}: {actual} (expected: {expected})")

    print(f"\n🔄 Rollback result:")
    if all_correct:
        print("✅ IMPALA has been successfully rolled back to original configuration!")
        print("✅ All parameters have been restored to initial values")
        print("✅ Can restart optimization strategy")
    else:
        print("❌ Rollback incomplete, needs checking")
        return False

    return True

if __name__ == "__main__":
    print("🚀 Starting IMPALA rollback verification...")

    success = verify_impala_rollback()

    if success:
        print("\n🎯 Rollback complete! IMPALA has been restored to original configuration.")
        print("Now you can redesign optimization strategy.")
    else:
        print("\n❌ Rollback failed, please check configuration.")