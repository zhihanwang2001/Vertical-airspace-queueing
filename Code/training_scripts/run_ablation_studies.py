"""
Ablation Studies for HCA2C

Tests the contribution of each component:
1. HCA2C-Full: Complete HCA2C (baseline)
2. HCA2C-Flat: Without neighbor-aware features
3. HCA2C-Wide: Without capacity-aware action clipping
4. HCA2C-Single: Without hierarchical decomposition
5. A2C-Enhanced: A2C with matched network capacity

Usage:
    python run_ablation_studies.py --seeds 42 43 44 --load 3.0
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import time
import numpy as np
from datetime import datetime
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

# Import algorithms
from algorithms.hca2c.hca2c_baseline import HCA2CBaseline
from algorithms.baselines.sb3_a2c_enhanced import SB3A2CEnhanced


def run_ablation_experiment(
    variant: str,
    load_multiplier: float,
    seed: int,
    total_timesteps: int,
    output_dir: str
) -> Dict:
    """
    Run single ablation experiment

    Args:
        variant: Variant name ('hca2c_full', 'hca2c_flat', etc.)
        load_multiplier: Load multiplier
        seed: Random seed
        total_timesteps: Training timesteps
        output_dir: Output directory

    Returns:
        Results dictionary
    """
    print(f"\n{'='*70}")
    print(f"Ablation: {variant.upper()} | Load: {load_multiplier}x | Seed: {seed}")
    print(f"{'='*70}")

    start_time = time.time()

    config = {
        'seed': seed,
        'load_multiplier': load_multiplier,
        'verbose': 1
    }

    # Create variant-specific baseline
    if variant == 'hca2c_full':
        # Standard HCA2C
        baseline = HCA2CBaseline(config)
        baseline.setup_env(load_multiplier=load_multiplier)

    elif variant == 'hca2c_flat':
        # HCA2C with flat observation (no neighbor info)
        from algorithms.hca2c.wrapper_flat import FlatObservationWrapper
        from env.drl_optimized_env_fixed import DRLOptimizedQueueEnvFixed

        baseline = HCA2CBaseline(config)
        # Create base env and wrap with flat wrapper
        base_env = DRLOptimizedQueueEnvFixed(max_episode_steps=10000)
        base_env.base_arrival_rate = 0.3 * load_multiplier
        baseline.env = FlatObservationWrapper(base_env)

    elif variant == 'hca2c_wide':
        # HCA2C with wide action space
        from algorithms.hca2c.wrapper_wide import WideActionWrapper
        from env.drl_optimized_env_fixed import DRLOptimizedQueueEnvFixed

        baseline = HCA2CBaseline(config)
        # Create base env and wrap with wide wrapper
        base_env = DRLOptimizedQueueEnvFixed(max_episode_steps=10000)
        base_env.base_arrival_rate = 0.3 * load_multiplier
        baseline.env = WideActionWrapper(base_env)

    elif variant == 'hca2c_single':
        # HCA2C with single policy network
        # Note: This requires modifying HCA2C to use SinglePolicyNetwork
        # For now, we'll skip this variant and implement it later if needed
        print(f"[WARNING] {variant} not yet implemented, skipping...")
        return None

    elif variant == 'a2c_enhanced':
        # Enhanced A2C with large network
        baseline = SB3A2CEnhanced(config)
        baseline.setup_env()
        # Access the base environment through the wrapper chain
        # baseline.vec_env -> DummyVecEnv -> [Monitor] -> SB3DictWrapper -> DRLOptimizedQueueEnvFixed
        baseline.vec_env.envs[0].env.env.base_arrival_rate = 0.3 * load_multiplier

    else:
        raise ValueError(f"Unknown variant: {variant}")

    # Train
    print(f"\n[Training] {variant} for {total_timesteps:,} timesteps...")
    train_results = baseline.train(total_timesteps=total_timesteps, progress_bar=True)

    # Evaluate
    print(f"\n[Evaluation] Running 50 episodes...")
    eval_results = baseline.evaluate(n_episodes=50, deterministic=True, verbose=False)

    training_time = time.time() - start_time

    # Compile results
    results = {
        'variant': variant,
        'load_multiplier': load_multiplier,
        'seed': seed,
        'total_timesteps': total_timesteps,
        'mean_reward': float(eval_results['mean_reward']),
        'std_reward': float(eval_results['std_reward']),
        'crash_rate': float(eval_results.get('crash_rate', 0.0)),
        'training_time_minutes': training_time / 60,
        'timestamp': datetime.now().isoformat()
    }

    # Save individual result
    variant_dir = os.path.join(output_dir, variant)
    os.makedirs(variant_dir, exist_ok=True)

    result_file = os.path.join(variant_dir, f'{variant}_seed{seed}_results.json')
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Save model
    model_file = os.path.join(variant_dir, f'{variant}_seed{seed}_model.zip')
    baseline.save(model_file)

    print(f"\n[Results] {variant}:")
    print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Crash Rate: {results['crash_rate']:.2%}")
    print(f"  Training Time: {results['training_time_minutes']:.1f} min")

    return results


def main():
    parser = argparse.ArgumentParser(description='Run HCA2C ablation studies')
    parser.add_argument('--variants', nargs='+',
                       default=['hca2c_full', 'hca2c_flat', 'hca2c_wide', 'a2c_enhanced'],
                       help='Variants to test')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44],
                       help='Random seeds')
    parser.add_argument('--load', type=float, default=3.0,
                       help='Load multiplier')
    parser.add_argument('--timesteps', type=int, default=500000,
                       help='Training timesteps')
    parser.add_argument('--output-dir', type=str, default='Data/ablation_studies',
                       help='Output directory')

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"HCA2C ABLATION STUDIES")
    print(f"{'='*70}")
    print(f"Variants: {args.variants}")
    print(f"Seeds: {args.seeds}")
    print(f"Load: {args.load}x")
    print(f"Timesteps: {args.timesteps:,}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*70}\n")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run all experiments
    all_results = []
    total_experiments = len(args.variants) * len(args.seeds)
    current_exp = 0

    for variant in args.variants:
        for seed in args.seeds:
            current_exp += 1
            print(f"\n\n{'#'*70}")
            print(f"# Experiment {current_exp}/{total_experiments}")
            print(f"{'#'*70}")

            try:
                results = run_ablation_experiment(
                    variant=variant,
                    load_multiplier=args.load,
                    seed=seed,
                    total_timesteps=args.timesteps,
                    output_dir=args.output_dir
                )

                if results is not None:
                    all_results.append(results)

            except Exception as e:
                print(f"\n[ERROR] Failed to run {variant} seed {seed}: {e}")
                import traceback
                traceback.print_exc()

    # Save summary
    if all_results:
        summary_file = os.path.join(args.output_dir, 'ablation_results.csv')
        import pandas as pd
        df = pd.DataFrame(all_results)
        df.to_csv(summary_file, index=False)

        print(f"\n\n{'='*70}")
        print(f"ABLATION STUDIES COMPLETE")
        print(f"{'='*70}")
        print(f"Total experiments: {len(all_results)}/{total_experiments}")
        print(f"Results saved to: {summary_file}")
        print(f"{'='*70}\n")
    else:
        print("\n[ERROR] No experiments completed successfully!")


if __name__ == '__main__':
    main()
