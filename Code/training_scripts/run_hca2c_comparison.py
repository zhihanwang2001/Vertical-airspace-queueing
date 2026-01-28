"""
HCA2C Comparison Experiment Script

Runs comprehensive comparison between HCA2C, A2C, and PPO across
multiple load levels and random seeds.

Usage:
    python run_hca2c_comparison.py --timesteps 500000 --seeds 42 43 44 45 46 --loads 3 5 10
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import time
import numpy as np
from datetime import datetime
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Import algorithms
from algorithms.hca2c.hca2c_baseline import HCA2CBaseline
from algorithms.baselines.sb3_a2c_baseline import SB3A2CBaseline
from algorithms.baselines.sb3_ppo_baseline import SB3PPOBaseline
from env.drl_optimized_env_fixed import DRLOptimizedQueueEnvFixed


def create_env_with_load(load_multiplier: float, max_steps: int = 10000):
    """Create environment with specified load multiplier"""
    env = DRLOptimizedQueueEnvFixed(max_episode_steps=max_steps)
    env.base_arrival_rate = 0.3 * load_multiplier
    return env


def run_single_experiment(
    algorithm: str,
    load_multiplier: float,
    seed: int,
    total_timesteps: int,
    output_dir: str,
    verbose: int = 1
) -> Dict:
    """
    Run a single experiment

    Args:
        algorithm: Algorithm name ('hca2c', 'a2c', 'ppo')
        load_multiplier: Load multiplier
        seed: Random seed
        total_timesteps: Training timesteps
        output_dir: Output directory
        verbose: Verbosity level

    Returns:
        Experiment results dictionary
    """
    print(f"\n{'='*60}")
    print(f"Running: {algorithm.upper()} | Load: {load_multiplier}x | Seed: {seed}")
    print(f"{'='*60}")

    start_time = time.time()

    # Create algorithm
    config = {
        'seed': seed,
        'load_multiplier': load_multiplier,
        'verbose': verbose
    }

    if algorithm.lower() == 'hca2c':
        baseline = HCA2CBaseline(config)
        baseline.setup_env(load_multiplier=load_multiplier)
    elif algorithm.lower() == 'a2c':
        baseline = SB3A2CBaseline(config)
        baseline.setup_env()
        # Modify load
        baseline.env.envs[0].env.env.base_arrival_rate = 0.3 * load_multiplier
    elif algorithm.lower() == 'ppo':
        baseline = SB3PPOBaseline(config)
        baseline.setup_env()
        # Modify load
        baseline.env.envs[0].env.env.base_arrival_rate = 0.3 * load_multiplier
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # Train
    train_results = baseline.train(
        total_timesteps=total_timesteps,
        progress_bar=True
    )

    # Evaluate
    eval_results = baseline.evaluate(n_episodes=50, deterministic=True, verbose=False)

    training_time = time.time() - start_time

    # Compile results
    results = {
        'algorithm': algorithm,
        'load_multiplier': load_multiplier,
        'seed': seed,
        'total_timesteps': total_timesteps,
        'training_time': training_time,
        'train': train_results,
        'eval': {
            'mean_reward': eval_results['mean_reward'],
            'std_reward': eval_results['std_reward'],
            'mean_length': eval_results['mean_length'],
            'crash_rate': eval_results.get('crash_rate', 0),
            'episode_rewards': eval_results['episode_rewards']
        }
    }

    # Save individual results
    result_file = os.path.join(
        output_dir,
        f"{algorithm}_load{load_multiplier}_seed{seed}_results.json"
    )
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

    # Save model
    model_file = os.path.join(
        output_dir,
        f"{algorithm}_load{load_multiplier}_seed{seed}_model"
    )
    if algorithm.lower() == 'hca2c':
        baseline.save(model_file + '.pt')
    else:
        baseline.save(model_file + '.zip')

    print(f"\n✅ Completed: {algorithm.upper()} | Load: {load_multiplier}x | Seed: {seed}")
    print(f"   Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"   Crash Rate: {eval_results.get('crash_rate', 0):.1%}")
    print(f"   Training Time: {training_time:.1f}s")

    return results


def run_comparison_experiment(
    algorithms: List[str],
    loads: List[float],
    seeds: List[int],
    total_timesteps: int,
    output_dir: str
) -> Dict:
    """
    Run full comparison experiment

    Args:
        algorithms: List of algorithms to compare
        loads: List of load multipliers
        seeds: List of random seeds
        total_timesteps: Training timesteps per run
        output_dir: Output directory

    Returns:
        Aggregated results dictionary
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"# HCA2C COMPARISON EXPERIMENT")
    print(f"{'#'*70}")
    print(f"Algorithms: {algorithms}")
    print(f"Load levels: {loads}")
    print(f"Seeds: {seeds}")
    print(f"Timesteps per run: {total_timesteps:,}")
    print(f"Total runs: {len(algorithms) * len(loads) * len(seeds)}")
    print(f"Output directory: {output_dir}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    all_results = []
    experiment_start = time.time()

    for algorithm in algorithms:
        for load in loads:
            for seed in seeds:
                try:
                    results = run_single_experiment(
                        algorithm=algorithm,
                        load_multiplier=load,
                        seed=seed,
                        total_timesteps=total_timesteps,
                        output_dir=output_dir,
                        verbose=0
                    )
                    all_results.append(results)
                except Exception as e:
                    print(f"\n❌ Error in {algorithm} | Load: {load}x | Seed: {seed}")
                    print(f"   Error: {str(e)}")
                    all_results.append({
                        'algorithm': algorithm,
                        'load_multiplier': load,
                        'seed': seed,
                        'error': str(e)
                    })

    total_time = time.time() - experiment_start

    # Aggregate results
    summary = aggregate_results(all_results, algorithms, loads, seeds)
    summary['total_experiment_time'] = total_time
    summary['timestamp'] = datetime.now().isoformat()

    # Save summary
    summary_file = os.path.join(output_dir, 'experiment_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

    # Print summary
    print_summary(summary, algorithms, loads)

    return summary


def aggregate_results(
    all_results: List[Dict],
    algorithms: List[str],
    loads: List[float],
    seeds: List[int]
) -> Dict:
    """Aggregate results across seeds"""
    summary = {
        'algorithms': algorithms,
        'loads': loads,
        'seeds': seeds,
        'results': {}
    }

    for algorithm in algorithms:
        summary['results'][algorithm] = {}

        for load in loads:
            # Filter results for this algorithm and load
            filtered = [
                r for r in all_results
                if r.get('algorithm') == algorithm
                and r.get('load_multiplier') == load
                and 'error' not in r
            ]

            if filtered:
                rewards = [r['eval']['mean_reward'] for r in filtered]
                crash_rates = [r['eval'].get('crash_rate', 0) for r in filtered]
                lengths = [r['eval']['mean_length'] for r in filtered]
                times = [r['training_time'] for r in filtered]

                summary['results'][algorithm][f'load_{load}x'] = {
                    'mean_reward': float(np.mean(rewards)),
                    'std_reward': float(np.std(rewards)),
                    'mean_crash_rate': float(np.mean(crash_rates)),
                    'mean_length': float(np.mean(lengths)),
                    'mean_training_time': float(np.mean(times)),
                    'n_successful_runs': len(filtered),
                    'rewards_per_seed': rewards
                }
            else:
                summary['results'][algorithm][f'load_{load}x'] = {
                    'error': 'No successful runs'
                }

    return summary


def print_summary(summary: Dict, algorithms: List[str], loads: List[float]):
    """Print formatted summary"""
    print(f"\n\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}")

    # Header
    header = f"{'Algorithm':<12}"
    for load in loads:
        header += f" | {load}x Load (Reward)"
    print(header)
    print("-" * 80)

    # Results per algorithm
    for algorithm in algorithms:
        row = f"{algorithm.upper():<12}"
        for load in loads:
            key = f'load_{load}x'
            if key in summary['results'].get(algorithm, {}):
                data = summary['results'][algorithm][key]
                if 'error' not in data:
                    row += f" | {data['mean_reward']:>8.1f} ± {data['std_reward']:<6.1f}"
                else:
                    row += f" | {'ERROR':<16}"
            else:
                row += f" | {'N/A':<16}"
        print(row)

    print("-" * 80)

    # Crash rates
    print("\nCrash Rates:")
    for algorithm in algorithms:
        row = f"  {algorithm.upper():<10}"
        for load in loads:
            key = f'load_{load}x'
            if key in summary['results'].get(algorithm, {}):
                data = summary['results'][algorithm][key]
                if 'error' not in data:
                    row += f" | {load}x: {data['mean_crash_rate']:>5.1%}"
        print(row)

    # Statistical comparison (HCA2C vs A2C)
    if 'hca2c' in [a.lower() for a in algorithms] and 'a2c' in [a.lower() for a in algorithms]:
        print("\n\nHCA2C vs A2C Comparison:")
        print("-" * 50)
        for load in loads:
            key = f'load_{load}x'
            hca2c_data = summary['results'].get('hca2c', {}).get(key, {})
            a2c_data = summary['results'].get('a2c', {}).get(key, {})

            if 'error' not in hca2c_data and 'error' not in a2c_data:
                hca2c_reward = hca2c_data['mean_reward']
                a2c_reward = a2c_data['mean_reward']
                improvement = (hca2c_reward - a2c_reward) / abs(a2c_reward) * 100 if a2c_reward != 0 else 0

                print(f"  {load}x Load:")
                print(f"    HCA2C: {hca2c_reward:.1f} | A2C: {a2c_reward:.1f}")
                print(f"    Improvement: {improvement:+.1f}%")

    print(f"\n{'='*80}")
    print(f"Total experiment time: {summary.get('total_experiment_time', 0)/3600:.2f} hours")
    print(f"Results saved to: {summary.get('output_dir', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(description='HCA2C Comparison Experiment')

    parser.add_argument('--algorithms', nargs='+', default=['hca2c', 'a2c', 'ppo'],
                        help='Algorithms to compare')
    parser.add_argument('--loads', nargs='+', type=float, default=[3.0, 5.0, 10.0],
                        help='Load multipliers')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44, 45, 46],
                        help='Random seeds')
    parser.add_argument('--timesteps', type=int, default=500000,
                        help='Training timesteps per run')
    parser.add_argument('--output-dir', type=str, default='../../Data/hca2c_experiments',
                        help='Output directory')
    parser.add_argument('--test-mode', action='store_true',
                        help='Run quick test (1000 steps, 1 seed)')

    args = parser.parse_args()

    # Test mode for quick validation
    if args.test_mode:
        print("Running in TEST MODE (quick validation)...")
        args.timesteps = 1000
        args.seeds = [42]
        args.loads = [5.0]

    # Run experiment
    output_dir = os.path.join(
        os.path.dirname(__file__),
        args.output_dir,
        datetime.now().strftime('%Y%m%d_%H%M%S')
    )

    summary = run_comparison_experiment(
        algorithms=args.algorithms,
        loads=args.loads,
        seeds=args.seeds,
        total_timesteps=args.timesteps,
        output_dir=output_dir
    )

    print(f"\n✅ Experiment completed!")
    print(f"   Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
