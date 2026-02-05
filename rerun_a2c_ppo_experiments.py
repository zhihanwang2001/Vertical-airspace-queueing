"""
Rerun A2C and PPO experiments for load 3.0 and 5.0
Fix the data duplication issue found in the analysis

Experiments to run:
- A2C: seeds 42-46 × loads 3.0, 5.0 = 10 experiments
- PPO: seeds 42-46 × loads 3.0, 5.0 = 10 experiments
Total: 20 experiments

Note: Load 7.0 experiments are already correct and will not be rerun.
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'Code'))
sys.path.insert(0, os.path.join(project_root, 'Code', 'algorithms'))

import json
import time
from pathlib import Path
from algorithms.baselines.sb3_a2c_baseline import SB3A2CBaseline
from algorithms.baselines.sb3_ppo_baseline import SB3PPOBaseline


def run_single_experiment(algo_name, algo_class, seed, load, output_dir):
    """Run a single experiment and save results."""

    print(f"\n{'='*60}")
    print(f"Running: {algo_name} seed={seed} load={load}")
    print(f"{'='*60}")

    start_time = time.time()

    # Create model with configuration
    config = {
        'seed': seed,
        'verbose': 1,
    }

    model = algo_class(config)

    # Setup environment (no load_multiplier parameter for A2C/PPO)
    model.setup_env()

    # Set load multiplier by adjusting base_arrival_rate
    # Default base_arrival_rate is 0.3, multiply by load to get desired load level
    model.env.unwrapped.base_arrival_rate = 0.3 * load
    model.env.unwrapped.current_arrival_rate = 0.3 * load

    # Train model
    timesteps = 100000  # Same as original experiments
    print(f"Training for {timesteps} timesteps...")
    model.train(total_timesteps=timesteps)

    train_time = time.time() - start_time

    # Evaluate model with the same load multiplier
    print(f"Evaluating with load={load}...")

    # Create evaluation environment with correct load multiplier
    from env.drl_optimized_env_fixed import DRLOptimizedQueueEnvFixed
    from algorithms.baselines.space_utils import SB3DictWrapper
    import numpy as np

    eval_env = SB3DictWrapper(DRLOptimizedQueueEnvFixed(max_episode_steps=10000))
    # Set load multiplier by adjusting base_arrival_rate
    eval_env.unwrapped.base_arrival_rate = 0.3 * load
    eval_env.unwrapped.current_arrival_rate = 0.3 * load

    # Manual evaluation with correct environment
    episode_rewards = []
    episode_lengths = []

    for episode in range(30):
        obs, _ = eval_env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            action, _ = model.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1

            if episode_length >= 10000:
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    eval_results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
    }

    # Prepare results
    results = {
        'algorithm': algo_name,
        'seed': seed,
        'load_multiplier': load,
        'timesteps': timesteps,
        'train_time': train_time,
        'mean_reward': eval_results['mean_reward'],
        'std_reward': eval_results['std_reward'],
        'mean_length': eval_results.get('mean_length', eval_results.get('mean_episode_length', 0.0)),
        'crash_rate': eval_results.get('crash_rate', 0.0),
        'episode_rewards': eval_results['episode_rewards'],
        'episode_lengths': eval_results['episode_lengths'],
    }

    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_file = output_dir / f"{algo_name}_seed{seed}_load{load}.json"
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Save model
    model_file = output_dir / f"{algo_name}_seed{seed}_load{load}_model.zip"
    model.save(str(model_file))

    print(f"\n✓ Completed: {algo_name} seed={seed} load={load}")
    print(f"  Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Train time: {train_time/60:.1f} minutes")
    print(f"  Saved to: {result_file}")

    return results


def main():
    """Main execution function."""

    print("\n" + "="*60)
    print("RERUNNING A2C AND PPO EXPERIMENTS")
    print("="*60)
    print("\nFixing data duplication issue:")
    print("- A2C: seeds 42-46 × loads 3.0, 5.0 = 10 experiments")
    print("- PPO: seeds 42-46 × loads 3.0, 5.0 = 10 experiments")
    print("- Total: 20 experiments")
    print("\nNote: Load 7.0 experiments are already correct.")
    print("="*60)

    # Configuration
    output_dir = Path("Data/hca2c_final_comparison")
    seeds = [42, 43, 44, 45, 46]
    loads = [3.0, 5.0]

    algorithms = [
        ('A2C', SB3A2CBaseline),
        ('PPO', SB3PPOBaseline),
    ]

    # Track progress
    total_experiments = len(algorithms) * len(seeds) * len(loads)
    completed = 0
    failed = []

    overall_start = time.time()

    # Run experiments
    for algo_name, algo_class in algorithms:
        for seed in seeds:
            for load in loads:
                try:
                    completed += 1
                    print(f"\n[{completed}/{total_experiments}] Starting experiment...")

                    run_single_experiment(
                        algo_name=algo_name,
                        algo_class=algo_class,
                        seed=seed,
                        load=load,
                        output_dir=output_dir
                    )

                except Exception as e:
                    print(f"\n✗ FAILED: {algo_name} seed={seed} load={load}")
                    print(f"  Error: {e}")
                    failed.append(f"{algo_name}_seed{seed}_load{load}")
                    continue

    # Summary
    overall_time = time.time() - overall_start

    print("\n" + "="*60)
    print("EXPERIMENT COMPLETION SUMMARY")
    print("="*60)
    print(f"Total experiments: {total_experiments}")
    print(f"Completed: {completed - len(failed)}")
    print(f"Failed: {len(failed)}")
    print(f"Total time: {overall_time/60:.1f} minutes")

    if failed:
        print("\nFailed experiments:")
        for exp in failed:
            print(f"  - {exp}")
    else:
        print("\n✓ All experiments completed successfully!")

    print("\nNext steps:")
    print("1. Verify the new data files")
    print("2. Rerun the analysis script: python3 Analysis/statistical_analysis/analyze_hca2c_ablation.py")
    print("3. Check the updated statistics and figures")
    print("="*60)


if __name__ == '__main__':
    main()
