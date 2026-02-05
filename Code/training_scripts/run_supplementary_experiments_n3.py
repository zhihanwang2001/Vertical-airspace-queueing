"""
Supplementary Experiments: Sample Size Augmentation from n=1 to n=3

Objective:
- Add 2 independent training runs for core claims (total n=3)
- Set 1: Structural comparison (Inverted vs Normal Pyramid) - A2C, PPO
- Set 2: Capacity paradox (K=10 vs K=30) - A2C only

Seeds:
- Existing: 42 (original experiment)
- New Run 1: 123
- New Run 2: 456

Total: 12 new training runs
- 4 configs × 2 algorithms × 2 seeds = 8 (structural)
- 2 configs × 1 algorithm × 2 seeds = 4 (capacity)
"""

import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import gymnasium as gym
import numpy as np
import json
import time
from datetime import datetime
from stable_baselines3 import A2C, PPO

from env.config import VerticalQueueConfig
from env.configurable_env_wrapper import ConfigurableEnvWrapper
from env.drl_wrapper_fixed import DictToBoxActionWrapperFixed, ObservationWrapperFixed


def create_config(config_type='inverted_pyramid', high_load_multiplier=10.0):
    """
    Create high load configuration

    Args:
        config_type: Configuration type
            - inverted_pyramid: [8,6,4,3,2] Inverted pyramid
            - normal_pyramid: [2,3,4,6,8] Normal pyramid
            - low_capacity: [2,2,2,2,2] K=10
            - capacity_30: [6,6,6,6,6] K=30
        high_load_multiplier: Load multiplier (default 10.0)
    """
    config = VerticalQueueConfig()

    # Set capacity
    if config_type == 'inverted_pyramid':
        config.layer_capacities = [8, 6, 4, 3, 2]  # Total 23
    elif config_type == 'normal_pyramid':
        config.layer_capacities = [2, 3, 4, 6, 8]  # Total 23
    elif config_type == 'low_capacity':
        config.layer_capacities = [2, 2, 2, 2, 2]  # Total 10 (K=10)
    elif config_type == 'capacity_30':
        config.layer_capacities = [6, 6, 6, 6, 6]  # Total 30 (K=30)
    else:
        raise ValueError(f"Unknown config type: {config_type}")

    # Fixed real UAM traffic pattern
    config.arrival_weights = [0.3, 0.25, 0.2, 0.15, 0.1]

    # High load setting (10x)
    total_capacity = sum(config.layer_capacities)
    avg_service_rate = np.mean(config.layer_service_rates)
    base_rate_v3 = 0.75 * total_capacity * avg_service_rate / 5
    config.base_arrival_rate = base_rate_v3 * high_load_multiplier

    # Calculate theoretical load per layer
    layer_loads = []
    for i, (w, c) in enumerate(zip(config.arrival_weights, config.layer_capacities)):
        layer_arrival = config.base_arrival_rate * w
        actual_service_rate = config.layer_service_rates[i]
        layer_load = layer_arrival / (c * actual_service_rate)
        layer_loads.append(layer_load)

    print(f"\n{'='*80}")
    print(f"Configuration: {config_type}")
    print(f"Capacity: {config.layer_capacities} (Total: {total_capacity})")
    print(f"Arrival weights: {config.arrival_weights}")
    print(f"Total arrival rate: {config.base_arrival_rate:.2f} ({high_load_multiplier:.1f}x high load)")
    print(f"Average load: {np.mean(layer_loads)*100:.1f}%")
    print(f"{'='*80}\n")

    return config


def create_wrapped_env(config):
    """Create wrapped environment"""
    base_env = ConfigurableEnvWrapper(config=config)
    wrapped_env = DictToBoxActionWrapperFixed(base_env)
    wrapped_env = ObservationWrapperFixed(wrapped_env)
    return wrapped_env


def train_and_evaluate(algorithm_name='A2C', config_type='inverted_pyramid',
                       timesteps=100000, eval_episodes=50, seed=42,
                       high_load_multiplier=10.0):
    """
    Train and evaluate single experiment

    Args:
        algorithm_name: 'A2C' or 'PPO'
        config_type: Configuration type
        timesteps: Training steps (default 100K)
        eval_episodes: Evaluation episodes (default 50)
        seed: Random seed
        high_load_multiplier: High load multiplier (default 10x)
    """

    print(f"\n{'='*80}")
    print(f"Experiment: {algorithm_name} + {config_type}")
    print(f"Seed: {seed}")
    print(f"{'='*80}\n")

    config = create_config(config_type, high_load_multiplier)
    env = create_wrapped_env(config)

    # Save path: Data/ablation_studies/supplementary_n3/{config_type}/{algorithm}_{seed}_results.json
    save_dir = Path(project_root).parent / 'Data' / 'ablation_studies' / 'supplementary_n3' / config_type
    save_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # Create model
    if algorithm_name == 'A2C':
        model = A2C('MlpPolicy', env, learning_rate=0.0007, n_steps=32,
                   gamma=0.99, gae_lambda=0.95, ent_coef=0.01, vf_coef=0.5,
                   max_grad_norm=0.5, normalize_advantage=True,
                   verbose=1, seed=seed, device='cuda')
    elif algorithm_name == 'PPO':
        model = PPO('MlpPolicy', env, learning_rate=0.0003, n_steps=2048,
                   batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                   clip_range=0.2, ent_coef=0.0, vf_coef=0.5,
                   max_grad_norm=0.5, verbose=1, seed=seed, device='cuda')
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

    print(f"\nStarting training ({timesteps} timesteps)...")
    model.learn(total_timesteps=timesteps)
    training_time = time.time() - start_time

    # Save model
    model_path = save_dir / f'{algorithm_name}_seed{seed}_model.zip'
    model.save(str(model_path))

    # Evaluation
    print(f"\nEvaluation ({eval_episodes} episodes)...")
    eval_rewards = []
    eval_lengths = []
    eval_terminated_count = 0  # Actual crashes
    eval_truncated_count = 0   # Normal truncation
    eval_waiting_times = []
    eval_utilizations = []

    for ep in range(eval_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0
        ep_len = 0
        ep_waiting = []
        ep_utils = []
        episode_terminated = False
        episode_truncated = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            ep_reward += reward
            ep_len += 1

            if done:
                episode_terminated = term
                episode_truncated = trunc

            if 'avg_waiting_time' in info:
                ep_waiting.append(info['avg_waiting_time'])
            if 'utilization_rates' in info:
                ep_utils.append(np.mean(info['utilization_rates']))

        eval_rewards.append(ep_reward)
        eval_lengths.append(ep_len)

        if episode_terminated:
            eval_terminated_count += 1
            crash_marker = " 🔴[CRASHED]"
        elif episode_truncated:
            eval_truncated_count += 1
            crash_marker = " ✅[COMPLETED]"
        else:
            crash_marker = ""

        if ep_waiting:
            eval_waiting_times.append(np.mean(ep_waiting))
        if ep_utils:
            eval_utilizations.append(np.mean(ep_utils))

        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}: {ep_reward:.2f} (length {ep_len}){crash_marker}")

    # Calculate statistics
    mean_reward = np.mean(eval_rewards)
    std_reward = np.std(eval_rewards)
    terminated_rate = eval_terminated_count / eval_episodes
    truncated_rate = eval_truncated_count / eval_episodes
    mean_waiting = np.mean(eval_waiting_times) if eval_waiting_times else 0
    mean_util = np.mean(eval_utilizations) if eval_utilizations else 0
    mean_length = np.mean(eval_lengths)

    print(f"\n{'='*80}")
    print(f"Evaluation Results:")
    print(f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"  Best reward: {np.max(eval_rewards):.2f}")
    print(f"  🔴 Crash rate: {terminated_rate*100:.1f}% ({eval_terminated_count}/{eval_episodes})")
    print(f"  ✅ Completion rate: {truncated_rate*100:.1f}% ({eval_truncated_count}/{eval_episodes})")
    print(f"  Mean episode length: {mean_length:.1f}")
    print(f"  Training time: {training_time/60:.2f} minutes")
    print(f"{'='*80}")

    # Save results
    results = {
        'config_type': config_type,
        'algorithm': algorithm_name,
        'seed': seed,
        'layer_capacities': config.layer_capacities,
        'total_capacity': sum(config.layer_capacities),
        'arrival_weights': config.arrival_weights,
        'base_arrival_rate': float(config.base_arrival_rate),
        'high_load_multiplier': high_load_multiplier,
        'mean_reward': float(mean_reward),
        'std_reward': float(std_reward),
        'best_reward': float(np.max(eval_rewards)),
        'worst_reward': float(np.min(eval_rewards)),
        'crash_rate': float(terminated_rate),
        'completion_rate': float(truncated_rate),
        'terminated_count': eval_terminated_count,
        'truncated_count': eval_truncated_count,
        'mean_episode_length': float(mean_length),
        'mean_waiting_time': float(mean_waiting),
        'mean_utilization': float(mean_util),
        'training_time_minutes': float(training_time / 60),
        'eval_rewards': [float(r) for r in eval_rewards],
        'eval_lengths': [int(l) for l in eval_lengths],
        'timestamp': datetime.now().isoformat()
    }

    results_path = save_dir / f'{algorithm_name}_seed{seed}_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to: {results_path}\n")

    env.close()
    return results


def run_all_supplementary_experiments():
    """
    Run all 12 supplementary experiments

    Experiment Set 1: Structural comparison (8 runs)
    - Inverted vs Normal Pyramid
    - Algorithms: A2C, PPO
    - Seeds: 123, 456

    Experiment Set 2: Capacity paradox (4 runs)
    - K=10 vs K=30
    - Algorithm: A2C only
    - Seeds: 123, 456
    """

    # Define experiment matrix
    experiments = [
        # Set 1: Structural Comparison
        {'config': 'inverted_pyramid', 'algo': 'A2C', 'seeds': [123, 456]},
        {'config': 'inverted_pyramid', 'algo': 'PPO', 'seeds': [123, 456]},
        {'config': 'normal_pyramid', 'algo': 'A2C', 'seeds': [123, 456]},
        {'config': 'normal_pyramid', 'algo': 'PPO', 'seeds': [123, 456]},

        # Set 2: Capacity Paradox
        {'config': 'low_capacity', 'algo': 'A2C', 'seeds': [123, 456]},  # K=10
        {'config': 'capacity_30', 'algo': 'A2C', 'seeds': [123, 456]},   # K=30
    ]

    total_experiments = sum(len(exp['seeds']) for exp in experiments)
    print(f"\n{'='*80}")
    print(f"Supplementary experiment plan: Total {total_experiments} training runs")
    print(f"{'='*80}")

    # Run experiments
    all_results = []
    completed = 0

    for exp_config in experiments:
        config_type = exp_config['config']
        algorithm = exp_config['algo']
        seeds = exp_config['seeds']

        for seed in seeds:
            completed += 1
            print(f"\n\n{'#'*80}")
            print(f"Progress: [{completed}/{total_experiments}] {config_type} + {algorithm} (seed={seed})")
            print(f"{'#'*80}")

            try:
                result = train_and_evaluate(
                    algorithm_name=algorithm,
                    config_type=config_type,
                    timesteps=100000,
                    eval_episodes=50,
                    seed=seed,
                    high_load_multiplier=10.0
                )
                all_results.append(result)
                print(f"\n✅ [{completed}/{total_experiments}] Completed: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")

            except Exception as e:
                print(f"\n❌ [{completed}/{total_experiments}] Failed: {config_type} + {algorithm} (seed={seed})")
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()

    # Save summary
    summary = {
        'total_experiments': total_experiments,
        'completed': len(all_results),
        'failed': total_experiments - len(all_results),
        'timestamp': datetime.now().isoformat(),
        'experiments': all_results
    }

    summary_path = Path(project_root).parent / 'Data' / 'ablation_studies' / 'supplementary_n3' / 'EXPERIMENT_SUMMARY.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n\n{'='*80}")
    print(f"All supplementary experiments completed!")
    print(f"Success: {len(all_results)}/{total_experiments}")
    print(f"Summary saved to: {summary_path}")
    print(f"{'='*80}\n")

    return all_results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run supplementary experiments (n=1→n=3)')
    parser.add_argument('--mode', choices=['single', 'all'], default='all',
                       help='Run mode: single (single experiment) or all (all 12 runs)')
    parser.add_argument('--algorithm', choices=['A2C', 'PPO'],
                       help='Algorithm (single mode only)')
    parser.add_argument('--config',
                       choices=['inverted_pyramid', 'normal_pyramid', 'low_capacity', 'capacity_30'],
                       help='Configuration type (single mode only)')
    parser.add_argument('--seed', type=int, default=123,
                       help='Random seed (single mode only)')
    parser.add_argument('--timesteps', type=int, default=100000,
                       help='Training timesteps')
    parser.add_argument('--eval-episodes', type=int, default=50,
                       help='Evaluation episodes')

    args = parser.parse_args()

    if args.mode == 'all':
        print("\n🚀 Starting all supplementary experiments (12 training runs)...\n")
        run_all_supplementary_experiments()

    elif args.mode == 'single':
        if not args.algorithm or not args.config:
            print("❌ Error: single mode requires --algorithm and --config")
            parser.print_help()
        else:
            print(f"\n🚀 Running single experiment: {args.algorithm} + {args.config} (seed={args.seed})\n")
            result = train_and_evaluate(
                algorithm_name=args.algorithm,
                config_type=args.config,
                timesteps=args.timesteps,
                eval_episodes=args.eval_episodes,
                seed=args.seed
            )
            print(f"\n✅ Completed: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
