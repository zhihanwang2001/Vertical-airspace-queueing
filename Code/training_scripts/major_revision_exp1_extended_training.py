"""
Major Revision Experiment 1.1: Extended Training for K=30
Verify whether capacity paradox is an inherent system property or insufficient training budget

Key Question:
- Paper claims K=30 crashes after 100K training steps (100% crash rate)
- Reviewer questions: May be just insufficient training, not an inherent system issue

Experimental Design:
1. K=30 (uniform [6,6,6,6,6]) train for 1M steps (vs original 100K)
2. K=23 (inverted pyramid) train for 1M steps as control
3. K=10 (low capacity) train for 1M steps as baseline

Algorithms: A2C, PPO (main algorithms in original paper)
Each configuration: 3 seeds
Evaluation: Evaluate every 10K steps, using T=200 unified protocol

Expected Results:
- Best case: K=30 still crashes → capacity paradox is real
- Worst case: K=30 successfully converges → capacity paradox is insufficient training
- Most likely: K=30 partially improves but still worse than K=10 → nuanced conclusion
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
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from env.config import VerticalQueueConfig
from env.configurable_env_wrapper import ConfigurableEnvWrapper
from env.drl_wrapper_fixed import DictToBoxActionWrapperFixed, ObservationWrapperFixed


def create_config(capacity_type='k30_uniform', high_load_multiplier=10.0):
    """
    Create configuration

    Args:
        capacity_type:
            - 'k30_uniform': [6,6,6,6,6] total 30
            - 'k23_inverted': [8,6,4,3,2] total 23 (baseline)
            - 'k10_low': [2,2,2,2,2] total 10 (best performer in original)
        high_load_multiplier: Load multiplier (default 10.0)
    """
    config = VerticalQueueConfig()

    if capacity_type == 'k30_uniform':
        config.layer_capacities = [6, 6, 6, 6, 6]  # Total 30
        name = "K=30 Uniform"
    elif capacity_type == 'k23_inverted':
        config.layer_capacities = [8, 6, 4, 3, 2]  # Total 23
        name = "K=23 Inverted Pyramid"
    elif capacity_type == 'k10_low':
        config.layer_capacities = [2, 2, 2, 2, 2]  # Total 10
        name = "K=10 Low Capacity"
    else:
        raise ValueError(f"Unknown capacity type: {capacity_type}")

    # Fixed UAM traffic pattern (original paper setting)
    config.arrival_weights = [0.3, 0.25, 0.2, 0.15, 0.1]

    # 10× high load
    total_capacity = sum(config.layer_capacities)
    avg_service_rate = np.mean(config.layer_service_rates)
    base_rate_v3 = 0.75 * total_capacity * avg_service_rate / 5
    config.base_arrival_rate = base_rate_v3 * high_load_multiplier

    # Calculate theoretical load
    layer_loads = []
    for i, (w, c) in enumerate(zip(config.arrival_weights, config.layer_capacities)):
        layer_arrival = config.base_arrival_rate * w
        actual_service_rate = config.layer_service_rates[i]
        layer_load = layer_arrival / (c * actual_service_rate)
        layer_loads.append(layer_load)

    print(f"\n{'='*80}")
    print(f"Configuration: {name}")
    print(f"Capacity: {config.layer_capacities} (Total: {total_capacity})")
    print(f"Arrival weights: {config.arrival_weights}")
    print(f"Total arrival rate: {config.base_arrival_rate:.2f}")
    print(f"\nTheoretical load per layer:")
    for i, load in enumerate(layer_loads):
        status = "RED" if load >= 1.0 else "YELLOW" if load > 0.8 else "GREEN"
        print(f"  L{i}: {load*100:.1f}% {status}")
    print(f"Average load: {np.mean(layer_loads)*100:.1f}%")
    print(f"Maximum load: {np.max(layer_loads)*100:.1f}%")
    print(f"{'='*80}\n")

    return config, name


def create_env(config):
    """Create environment"""
    base_env = ConfigurableEnvWrapper(config)
    wrapped_env = DictToBoxActionWrapperFixed(base_env)
    env = ObservationWrapperFixed(wrapped_env)
    return env


def train_and_evaluate(
    algo_name='A2C',
    capacity_type='k30_uniform',
    seed=42,
    total_timesteps=1_000_000,  # 1M steps (vs original 100K)
    eval_freq=10_000,  # Evaluate every 10K
    n_eval_episodes=10
):
    """
    Train and evaluate

    Key parameters:
    - total_timesteps: 1M (10× original)
    - eval_freq: 10K (vs original 5K)
    - max_episode_steps: 200 (unified protocol, consistent with original paper A2C/PPO)
    """

    # Create configuration
    config, config_name = create_config(capacity_type)

    # Set output directory
    output_dir = Path(f"Results/major_revision_exp1/{capacity_type}/{algo_name}_seed{seed}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Training: {algo_name} on {config_name}")
    print(f"Seed: {seed}")
    print(f"Total steps: {total_timesteps:,}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")

    # Create training and evaluation environments
    train_env = create_env(config)
    eval_env = create_env(config)

    # Set episode length (unified protocol)
    train_env.env.env._max_episode_steps = 1000  # Longer for training
    eval_env.env.env._max_episode_steps = 200    # Unified T=200 for evaluation

    # Create algorithm
    if algo_name == 'A2C':
        # Use staged learning rate from original paper
        # But adjust transition point for 1M steps
        model = A2C(
            "MlpPolicy",
            train_env,
            learning_rate=7e-4,  # Initial high learning rate
            n_steps=32,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=[512, 512, 256]),
            verbose=1,
            seed=seed,
            device='auto'
        )
    elif algo_name == 'PPO':
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            policy_kwargs=dict(net_arch=[256, 256]),
            verbose=1,
            seed=seed,
            device='auto'
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best_model"),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1
    )

    # Create checkpoint callback (save every 50K)
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=str(output_dir / "checkpoints"),
        name_prefix=f"{algo_name}_checkpoint"
    )

    # Training
    print(f"\n{'='*80}")
    print(f"Starting training...")
    print(f"{'='*80}\n")

    start_time = time.time()

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback],
            progress_bar=True
        )

        training_time = time.time() - start_time

        print(f"\n{'='*80}")
        print(f"Training complete!")
        print(f"Time taken: {training_time/60:.1f} minutes")
        print(f"{'='*80}\n")

        # Save final model
        model.save(output_dir / "final_model")

        # Final evaluation (T=200)
        print(f"\n{'='*80}")
        print(f"Final evaluation (T=200, {n_eval_episodes} episodes)...")
        print(f"{'='*80}\n")

        eval_env.env.env._max_episode_steps = 200

        episode_rewards = []
        episode_lengths = []
        crash_count = 0

        for ep in range(n_eval_episodes):
            obs, _ = eval_env.reset()
            episode_reward = 0
            episode_length = 0
            done = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1

                if terminated and episode_length < 200:
                    crash_count += 1
                    break

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            print(f"  Episode {ep+1}: Reward={episode_reward:.1f}, Length={episode_length}")

        # Calculate statistics
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_length = np.mean(episode_lengths)
        crash_rate = crash_count / n_eval_episodes * 100
        completion_rate = 100 - crash_rate

        results = {
            'algorithm': algo_name,
            'capacity_type': capacity_type,
            'config_name': config_name,
            'seed': seed,
            'total_timesteps': total_timesteps,
            'training_time_minutes': training_time / 60,

            'final_eval': {
                'episode_steps': 200,
                'n_episodes': n_eval_episodes,
                'mean_reward': float(mean_reward),
                'std_reward': float(std_reward),
                'mean_episode_length': float(mean_length),
                'crash_rate': float(crash_rate),
                'completion_rate': float(completion_rate),
                'all_rewards': [float(r) for r in episode_rewards],
                'all_lengths': [int(l) for l in episode_lengths]
            },

            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'purpose': 'Major Revision Exp 1.1: Extended Training',
                'hypothesis_test': 'Capacity paradox: inherent vs training budget'
            }
        }

        # Save results
        results_file = output_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n{'='*80}")
        print(f"Final results:")
        print(f"  Mean reward: {mean_reward:.1f} ± {std_reward:.1f}")
        print(f"  Mean length: {mean_length:.1f}")
        print(f"  Crash rate: {crash_rate:.1f}%")
        print(f"  Completion rate: {completion_rate:.1f}%")
        print(f"{'='*80}\n")

        print(f"Results saved to: {results_file}")

        return results

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        train_env.close()
        eval_env.close()


def main():
    """
    Main function: Run all configurations

    Priority:
    1. K=30 (critical) - Verify capacity paradox
    2. K=23 (control) - Confirm extended training doesn't break known results
    3. K=10 (baseline) - Verify if optimal configuration improves further
    """

    configurations = [
        # Most critical: K=30
        ('A2C', 'k30_uniform', 42),
        ('A2C', 'k30_uniform', 123),
        ('A2C', 'k30_uniform', 456),

        ('PPO', 'k30_uniform', 42),
        ('PPO', 'k30_uniform', 123),
        ('PPO', 'k30_uniform', 456),

        # Control: K=23
        ('A2C', 'k23_inverted', 42),
        ('A2C', 'k23_inverted', 123),

        ('PPO', 'k23_inverted', 42),
        ('PPO', 'k23_inverted', 123),

        # Baseline: K=10
        ('A2C', 'k10_low', 42),
        ('PPO', 'k10_low', 42),
    ]

    print(f"\n{'#'*80}")
    print(f"# Major Revision Experiment 1.1: Extended Training")
    print(f"# Total configurations: {len(configurations)}")
    print(f"# Estimated time: ~{len(configurations) * 2} hours (can be reduced with parallel execution)")
    print(f"{'#'*80}\n")

    all_results = []

    for i, (algo, capacity, seed) in enumerate(configurations):
        print(f"\n{'#'*80}")
        print(f"# Configuration {i+1}/{len(configurations)}")
        print(f"{'#'*80}\n")

        result = train_and_evaluate(
            algo_name=algo,
            capacity_type=capacity,
            seed=seed,
            total_timesteps=1_000_000,
            eval_freq=10_000,
            n_eval_episodes=10
        )

        if result:
            all_results.append(result)

    # Save summary results
    summary_file = Path("Results/major_revision_exp1/summary.json")
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'#'*80}")
    print(f"# All experiments complete!")
    print(f"# Summary results: {summary_file}")
    print(f"{'#'*80}\n")

    # Quick analysis
    print("\nQuick analysis:")
    print("="*80)

    for capacity_type in ['k30_uniform', 'k23_inverted', 'k10_low']:
        relevant = [r for r in all_results if r['capacity_type'] == capacity_type]
        if not relevant:
            continue

        print(f"\n{relevant[0]['config_name']}:")

        for algo in ['A2C', 'PPO']:
            algo_results = [r for r in relevant if r['algorithm'] == algo]
            if not algo_results:
                continue

            rewards = [r['final_eval']['mean_reward'] for r in algo_results]
            crash_rates = [r['final_eval']['crash_rate'] for r in algo_results]

            print(f"  {algo}:")
            print(f"    Reward: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
            print(f"    Crash:  {np.mean(crash_rates):.1f}%")

    print("\n" + "="*80)
    print("Analysis complete! Please review detailed results for paper revision.")


if __name__ == "__main__":
    main()
