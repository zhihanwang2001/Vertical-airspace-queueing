"""
Reward Function Sensitivity Analysis Experiment

Tests whether main findings are robust to different reward function weights.

This addresses reviewer concern: "How were reward weights chosen?
Are results robust to weight variations?"

Experiment Design:
- Capacity: K=10 (uniform)
- Load: 5× (moderate load)
- Training: 100,000 timesteps
- Algorithms: A2C, PPO
- Seeds: 42, 43, 44 (n=3)
- Weight configurations: 4 variants

Weight Configurations:
1. Baseline (current): throughput=1.0, waiting=-0.1, queue=-0.05, balance=0.5, transfer=0.2
2. Throughput-focused: throughput=2.0, waiting=-0.05, queue=-0.025, balance=0.25, transfer=0.1
3. Balance-focused: throughput=0.5, waiting=-0.05, queue=-0.025, balance=1.0, transfer=0.1
4. Efficiency-focused: throughput=1.0, waiting=-0.2, queue=-0.1, balance=0.25, transfer=0.1

Outputs:
  - Data/optional_experiments/reward_sensitivity_results.csv
  - Data/optional_experiments/reward_sensitivity_summary.csv
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
import numpy as np
import pandas as pd
import time

# Ensure repository Code/ is on sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from env.config import VerticalQueueConfig
from env.configurable_env_wrapper import ConfigurableEnvWrapper
from env.drl_wrapper_fixed import DictToBoxActionWrapperFixed, ObservationWrapperFixed


def build_capacities_uniform(total: int) -> List[int]:
    """Distribute total capacity uniformly across 5 layers."""
    base = total // 5
    rem = total % 5
    caps = [base] * 5
    for i in range(rem):
        caps[i] += 1
    return caps


def make_config(total_capacity: int, load_multiplier: float,
                reward_weights: Dict[str, float]) -> VerticalQueueConfig:
    """Create environment configuration with custom reward weights."""
    cfg = VerticalQueueConfig()
    caps = build_capacities_uniform(total_capacity)
    cfg.layer_capacities = caps
    cfg.arrival_weights = [0.3, 0.25, 0.2, 0.15, 0.1]

    avg_service_rate = float(np.mean(cfg.layer_service_rates))
    base_rate = 0.75 * sum(caps) * avg_service_rate / 5.0
    cfg.base_arrival_rate = base_rate * load_multiplier

    # Override reward weights
    cfg.reward_throughput = reward_weights['throughput']
    cfg.reward_waiting_time = reward_weights['waiting']
    cfg.reward_queue_length = reward_weights['queue']
    cfg.reward_balance = reward_weights['balance']
    cfg.reward_transfer_success = reward_weights['transfer']
    cfg.reward_transfer_fail = -abs(reward_weights['transfer']) / 2.0
    # Crash penalty stays constant
    cfg.reward_crash = -10000.0

    return cfg


def wrap_for_rl(cfg: VerticalQueueConfig):
    """Wrap environment for RL training."""
    base = ConfigurableEnvWrapper(cfg)
    wrapped = DictToBoxActionWrapperFixed(base)
    wrapped = ObservationWrapperFixed(wrapped)
    return wrapped


def eval_rl(model, env, n_episodes: int, seed: int) -> Dict:
    """Evaluate trained RL model."""
    rewards = []
    lengths = []
    crashes = 0

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        ep_r = 0.0
        ep_len = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            ep_r += float(reward)
            ep_len += 1
            done = bool(term or trunc)
            if done and term:
                crashes += 1

        rewards.append(ep_r)
        lengths.append(ep_len)

    return {
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'mean_length': float(np.mean(lengths)),
        'crash_rate': float(crashes / n_episodes)
    }


def get_weight_configurations() -> Dict[str, Dict[str, float]]:
    """Define reward weight configurations to test."""
    return {
        'baseline': {
            'throughput': 1.0,
            'waiting': -0.1,
            'queue': -0.05,
            'balance': 0.5,
            'transfer': 0.2
        },
        'throughput_focused': {
            'throughput': 2.0,
            'waiting': -0.05,
            'queue': -0.025,
            'balance': 0.25,
            'transfer': 0.1
        },
        'balance_focused': {
            'throughput': 0.5,
            'waiting': -0.05,
            'queue': -0.025,
            'balance': 1.0,
            'transfer': 0.1
        },
        'efficiency_focused': {
            'throughput': 1.0,
            'waiting': -0.2,
            'queue': -0.1,
            'balance': 0.25,
            'transfer': 0.1
        }
    }


def run_experiment():
    """Run reward sensitivity experiment."""
    from stable_baselines3 import A2C, PPO

    # Experiment configuration
    capacity = 10
    load_multiplier = 5.0
    timesteps = 100000
    eval_episodes = 50
    seeds = [42, 43, 44]
    algorithms = ['A2C', 'PPO']
    weight_configs = get_weight_configurations()

    results_rows = []
    out_dir = Path('Data/optional_experiments')
    out_dir.mkdir(parents=True, exist_ok=True)

    total_runs = len(weight_configs) * len(algorithms) * len(seeds)
    current_run = 0

    print(f"🚀 Starting Reward Function Sensitivity Analysis")
    print(f"📊 Configuration:")
    print(f"   - Capacity: K={capacity}")
    print(f"   - Load: {load_multiplier}×")
    print(f"   - Timesteps: {timesteps:,}")
    print(f"   - Algorithms: {algorithms}")
    print(f"   - Seeds: {seeds}")
    print(f"   - Weight configs: {list(weight_configs.keys())}")
    print(f"   - Total runs: {total_runs}")
    print()

    for config_name, weights in weight_configs.items():
        print(f"🎯 Testing weight configuration: {config_name}")
        print(f"   Weights: {weights}")

        cfg = make_config(capacity, load_multiplier, weights)
        rl_env = wrap_for_rl(cfg)

        for algo_name in algorithms:
            for seed in seeds:
                current_run += 1
                start_time = time.time()

                print(f"  [{current_run}/{total_runs}] {algo_name} seed={seed}...", end=' ', flush=True)

                # Create model
                if algo_name == 'A2C':
                    model = A2C('MlpPolicy', rl_env,
                                learning_rate=0.0007, n_steps=32,
                                gamma=0.99, gae_lambda=0.95, ent_coef=0.01, vf_coef=0.5,
                                max_grad_norm=0.5, normalize_advantage=True,
                                verbose=0, seed=seed, device='auto')
                elif algo_name == 'PPO':
                    model = PPO('MlpPolicy', rl_env,
                                learning_rate=0.0003, n_steps=2048,
                                batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                                clip_range=0.2, ent_coef=0.0, vf_coef=0.5,
                                max_grad_norm=0.5, verbose=0, seed=seed, device='auto')

                # Train
                model.learn(total_timesteps=timesteps)

                # Evaluate
                stats = eval_rl(model, rl_env, n_episodes=eval_episodes, seed=seed)

                elapsed = time.time() - start_time

                results_rows.append({
                    'experiment': 'reward_sensitivity',
                    'weight_config': config_name,
                    'algorithm': algo_name,
                    'total_capacity': capacity,
                    'load_multiplier': load_multiplier,
                    'timesteps': timesteps,
                    'seed': seed,
                    'training_time_sec': elapsed,
                    **{f'weight_{k}': v for k, v in weights.items()},
                    **stats
                })

                print(f"reward={stats['mean_reward']:.1f}, crash={stats['crash_rate']:.1%}, time={elapsed:.1f}s")

        print()

    # Save results
    df = pd.DataFrame(results_rows)
    raw_path = out_dir / 'reward_sensitivity_results.csv'
    df.to_csv(raw_path, index=False)

    # Save summary
    group_cols = ['weight_config', 'algorithm']
    agg = df.groupby(group_cols).agg(
        n=('mean_reward', 'size'),
        mean_reward=('mean_reward', 'mean'),
        std_reward=('mean_reward', 'std'),
        mean_crash_rate=('crash_rate', 'mean'),
        mean_training_time=('training_time_sec', 'mean')
    ).reset_index()
    summary_path = out_dir / 'reward_sensitivity_summary.csv'
    agg.to_csv(summary_path, index=False)

    print()
    print(f"✅ Saved raw results to {raw_path}")
    print(f"✅ Saved summary to {summary_path}")
    print()
    print("📊 Summary Statistics:")
    print(agg.to_string(index=False))


if __name__ == '__main__':
    run_experiment()
