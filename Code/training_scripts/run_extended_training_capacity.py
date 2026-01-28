"""
Extended Training Capacity Paradox Experiment

Tests whether capacity paradox persists with extended training (500K timesteps)
for K=30 and K=40 at extreme load (10×).

This addresses reviewer concern: "Is the capacity paradox a training artifact?"

Experiment Design:
- Capacities: K ∈ {30, 40}
- Load: 10× (extreme load where paradox was observed)
- Training: 500,000 timesteps (5× original)
- Algorithms: A2C, PPO
- Seeds: 42, 43, 44, 45, 46 (n=5)
- Shape: uniform distribution

Outputs:
  - Data/optional_experiments/extended_training_results.csv
  - Data/optional_experiments/extended_training_summary.csv
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict
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


def make_config(total_capacity: int, load_multiplier: float) -> VerticalQueueConfig:
    """Create environment configuration."""
    cfg = VerticalQueueConfig()
    caps = build_capacities_uniform(total_capacity)
    cfg.layer_capacities = caps
    cfg.arrival_weights = [0.3, 0.25, 0.2, 0.15, 0.1]

    avg_service_rate = float(np.mean(cfg.layer_service_rates))
    base_rate = 0.75 * sum(caps) * avg_service_rate / 5.0
    cfg.base_arrival_rate = base_rate * load_multiplier
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


def run_experiment():
    """Run extended training experiment."""
    from stable_baselines3 import A2C, PPO

    # Experiment configuration
    capacities = [30, 40]
    load_multiplier = 10.0  # Extreme load
    timesteps = 500000  # Extended training
    eval_episodes = 50
    seeds = [42, 43, 44, 45, 46]
    algorithms = ['A2C', 'PPO']

    results_rows = []
    out_dir = Path('Data/optional_experiments')
    out_dir.mkdir(parents=True, exist_ok=True)

    total_runs = len(capacities) * len(algorithms) * len(seeds)
    current_run = 0

    print(f"🚀 Starting Extended Training Capacity Paradox Experiment")
    print(f"📊 Configuration:")
    print(f"   - Capacities: {capacities}")
    print(f"   - Load: {load_multiplier}×")
    print(f"   - Timesteps: {timesteps:,}")
    print(f"   - Algorithms: {algorithms}")
    print(f"   - Seeds: {seeds}")
    print(f"   - Total runs: {total_runs}")
    print()

    for K in capacities:
        cfg = make_config(K, load_multiplier)
        rl_env = wrap_for_rl(cfg)

        print(f"📦 Capacity K={K}, uniform={cfg.layer_capacities}")

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
                    'experiment': 'extended_training',
                    'algorithm': algo_name,
                    'total_capacity': K,
                    'layer_capacities': json.dumps(cfg.layer_capacities),
                    'load_multiplier': load_multiplier,
                    'timesteps': timesteps,
                    'seed': seed,
                    'training_time_sec': elapsed,
                    **stats
                })

                print(f"reward={stats['mean_reward']:.1f}, crash={stats['crash_rate']:.1%}, time={elapsed:.1f}s")

    # Save results
    df = pd.DataFrame(results_rows)
    raw_path = out_dir / 'extended_training_results.csv'
    df.to_csv(raw_path, index=False)

    # Save summary
    group_cols = ['algorithm', 'total_capacity', 'load_multiplier', 'timesteps']
    agg = df.groupby(group_cols).agg(
        n=('mean_reward', 'size'),
        mean_reward=('mean_reward', 'mean'),
        std_reward=('mean_reward', 'std'),
        mean_crash_rate=('crash_rate', 'mean'),
        mean_training_time=('training_time_sec', 'mean')
    ).reset_index()
    summary_path = out_dir / 'extended_training_summary.csv'
    agg.to_csv(summary_path, index=False)

    print()
    print(f"✅ Saved raw results to {raw_path}")
    print(f"✅ Saved summary to {summary_path}")
    print()
    print("📊 Summary Statistics:")
    print(agg.to_string(index=False))


if __name__ == '__main__':
    run_experiment()
