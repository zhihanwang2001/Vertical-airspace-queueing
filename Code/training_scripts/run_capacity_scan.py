"""
Capacity Scan Experiments (K ∈ {10,15,20,25,30}, load ∈ {3×,5×})

Runs RL (A2C/PPO) with SB3 and Heuristics (FCFS/SJF/Priority/Heuristic)
across capacity totals and load multipliers, saving tidy CSV suitable
for statistical analysis and figure generation.

Outputs:
  - Data/summary/capacity_scan_results.csv  (per-run rows)
  - Data/summary/capacity_scan_summary.csv  (grouped means/stds)
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple
import argparse
import numpy as np
import pandas as pd

# Ensure repository Code/ is on sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from env.config import VerticalQueueConfig
from env.configurable_env_wrapper import ConfigurableEnvWrapper
from env.drl_wrapper_fixed import DictToBoxActionWrapperFixed, ObservationWrapperFixed


def build_capacities_uniform(total: int) -> List[int]:
    base = total // 5
    rem = total % 5
    caps = [base] * 5
    for i in range(rem):
        caps[i] += 1
    return caps


def build_capacities_scaled(total: int, base_pattern: List[int]) -> List[int]:
    base_sum = sum(base_pattern)
    if base_sum == 0:
        return build_capacities_uniform(total)
    scale = total / base_sum
    raw = [int(round(x * scale)) for x in base_pattern]
    diff = total - sum(raw)
    # Distribute remainder from the top layer down
    idx = 0
    while diff != 0:
        raw[idx % 5] += 1 if diff > 0 else -1
        diff += -1 if diff > 0 else 1
        idx += 1
    return raw


def make_config(total_capacity: int, shape: str, load_multiplier: float) -> VerticalQueueConfig:
    cfg = VerticalQueueConfig()

    if shape == 'uniform':
        caps = build_capacities_uniform(total_capacity)
    elif shape == 'inverted':
        caps = build_capacities_scaled(total_capacity, [8, 6, 4, 3, 2])
    elif shape == 'reverse':
        caps = build_capacities_scaled(total_capacity, [2, 3, 4, 6, 8])
    else:
        raise ValueError(f"Unknown shape: {shape}")

    cfg.layer_capacities = caps  # overwrite default after __post_init__
    cfg.arrival_weights = [0.3, 0.25, 0.2, 0.15, 0.1]

    avg_service_rate = float(np.mean(cfg.layer_service_rates))
    base_rate = 0.75 * sum(caps) * avg_service_rate / 5.0
    cfg.base_arrival_rate = base_rate * load_multiplier
    return cfg


def wrap_for_rl(cfg: VerticalQueueConfig):
    base = ConfigurableEnvWrapper(cfg)
    wrapped = DictToBoxActionWrapperFixed(base)
    wrapped = ObservationWrapperFixed(wrapped)
    return wrapped


def make_env_for_heuristic(cfg: VerticalQueueConfig):
    return ConfigurableEnvWrapper(cfg)


def parse_list(s: str, cast=str) -> List:
    return [cast(x.strip()) for x in s.split(',') if x.strip()]


def eval_rl(model, env, n_episodes: int, seed: int) -> Dict:
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


def eval_heuristic(policy, env, n_episodes: int, seed: int) -> Dict:
    rewards = []
    lengths = []
    crashes = 0
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        ep_r = 0.0
        ep_len = 0
        while not done:
            action, _ = policy.predict(obs, deterministic=True)
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


def run(args):
    # Heuristic baselines are always available
    from algorithms.baselines.traditional_baselines_fixed import FCFSBaseline, SJFBaseline, PriorityBaseline
    from algorithms.baselines.heuristic_baseline import HeuristicBaseline

    capacities = [int(x) for x in parse_list(args.capacities, int)]
    loads = [float(x) for x in parse_list(args.loads, float)]
    rl_algos = parse_list(args.algos)
    include_heur = args.include_heuristics

    # Seeds
    if args.seeds:
        seeds = [int(x) for x in parse_list(args.seeds, int)]
    elif args.n_seeds:
        seeds = list(range(42, 42 + int(args.n_seeds)))
    else:
        seeds = [42, 123, 456, 789, 101112]

    results_rows = []

    out_dir = Path('Data/summary')
    out_dir.mkdir(parents=True, exist_ok=True)

    for K in capacities:
        for lm in loads:
            cfg = make_config(K, args.shape, lm)

            # Heuristics
            if include_heur:
                heur_env = make_env_for_heuristic(cfg)
                heuristics = [
                    ('Heuristic', HeuristicBaseline(heur_env)),
                    ('FCFS', FCFSBaseline(heur_env)),
                    ('SJF', SJFBaseline(heur_env)),
                    ('Priority', PriorityBaseline(heur_env))
                ]
                for algo_name, baseline in heuristics:
                    for seed in seeds:
                        stats = eval_heuristic(baseline, heur_env, n_episodes=args.eval_episodes, seed=seed)
                        results_rows.append({
                            'family': 'Heuristic',
                            'algorithm': algo_name,
                            'shape': args.shape,
                            'total_capacity': K,
                            'layer_capacities': json.dumps(cfg.layer_capacities),
                            'load_multiplier': lm,
                            'seed': seed,
                            **stats
                        })

            # RL
            rl_env = None
            if rl_algos:
                # Import SB3 lazily only if RL is requested
                try:
                    from stable_baselines3 import A2C, PPO
                except Exception as e:
                    raise RuntimeError("stable-baselines3 not installed. Install or pass --algos '' to run heuristics only.") from e
                rl_env = wrap_for_rl(cfg)
            for algo_name in rl_algos:
                for seed in seeds:
                    if algo_name.upper() == 'A2C':
                        model = A2C('MlpPolicy', rl_env,
                                    learning_rate=0.0007, n_steps=32,
                                    gamma=0.99, gae_lambda=0.95, ent_coef=0.01, vf_coef=0.5,
                                    max_grad_norm=0.5, normalize_advantage=True,
                                    verbose=0, seed=seed, device='auto')
                    elif algo_name.upper() == 'PPO':
                        model = PPO('MlpPolicy', rl_env,
                                    learning_rate=0.0003, n_steps=2048,
                                    batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                                    clip_range=0.2, ent_coef=0.0, vf_coef=0.5,
                                    max_grad_norm=0.5, verbose=0, seed=seed, device='auto')
                    else:
                        raise ValueError(f"Unsupported RL algo: {algo_name}")

                    model.learn(total_timesteps=args.timesteps)
                    stats = eval_rl(model, rl_env, n_episodes=args.eval_episodes, seed=seed)
                    results_rows.append({
                        'family': 'RL',
                        'algorithm': algo_name.upper(),
                        'shape': args.shape,
                        'total_capacity': K,
                        'layer_capacities': json.dumps(cfg.layer_capacities),
                        'load_multiplier': lm,
                        'seed': seed,
                        **stats
                    })

    # Save raw
    df = pd.DataFrame(results_rows)
    # Include shape and load info in filename to avoid overwriting
    load_str = '_'.join(str(int(lm)) if lm == int(lm) else str(lm) for lm in loads)
    raw_path = out_dir / f'capacity_scan_results_{args.shape}_{load_str}.csv'
    df.to_csv(raw_path, index=False)

    # Save grouped summary
    group_cols = ['family', 'algorithm', 'shape', 'total_capacity', 'load_multiplier']
    agg = df.groupby(group_cols).agg(
        n=('mean_reward', 'size'),
        mean_reward=('mean_reward', 'mean'),
        std_reward=('mean_reward', 'std'),
        mean_crash_rate=('crash_rate', 'mean')
    ).reset_index()
    summary_path = out_dir / f'capacity_scan_summary_{args.shape}_{load_str}.csv'
    agg.to_csv(summary_path, index=False)

    print(f"✅ Saved raw to {raw_path}")
    print(f"✅ Saved summary to {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Capacity scan across K and load multipliers')
    parser.add_argument('--capacities', type=str, default='10,15,20,25,30',
                        help='Comma-separated total capacities, e.g., 10,15,20,25,30')
    parser.add_argument('--loads', type=str, default='3,5',
                        help='Comma-separated load multipliers, e.g., 3,5')
    parser.add_argument('--shape', type=str, choices=['uniform', 'inverted', 'reverse'], default='uniform',
                        help='Capacity shape used when distributing K across layers')
    parser.add_argument('--algos', type=str, default='A2C,PPO',
                        help='Comma-separated RL algos to run (A2C,PPO)')
    parser.add_argument('--include-heuristics', action='store_true', help='Include heuristic baselines')
    parser.add_argument('--timesteps', type=int, default=100000, help='Training timesteps for RL')
    parser.add_argument('--eval-episodes', type=int, default=50, help='Evaluation episodes per run')
    parser.add_argument('--seeds', type=str, default=None, help='Comma-separated seeds (e.g., 42,43,44)')
    parser.add_argument('--n-seeds', type=int, default=None, help='Number of seeds to auto-generate starting at 42')

    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
