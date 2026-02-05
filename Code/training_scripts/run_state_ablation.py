"""
State Space Ablation Study for MCRPS/D/K System

This script runs ablation experiments to evaluate the impact of different
state space configurations on DRL performance.

Configurations:
- Minimal (15-dim): queue_lengths(5) + capacities(5) + utilization(5)
- Core (20-dim): Minimal + service_rates(5)
- Extended (25-dim): Core + arrival_rates(5)
- Full (29-dim): Extended + timestep + total_load + avg_wait + crash_flag
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Tuple

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


class AblationEnv(gym.Env):
    """
    Configurable state space environment for ablation study.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    # State space configurations
    STATE_CONFIGS = {
        'minimal': {
            'dim': 15,
            'features': ['queue_lengths', 'capacities', 'utilization'],
            'description': 'Queue lengths + Capacities + Utilization ratios'
        },
        'core': {
            'dim': 20,
            'features': ['queue_lengths', 'capacities', 'utilization', 'service_rates'],
            'description': 'Minimal + Service rates'
        },
        'extended': {
            'dim': 25,
            'features': ['queue_lengths', 'capacities', 'utilization', 'service_rates', 'arrival_rates'],
            'description': 'Core + Arrival rates'
        },
        'full': {
            'dim': 29,
            'features': ['queue_lengths', 'capacities', 'utilization', 'service_rates',
                        'arrival_rates', 'timestep', 'total_load', 'avg_wait', 'crash_flag'],
            'description': 'All features (default)'
        }
    }

    def __init__(self, state_config: str = 'full', render_mode: str = None,
                 max_episode_steps: int = 1000, load_multiplier: float = 5.0):
        super().__init__()

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.state_config = state_config
        self.load_multiplier = load_multiplier

        # System parameters
        self.n_layers = 5
        self.capacities = np.array([8, 6, 4, 3, 2], dtype=np.int32)
        self.base_arrival_rate = 0.3 * load_multiplier
        self.arrival_weights = np.array([0.30, 0.25, 0.20, 0.15, 0.10], dtype=np.float32)
        self.service_rates = np.array([1.2, 1.0, 0.8, 0.6, 0.4], dtype=np.float32)

        # State space dimension based on config
        config = self.STATE_CONFIGS[state_config]
        self.state_dim = config['dim']
        self.features = config['features']

        # Action space: 11-dimensional continuous
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(11,), dtype=np.float32
        )

        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.queue_lengths = np.zeros(self.n_layers, dtype=np.float32)
        self.step_count = 0
        self.total_served = 0
        self.total_arrived = 0
        self.avg_wait = 0.0
        self.crashed = False

        obs = self._get_observation()
        return obs, {}

    def _get_observation(self) -> np.ndarray:
        """Build observation based on state configuration."""
        obs_parts = []

        # Queue lengths (5)
        if 'queue_lengths' in self.features:
            obs_parts.append(self.queue_lengths / self.capacities.max())

        # Capacities (5)
        if 'capacities' in self.features:
            obs_parts.append(self.capacities.astype(np.float32) / self.capacities.max())

        # Utilization ratios (5)
        if 'utilization' in self.features:
            utilization = self.queue_lengths / self.capacities.astype(np.float32)
            obs_parts.append(utilization)

        # Service rates (5)
        if 'service_rates' in self.features:
            obs_parts.append(self.service_rates / self.service_rates.max())

        # Arrival rates (5)
        if 'arrival_rates' in self.features:
            arrival_rates = self.arrival_weights * self.base_arrival_rate
            obs_parts.append(arrival_rates / arrival_rates.max())

        # Timestep (1)
        if 'timestep' in self.features:
            obs_parts.append(np.array([self.step_count / self.max_episode_steps]))

        # Total load (1)
        if 'total_load' in self.features:
            total_load = self.queue_lengths.sum() / self.capacities.sum()
            obs_parts.append(np.array([total_load]))

        # Average wait (1)
        if 'avg_wait' in self.features:
            obs_parts.append(np.array([self.avg_wait / 10.0]))  # Normalized

        # Crash flag (1)
        if 'crash_flag' in self.features:
            obs_parts.append(np.array([1.0 if self.crashed else 0.0]))

        return np.concatenate(obs_parts).astype(np.float32)

    def step(self, action):
        self.step_count += 1

        # Parse action
        service_priorities = (action[:5] + 1) / 2  # [0, 1]
        transfer_decisions = action[5:9]  # [-1, 1]
        admission_control = (action[9:11] + 1) / 2  # [0, 1]

        # Arrivals
        total_arrivals = np.random.poisson(self.base_arrival_rate)
        self.total_arrived += total_arrivals

        if total_arrivals > 0:
            layer_arrivals = np.random.multinomial(total_arrivals, self.arrival_weights)
            for i in range(self.n_layers):
                # Admission control at boundaries
                if i == 0:
                    layer_arrivals[i] = int(layer_arrivals[i] * admission_control[0])
                elif i == 4:
                    layer_arrivals[i] = int(layer_arrivals[i] * admission_control[1])

                available = max(0, self.capacities[i] - self.queue_lengths[i])
                actual = min(layer_arrivals[i], available)
                self.queue_lengths[i] += actual

        # Service
        for i in range(self.n_layers):
            if self.queue_lengths[i] > 0:
                service_rate = self.service_rates[i] * (0.5 + service_priorities[i])
                served = min(np.random.poisson(service_rate), int(self.queue_lengths[i]))
                self.queue_lengths[i] -= served
                self.total_served += served

        # Transfers
        for i in range(self.n_layers - 1):
            if transfer_decisions[i] > 0.3 and self.queue_lengths[i] > 1:
                target_available = max(0, self.capacities[i+1] - self.queue_lengths[i+1])
                if target_available > 0:
                    transfer = min(int(self.queue_lengths[i] * 0.3), target_available)
                    self.queue_lengths[i] -= transfer
                    self.queue_lengths[i+1] += transfer

        # Check crash
        self.crashed = any(self.queue_lengths > self.capacities)

        # Calculate reward
        reward = self._calculate_reward()

        # Update avg wait estimate
        self.avg_wait = 0.9 * self.avg_wait + 0.1 * (self.queue_lengths.sum() / max(1, self.total_served))

        terminated = self.crashed
        truncated = self.step_count >= self.max_episode_steps

        obs = self._get_observation()
        info = {
            'total_served': self.total_served,
            'total_arrived': self.total_arrived,
            'crashed': self.crashed,
            'step': self.step_count
        }

        return obs, reward, terminated, truncated, info

    def _calculate_reward(self) -> float:
        """Calculate reward with hierarchical structure."""
        # Crash penalty (dominant)
        if self.crashed:
            return -10000.0

        # Throughput reward
        throughput = self.total_served / max(1, self.step_count)
        r_throughput = throughput * 10.0

        # Balance reward
        utilization = self.queue_lengths / self.capacities.astype(np.float32)
        r_balance = -np.std(utilization) * 5.0

        # Queue penalty
        r_queue = -self.queue_lengths.sum() * 0.1

        return r_throughput + r_balance + r_queue


def run_ablation_experiment(config_name: str, n_seeds: int = 3,
                           timesteps: int = 100000) -> Dict:
    """Run ablation experiment for a single configuration."""

    print(f"\n{'='*60}")
    print(f"Running ablation: {config_name}")
    print(f"State dimension: {AblationEnv.STATE_CONFIGS[config_name]['dim']}")
    print(f"Features: {AblationEnv.STATE_CONFIGS[config_name]['description']}")
    print(f"{'='*60}")

    results = {
        'config': config_name,
        'state_dim': AblationEnv.STATE_CONFIGS[config_name]['dim'],
        'features': AblationEnv.STATE_CONFIGS[config_name]['features'],
        'seeds': [],
        'rewards': [],
        'crash_rates': [],
        'throughputs': []
    }

    for seed in range(42, 42 + n_seeds):
        print(f"\n  Seed {seed}...")

        # Create environment
        env = DummyVecEnv([lambda: AblationEnv(state_config=config_name,
                                                max_episode_steps=1000,
                                                load_multiplier=5.0)])

        # Train A2C
        model = A2C(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=5,
            gamma=0.99,
            verbose=0,
            seed=seed
        )

        model.learn(total_timesteps=timesteps, progress_bar=True)

        # Evaluate
        eval_env = AblationEnv(state_config=config_name,
                               max_episode_steps=1000,
                               load_multiplier=5.0)

        episode_rewards = []
        crash_count = 0
        total_throughput = 0
        n_eval_episodes = 20

        for _ in range(n_eval_episodes):
            obs, _ = eval_env.reset(seed=seed + 100)
            done = False
            ep_reward = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                ep_reward += reward
                done = terminated or truncated

            episode_rewards.append(ep_reward)
            if info.get('crashed', False):
                crash_count += 1
            total_throughput += info.get('total_served', 0)

        mean_reward = np.mean(episode_rewards)
        crash_rate = crash_count / n_eval_episodes
        avg_throughput = total_throughput / n_eval_episodes

        results['seeds'].append(seed)
        results['rewards'].append(mean_reward)
        results['crash_rates'].append(crash_rate)
        results['throughputs'].append(avg_throughput)

        print(f"    Reward: {mean_reward:.2f}, Crash Rate: {crash_rate:.1%}, Throughput: {avg_throughput:.1f}")

        env.close()

    # Summary statistics
    results['mean_reward'] = float(np.mean(results['rewards']))
    results['std_reward'] = float(np.std(results['rewards']))
    results['mean_crash_rate'] = float(np.mean(results['crash_rates']))
    results['mean_throughput'] = float(np.mean(results['throughputs']))

    return results


def main():
    """Run full ablation study."""

    print("="*70)
    print("STATE SPACE ABLATION STUDY")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    configs = ['minimal', 'core', 'extended', 'full']
    all_results = {}

    for config in configs:
        results = run_ablation_experiment(
            config_name=config,
            n_seeds=3,
            timesteps=100000
        )
        all_results[config] = results

    # Save results
    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'Data', 'ablation_studies')
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, 'state_space_ablation_results.json')
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print summary
    print("\n" + "="*70)
    print("ABLATION STUDY SUMMARY")
    print("="*70)
    print(f"{'Config':<12} {'Dim':<6} {'Reward':<15} {'Crash Rate':<12} {'Throughput':<12}")
    print("-"*70)

    for config, results in all_results.items():
        print(f"{config:<12} {results['state_dim']:<6} "
              f"{results['mean_reward']:>8.2f} ± {results['std_reward']:<6.2f} "
              f"{results['mean_crash_rate']:>8.1%}    "
              f"{results['mean_throughput']:>8.1f}")

    print("-"*70)
    print(f"\nResults saved to: {output_file}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return all_results


if __name__ == "__main__":
    main()
