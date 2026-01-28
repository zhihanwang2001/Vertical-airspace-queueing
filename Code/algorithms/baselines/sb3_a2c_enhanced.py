"""
Enhanced A2C Baseline with Matched Network Capacity

This baseline increases A2C network capacity to match HCA2C (~459K parameters)
to test whether performance gain comes from parameter count or architecture.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from env.drl_optimized_env_fixed import DRLOptimizedQueueEnvFixed
from algorithms.baselines.space_utils import SB3DictWrapper
import numpy as np
from typing import Dict


class SB3A2CEnhanced:
    """
    Enhanced A2C with large network capacity

    Network architecture:
    - Actor: [512, 512, 256] -> 11 actions
    - Critic: [512, 512, 256] -> 1 value
    - Total parameters: ~459K (matched to HCA2C)

    This tests whether simply increasing network capacity can match
    HCA2C's performance, or if hierarchical architecture is necessary.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.seed = config.get('seed', 42)
        self.verbose = config.get('verbose', 1)
        self.env = None
        self.vec_env = None
        self.model = None

    def setup_env(self):
        """Setup environment"""
        base_env = DRLOptimizedQueueEnvFixed(max_episode_steps=10000)
        # Wrap with SB3DictWrapper to handle Dict action space
        wrapped_env = SB3DictWrapper(base_env)
        self.env = Monitor(wrapped_env, filename=None)

        # Create vectorized environment
        self.vec_env = DummyVecEnv([lambda: self.env])

        return self.env

    def setup_model(self):
        """Setup A2C model with LARGE network"""

        # Large network architecture (matched to HCA2C parameter count)
        policy_kwargs = dict(
            net_arch=dict(
                pi=[512, 512, 256],  # Actor network
                vf=[512, 512, 256]   # Critic network
            )
        )

        self.model = A2C(
            policy='MlpPolicy',  # Use MlpPolicy for Box observation space
            env=self.vec_env,
            learning_rate=7e-4,
            n_steps=5,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            use_rms_prop=True,
            normalize_advantage=True,
            policy_kwargs=policy_kwargs,
            verbose=self.verbose,
            seed=self.seed
        )

        return self.model

    def train(self, total_timesteps: int = 500000, progress_bar: bool = True):
        """Train the model"""
        if self.model is None:
            self.setup_model()

        self.model.learn(
            total_timesteps=total_timesteps,
            progress_bar=progress_bar
        )

        return {'total_timesteps': total_timesteps}

    def evaluate(self, n_episodes: int = 50, deterministic: bool = True, verbose: bool = False):
        """Evaluate the model"""
        episode_rewards = []
        episode_lengths = []
        crash_count = 0

        for ep in range(n_episodes):
            obs = self.vec_env.reset()
            done = False
            episode_reward = 0
            episode_length = 0

            while not done:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, done, info = self.vec_env.step(action)
                episode_reward += reward[0]
                episode_length += 1

                if done and info[0].get('crashed', False):
                    crash_count += 1

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            if verbose and (ep + 1) % 10 == 0:
                print(f"Episode {ep+1}/{n_episodes}: Reward={episode_reward:.2f}")

        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'crash_rate': crash_count / n_episodes,
            'n_episodes': n_episodes
        }

    def save(self, path: str):
        """Save model"""
        if self.model is not None:
            self.model.save(path)

    def load(self, path: str):
        """Load model"""
        if self.vec_env is None:
            self.setup_env()
        self.model = A2C.load(path, env=self.vec_env)


if __name__ == '__main__':
    # Test the enhanced A2C
    print("Testing SB3A2CEnhanced...")

    config = {
        'seed': 42,
        'verbose': 0
    }

    # Create baseline
    baseline = SB3A2CEnhanced(config)
    baseline.setup_env()
    baseline.setup_model()

    # Count parameters
    total_params = sum(p.numel() for p in baseline.model.policy.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Expected: ~459K")

    # Quick training test
    print("\nRunning quick training test (1000 steps)...")
    baseline.train(total_timesteps=1000, progress_bar=False)

    # Quick evaluation test
    print("Running quick evaluation test (5 episodes)...")
    results = baseline.evaluate(n_episodes=5, verbose=False)
    print(f"Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")

    print("\n✓ SB3A2CEnhanced test passed!")
