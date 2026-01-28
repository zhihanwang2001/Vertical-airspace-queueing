"""
HCA2C Baseline - Compatible interface with the comparison framework

Provides a standardized interface for comparing HCA2C with other algorithms
like A2C, PPO, etc.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import json
import time
from typing import Dict, Optional, Tuple

from .hca2c_agent import HCA2C
from .wrapper import HierarchicalEnvWrapper
from env.drl_optimized_env_fixed import DRLOptimizedQueueEnvFixed


class HCA2CBaseline:
    """
    HCA2C Baseline for comparison framework

    Provides the same interface as SB3 baselines for fair comparison.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize HCA2C baseline

        Args:
            config: Configuration dictionary
        """
        default_config = {
            # Learning parameters
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'n_steps': 32,

            # Loss coefficients
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,

            # Network architecture
            'global_hidden_dim': 256,
            'layer_hidden_dim': 128,

            # Environment
            'load_multiplier': 5.0,
            'max_episode_steps': 10000,

            # Other
            'seed': 42,
            'verbose': 1,
            'device': 'auto'
        }

        if config:
            default_config.update(config)

        self.config = default_config
        self.model = None
        self.env = None
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'avg_rewards': [],
            'loss_values': [],
            'training_time': []
        }

    def setup_env(self, load_multiplier: float = None):
        """Setup environment with hierarchical wrapper"""
        if load_multiplier is None:
            load_multiplier = self.config.get('load_multiplier', 5.0)

        # Create base environment
        base_env = DRLOptimizedQueueEnvFixed(
            max_episode_steps=self.config.get('max_episode_steps', 10000)
        )

        # Modify arrival rate for load testing
        base_env.base_arrival_rate = 0.3 * load_multiplier

        # Wrap with hierarchical wrapper
        self.env = HierarchicalEnvWrapper(base_env)

        return self.env

    def create_model(self):
        """Create HCA2C model"""
        if self.env is None:
            self.setup_env()

        self.model = HCA2C(
            env=self.env,
            learning_rate=self.config['learning_rate'],
            gamma=self.config['gamma'],
            gae_lambda=self.config['gae_lambda'],
            n_steps=self.config['n_steps'],
            ent_coef=self.config['ent_coef'],
            vf_coef=self.config['vf_coef'],
            max_grad_norm=self.config['max_grad_norm'],
            global_hidden_dim=self.config['global_hidden_dim'],
            layer_hidden_dim=self.config['layer_hidden_dim'],
            device=self.config['device'],
            seed=self.config['seed'],
            verbose=self.config['verbose']
        )

        print(f"HCA2C model created with device: {self.model.device}")
        return self.model

    def train(self, total_timesteps: int, eval_freq: int = 10000,
              save_freq: int = 50000, progress_bar: bool = True) -> Dict:
        """
        Train the model

        Args:
            total_timesteps: Total training timesteps
            eval_freq: Evaluation frequency
            save_freq: Checkpoint save frequency
            progress_bar: Show progress bar

        Returns:
            Training results dictionary
        """
        if self.model is None:
            self.create_model()

        start_time = time.time()

        print(f"Starting HCA2C training for {total_timesteps:,} timesteps...")

        # Training callback to collect history
        def training_callback(model):
            if len(model.ep_rewards) > 0:
                self.training_history['episode_rewards'].append(
                    float(np.mean(list(model.ep_rewards)[-10:]))
                )

        # Train
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=training_callback,
            log_interval=100,
            progress_bar=progress_bar
        )

        training_time = time.time() - start_time
        self.training_history['training_time'].append(training_time)

        print(f"HCA2C training completed in {training_time:.1f}s!")

        return {
            'episodes': self.model.num_episodes,
            'total_timesteps': self.model.num_timesteps,
            'training_time': training_time,
            'final_reward': np.mean(self.model.ep_rewards) if self.model.ep_rewards else 0
        }

    def evaluate(self, n_episodes: int = 10, deterministic: bool = True,
                 verbose: bool = True) -> Dict:
        """
        Evaluate the model

        Args:
            n_episodes: Number of evaluation episodes
            deterministic: Use deterministic actions
            verbose: Print progress

        Returns:
            Evaluation results dictionary
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")

        return self.model.evaluate(
            n_episodes=n_episodes,
            deterministic=deterministic,
            verbose=verbose
        )

    def predict(self, observation, deterministic: bool = True) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Predict action

        Args:
            observation: Environment observation
            deterministic: Use deterministic action

        Returns:
            (action, info)
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")

        return self.model.predict(observation, deterministic=deterministic)

    def save(self, path: str):
        """Save model"""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        self.model.save(path)

    def load(self, path: str):
        """Load model"""
        if self.env is None:
            self.setup_env()

        if self.model is None:
            self.create_model()

        self.model.load(path)

    def save_results(self, path_prefix: str):
        """Save training history and results"""
        os.makedirs(os.path.dirname(path_prefix) if os.path.dirname(path_prefix) else ".", exist_ok=True)

        # Save training history
        with open(f"{path_prefix}_history.json", 'w') as f:
            json.dump(self.training_history, f, indent=2)

        # Save config
        with open(f"{path_prefix}_config.json", 'w') as f:
            json.dump(self.config, f, indent=2)

        print(f"HCA2C results saved to: {path_prefix}")

    def get_info(self) -> Dict:
        """Get algorithm information"""
        return {
            'algorithm_name': 'HCA2C',
            'total_timesteps': self.model.num_timesteps if self.model else 0,
            'episode_count': self.model.num_episodes if self.model else 0,
            'config': self.config,
            'training_completed': self.model is not None and self.model.num_timesteps > 0
        }


def run_hca2c_experiment(
    total_timesteps: int = 500000,
    load_multiplier: float = 5.0,
    seed: int = 42,
    save_path: str = None
) -> Dict:
    """
    Run a single HCA2C experiment

    Args:
        total_timesteps: Training timesteps
        load_multiplier: Load multiplier for environment
        seed: Random seed
        save_path: Path to save model

    Returns:
        Experiment results
    """
    config = {
        'seed': seed,
        'load_multiplier': load_multiplier,
        'verbose': 1
    }

    baseline = HCA2CBaseline(config)
    baseline.setup_env(load_multiplier=load_multiplier)

    # Train
    train_results = baseline.train(total_timesteps=total_timesteps)

    # Evaluate
    eval_results = baseline.evaluate(n_episodes=20, deterministic=True)

    # Save if path provided
    if save_path:
        baseline.save(save_path)
        baseline.save_results(save_path.replace('.pt', ''))

    return {
        'train': train_results,
        'eval': eval_results,
        'config': config
    }


if __name__ == "__main__":
    print("Testing HCA2CBaseline...")

    # Quick test
    baseline = HCA2CBaseline({
        'seed': 42,
        'verbose': 1,
        'load_multiplier': 5.0
    })

    # Setup and create
    baseline.setup_env()
    baseline.create_model()

    print(f"\n✅ Baseline created successfully!")
    print(f"   Config: {baseline.config}")

    # Short training test
    print("\n\nRunning short training test (5000 steps)...")
    results = baseline.train(total_timesteps=5000, progress_bar=False)
    print(f"✅ Training completed: {results}")

    # Evaluation test
    print("\n\nRunning evaluation test...")
    eval_results = baseline.evaluate(n_episodes=3, verbose=True)
    print(f"✅ Evaluation completed: Mean reward = {eval_results['mean_reward']:.2f}")

    # Save test
    print("\n\nTesting save/load...")
    baseline.save("/tmp/hca2c_baseline_test.pt")
    baseline.save_results("/tmp/hca2c_baseline_test")

    print("\n✅ All HCA2CBaseline tests passed!")
