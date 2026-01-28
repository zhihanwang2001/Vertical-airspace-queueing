"""
SB3 DDPG Baseline Algorithm
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
from env.drl_optimized_env_fixed import DRLOptimizedQueueEnvFixed
from .space_utils import SB3DictWrapper


class SB3DDPGBaseline:
    """SB3 DDPG Baseline Algorithm"""
    
    def __init__(self, config=None):
        # Configuration optimization - Improve DDPG training stability
        default_config = {
            # Learning parameter optimization
            'learning_rate': 5e-5,          # Optimization: 1e-4 → 5e-5 (reduce 50% to prevent oscillation)
            'buffer_size': 500000,          # Optimization: 1M → 500k (reduce stale experience)
            'learning_starts': 10000,       # Optimization: 100 → 10000 (sufficient warm-up)
            'batch_size': 256,              # Keep unchanged
            'tau': 0.005,                   # Keep unchanged (soft update rate)
            'gamma': 0.99,                  # Keep unchanged
            'train_freq': 1,                # Keep unchanged
            'gradient_steps': 1,            # Keep unchanged

            # Exploration noise configuration
            'action_noise_type': 'ou',      # Optimization: normal → ou (OU noise is smoother)
            'action_noise_sigma': 0.15,     # Optimization: 0.1 → 0.15 (more initial exploration)

            # Other configurations
            'tensorboard_log': "./tensorboard_logs/",
            'verbose': 1,
            'seed': 42
        }
        
        if config:
            default_config.update(config)
        
        self.config = default_config
        self.model = None
        self.env = None
        
    def setup_env(self):
        """Setup environment"""
        base_env = DRLOptimizedQueueEnvFixed()
        wrapped_env = SB3DictWrapper(base_env)
        self.env = Monitor(wrapped_env, filename=None)
        
        # Create vectorized environment
        self.vec_env = DummyVecEnv([lambda: self.env])
        
        return self.env
    
    def create_model(self):
        """Create DDPG model"""
        if self.env is None:
            self.setup_env()
        
        # Setup action noise
        # Get action dimension from wrapped vectorized environment
        n_actions = self.vec_env.action_space.shape[-1]
        
        print(f"Action space dimension: {n_actions}")
        
        if self.config['action_noise_type'] == 'ou':
            # Ornstein-Uhlenbeck noise (better for continuous control)
            action_noise = OrnsteinUhlenbeckActionNoise(
                mean=np.zeros(n_actions),
                sigma=self.config['action_noise_sigma'] * np.ones(n_actions)
            )
        else:
            # Normal noise
            action_noise = NormalActionNoise(
                mean=np.zeros(n_actions), 
                sigma=self.config['action_noise_sigma'] * np.ones(n_actions)
            )
        
        # Optimization: Add policy network configuration and gradient clipping
        policy_kwargs = dict(
            net_arch=[512, 512, 256],  # Optimization: Increase network capacity to improve expressiveness
        )

        # Create DDPG model
        self.model = DDPG(
            "MlpPolicy",
            self.vec_env,
            learning_rate=self.config['learning_rate'],
            buffer_size=self.config['buffer_size'],
            learning_starts=self.config['learning_starts'],
            batch_size=self.config['batch_size'],
            tau=self.config['tau'],
            gamma=self.config['gamma'],
            train_freq=self.config['train_freq'],
            gradient_steps=self.config['gradient_steps'],
            action_noise=action_noise,
            policy_kwargs=policy_kwargs,           # New: Network architecture configuration
            tensorboard_log=self.config['tensorboard_log'],
            verbose=self.config['verbose'],
            seed=self.config['seed'],
            device='auto'
        )
        
        print(f"SB3 DDPG model created with device: {self.model.device}")
        print(f"Action noise type: {self.config['action_noise_type']}")
        return self.model
    
    def train(self, total_timesteps, eval_freq=10000, save_freq=50000):
        """Train model"""
        if self.model is None:
            self.create_model()
        
        # Create necessary directories
        os.makedirs('./logs/', exist_ok=True)
        os.makedirs('../../../Models/sb3_ddpg_best/', exist_ok=True)
        os.makedirs('../../../Models/sb3_ddpg_checkpoints/', exist_ok=True)
        
        # Create evaluation environment
        eval_env = DummyVecEnv([lambda: Monitor(
            SB3DictWrapper(DRLOptimizedQueueEnvFixed()), 
            filename=None
        )])
        
        # Create callbacks
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path='../../../Models/sb3_ddpg_best/',
            log_path='./logs/',
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
            n_eval_episodes=10,
            verbose=1
        )
        
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path='../../../Models/sb3_ddpg_checkpoints/',
            name_prefix='sb3_ddpg'
        )
        
        # Start training
        print(f"Starting SB3 DDPG training for {total_timesteps:,} timesteps...")
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=None,  # Remove problematic callbacks
            log_interval=10,
            tb_log_name="SB3_DDPG"
        )
        
        print("SB3 DDPG training completed!")
        
        # Return training results dictionary to be compatible with comparison framework
        return {
            'episodes': 0,  # SB3 doesn't have direct episode counting
            'total_timesteps': total_timesteps,
            'final_reward': 0  # Will be obtained in evaluation
        }
    
    def evaluate(self, n_episodes=10, deterministic=True, verbose=True):
        """Evaluate model"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Create evaluation environment
        eval_env = SB3DictWrapper(DRLOptimizedQueueEnvFixed())
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(n_episodes):
            obs, _ = eval_env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                
                if episode_length >= 200:  # Prevent infinite loop
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            if verbose:
                print(f"  Episode {episode+1}/{n_episodes}: Reward = {episode_reward:.2f}")
        
        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'system_metrics': []  # SB3 algorithm has no system metrics
        }
        
        return results
    
    def save_results(self, path_prefix):
        """Save training history and results"""
        # Create directory
        os.makedirs(os.path.dirname(path_prefix) if os.path.dirname(path_prefix) else ".", exist_ok=True)
        
        # SB3 algorithm has no training history, create empty history record
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'avg_rewards': [],
            'loss_values': []
        }
        
        # Save as JSON file (if needed)
        import json
        with open(f"{path_prefix}_history.json", 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        print(f"SB3 DDPG results saved to: {path_prefix}")
    
    def save(self, path):
        """Save model"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        self.model.save(path)
        print(f"SB3 DDPG model saved to: {path}")
    
    def load(self, path):
        """Load model"""
        if self.env is None:
            self.setup_env()
        
        self.model = DDPG.load(path, env=self.vec_env)
        print(f"SB3 DDPG model loaded from: {path}")
        return self.model


def test_sb3_ddpg():
    """Test SB3 DDPG"""
    print("Testing SB3 DDPG...")
    
    # Test different noise types
    configs = [
        {'action_noise_type': 'normal', 'action_noise_sigma': 0.1},
        {'action_noise_type': 'ou', 'action_noise_sigma': 0.1}
    ]
    
    for i, config in enumerate(configs):
        print(f"\n--- Testing config {i+1}: {config['action_noise_type']} noise ---")
        
        # Create baseline
        baseline = SB3DDPGBaseline(config)
        
        # Train
        baseline.train(total_timesteps=50000)
        
        # Evaluate
        results = baseline.evaluate(n_episodes=10)
        print(f"SB3 DDPG ({config['action_noise_type']}) Results: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        
        # Save
        baseline.save(f"../../../Models/sb3_ddpg_{config['action_noise_type']}_test.zip")


if __name__ == "__main__":
    test_sb3_ddpg()
