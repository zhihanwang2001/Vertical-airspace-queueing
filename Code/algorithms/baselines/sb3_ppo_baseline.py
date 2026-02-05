"""
SB3 PPO Baseline Algorithm
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import math
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
from env.drl_optimized_env_fixed import DRLOptimizedQueueEnvFixed
from .space_utils import SB3DictWrapper


class LearningRateLogger(BaseCallback):
    """
    Enhanced Learning Rate Logger for TensorBoard
    """
    
    def __init__(self, initial_lr: float = 3e-4, min_lr: float = 1e-6, verbose: int = 1):
        super().__init__(verbose)
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        
    def _on_step(self) -> bool:
        """Log current learning rate to TensorBoard"""
        # Get current learning rate
        current_lr = self.model.policy.optimizer.param_groups[0]['lr']
        progress_remaining = getattr(self.model, '_current_progress_remaining', 1.0)
        progress = 1.0 - progress_remaining
        
        # Don't save history to avoid pickle errors
        
        # Log to TensorBoard
        self.logger.record("train/learning_rate", current_lr)
        self.logger.record("train/lr_progress", progress)
        self.logger.record("train/lr_decay_ratio", current_lr / self.initial_lr)
        
        # Calculate theoretical learning rate (for verification)
        cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
        theoretical_lr = self.min_lr + (self.initial_lr - self.min_lr) * cosine_factor
        self.logger.record("train/theoretical_lr", theoretical_lr)
        self.logger.record("train/lr_error", abs(current_lr - theoretical_lr))
        
        # Periodically print learning rate
        if self.num_timesteps % 10000 == 0 and self.verbose > 0:
            print(f"Step {self.num_timesteps:6,}: LR={current_lr:.6f} (Progress: {progress:.1%}, Theoretical: {theoretical_lr:.6f})")
        
        return True


class SB3PPOBaseline:
    """SB3 PPO Baseline Algorithm"""
    
    def __init__(self, config=None):
        default_config = {
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'clip_range_vf': None,
            'normalize_advantage': True,
            'ent_coef': 0.0,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'target_kl': None,
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
        base_env = DRLOptimizedQueueEnvFixed(max_episode_steps=10000)
        wrapped_env = SB3DictWrapper(base_env)
        self.env = Monitor(wrapped_env, filename=None)
        
        # Create vectorized environment
        self.vec_env = DummyVecEnv([lambda: self.env])
        
        return self.env
    
    def create_model(self):
        """Create PPO model"""
        if self.env is None:
            self.setup_env()
        
        # Create learning rate schedule function
        def cosine_annealing_schedule(progress_remaining):
            """
            Cosine annealing learning rate schedule
            progress_remaining: 1.0 -> 0.0 (from start to end)
            """
            initial_lr = self.config['learning_rate']
            min_lr = self.config.get('min_lr', 1e-6)
            
            # Convert progress_remaining to progress (0.0 -> 1.0)
            progress = 1.0 - progress_remaining
            
            # Cosine annealing formula
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            current_lr = min_lr + (initial_lr - min_lr) * cosine_factor
            
            return current_lr
        
        # Create PPO model
        self.model = PPO(
            "MlpPolicy",
            self.vec_env,
            learning_rate=cosine_annealing_schedule,  # Use schedule function
            n_steps=self.config['n_steps'],
            batch_size=self.config['batch_size'],
            n_epochs=self.config['n_epochs'],
            gamma=self.config['gamma'],
            gae_lambda=self.config['gae_lambda'],
            clip_range=self.config['clip_range'],
            clip_range_vf=self.config['clip_range_vf'],
            normalize_advantage=self.config['normalize_advantage'],
            ent_coef=self.config['ent_coef'],
            vf_coef=self.config['vf_coef'],
            max_grad_norm=self.config['max_grad_norm'],
            target_kl=self.config['target_kl'],
            tensorboard_log=self.config['tensorboard_log'],
            verbose=self.config['verbose'],
            seed=self.config['seed'],
            device='auto'
        )
        
        print(f"SB3 PPO model created with device: {self.model.device}")
        return self.model
    
    def train(self, total_timesteps, eval_freq=10000, save_freq=50000):
        """Train model"""
        if self.model is None:
            self.create_model()
        
        # Create necessary directories
        os.makedirs('./logs/', exist_ok=True)
        os.makedirs('../../../Models/sb3_ppo_best/', exist_ok=True)
        os.makedirs('../../../Models/sb3_ppo_checkpoints/', exist_ok=True)
        
        # Create evaluation environment
        eval_env = DummyVecEnv([lambda: Monitor(
            SB3DictWrapper(DRLOptimizedQueueEnvFixed()), 
            filename=None
        )])
        
        # Create learning rate logger
        lr_logger = LearningRateLogger(
            initial_lr=self.config['learning_rate'],
            min_lr=self.config.get('min_lr', 1e-6),
            verbose=1
        )
        
        # Create evaluation callback
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path='../../../Models/sb3_ppo_best/',
            log_path='./logs/',
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
            n_eval_episodes=10,
            verbose=1
        )
        
        # Create checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path='../../../Models/sb3_ppo_checkpoints/',
            name_prefix='sb3_ppo'
        )
        
        # Start training
        print(f"Starting SB3 PPO training for {total_timesteps:,} timesteps...")
        print(f"Using Cosine Annealing LR: {self.config['learning_rate']} -> {self.config.get('min_lr', 1e-6)}")
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=lr_logger,  # Only use learning rate logger
            log_interval=10,
            tb_log_name="SB3_PPO_CosineAnneal"
        )
        
        print("SB3 PPO training completed!")
        
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
        eval_env = SB3DictWrapper(DRLOptimizedQueueEnvFixed(max_episode_steps=10000))

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

                if episode_length >= 10000:  # Prevent infinite loop
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
        
        print(f"SB3 PPO results saved to: {path_prefix}")
    
    def save(self, path):
        """Save model"""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        try:
            # Try using exclude parameter to avoid pickle errors
            # Exclude environment objects, only save model parameters
            self.model.save(path, exclude=['env', 'logger', 'ep_info_buffer', 'ep_success_buffer'])
            print(f"SB3 PPO model saved to: {path}")
        except Exception as e:
            # If still fails, save as PyTorch state dict
            print(f"Warning: Standard save failed ({e}), using state_dict fallback...")
            import torch
            state_dict = {
                'policy_state_dict': self.model.policy.state_dict(),
                'observation_space': self.model.observation_space,
                'action_space': self.model.action_space,
            }
            torch.save(state_dict, path + '.pth')
            print(f"SB3 PPO model saved as state_dict to: {path}.pth")
    
    def load(self, path):
        """Load model"""
        import os
        import torch

        if self.env is None:
            self.setup_env()

        # Check if it's a .pth file (fallback format)
        if path.endswith('.pth') or (not path.endswith('.zip') and os.path.exists(path + '.pth')):
            pth_path = path if path.endswith('.pth') else path + '.pth'
            print(f"Loading from state_dict format: {pth_path}")

            # Load state dict
            state_dict = torch.load(pth_path, weights_only=False)

            # Create new model
            self.create_model()

            # Load parameters
            self.model.policy.load_state_dict(state_dict['policy_state_dict'])
            print(f"✅ SB3 PPO model loaded from state_dict: {pth_path}")
        else:
            # Standard SB3 format
            self.model = PPO.load(path, env=self.vec_env)
            print(f"SB3 PPO model loaded from: {path}")

        return self.model


def test_sb3_ppo():
    """Test SB3 PPO"""
    print("Testing SB3 PPO...")
    
    # Create baseline
    baseline = SB3PPOBaseline()
    
    # Train
    baseline.train(total_timesteps=50000)
    
    # Evaluate
    results = baseline.evaluate(n_episodes=10)
    print(f"SB3 PPO Results: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    
    # Save
    baseline.save("../../../Models/sb3_ppo_test.zip")


if __name__ == "__main__":
    test_sb3_ppo()
