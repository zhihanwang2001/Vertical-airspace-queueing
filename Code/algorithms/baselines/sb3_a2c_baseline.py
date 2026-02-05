"""
SB3 A2C Baseline Algorithm
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
import math
from env.drl_optimized_env_fixed import DRLOptimizedQueueEnvFixed
from .space_utils import SB3DictWrapper


class SB3A2CBaseline:
    """SB3 A2C Baseline Algorithm"""
    
    def __init__(self, config=None):
        # Configuration optimization v2 - Add delayed cosine annealing learning rate schedule to solve training instability
        default_config = {
            # Learning parameter optimization
            'initial_lr': 7e-4,             # Initial learning rate (maintained for first 300k steps)
            'min_lr': 1e-5,                 # Final learning rate (decreased to at 500k steps)
            'warmup_steps': 300000,         # First 300k steps maintain fixed lr for sufficient exploration
            'total_steps': 500000,          # Total training steps
            'n_steps': 32,                  # Optimization: 5 → 32 (increase rollout length to improve advantage estimation)
            'gamma': 0.99,                  # Keep unchanged
            'gae_lambda': 0.95,             # Optimization: 1.0 → 0.95 (bias-variance tradeoff)

            # Exploration and value function
            'ent_coef': 0.01,               # Optimization: 0.0 → 0.01 (add entropy regularization to promote exploration)
            'vf_coef': 0.5,                 # Keep unchanged
            'max_grad_norm': 0.5,           # Keep unchanged

            # Optimizer configuration
            'rms_prop_eps': 1e-5,           # Keep unchanged
            'use_rms_prop': True,           # Keep unchanged
            'use_sde': False,               # Keep unchanged
            'normalize_advantage': True,    # Optimization: False → True (normalize advantage to reduce variance)

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
        base_env = DRLOptimizedQueueEnvFixed(max_episode_steps=10000)
        wrapped_env = SB3DictWrapper(base_env)
        self.env = Monitor(wrapped_env, filename=None)
        
        # Create vectorized environment
        self.vec_env = DummyVecEnv([lambda: self.env])
        
        return self.env
    
    def create_model(self):
        """Create A2C model"""
        if self.env is None:
            self.setup_env()

        # Create delayed cosine annealing learning rate schedule (maintain fixed lr for first 300k steps, then cosine decay)
        def delayed_cosine_annealing_schedule(progress_remaining):
            """
            Delayed cosine annealing learning rate schedule
            First 300k steps: Maintain fixed learning rate 7e-4 (sufficient exploration)
            After 300k steps: Cosine annealing decay to 1e-5 (stable convergence)

            progress_remaining: 1.0 -> 0.0 (from start to end)
            """
            initial_lr = self.config.get('initial_lr', 7e-4)
            min_lr = self.config.get('min_lr', 1e-5)
            warmup_steps = self.config.get('warmup_steps', 300000)  # First 300k steps maintain fixed lr
            total_steps = self.config.get('total_steps', 500000)     # Total training steps

            # Calculate current step
            current_step = int((1.0 - progress_remaining) * total_steps)

            # First 300k steps: Maintain fixed learning rate
            if current_step < warmup_steps:
                return initial_lr

            # After 300k steps: Cosine annealing (map remaining 200k steps to 0→1)
            annealing_progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * annealing_progress))
            current_lr = min_lr + (initial_lr - min_lr) * cosine_factor

            return current_lr

        # Optimization: Increase network capacity to improve expressiveness
        policy_kwargs = dict(
            net_arch=dict(
                pi=[512, 512, 256],  # Policy network: Increase depth and width
                vf=[512, 512, 256]   # Value network: Independent large capacity network
            )
        )

        # Create A2C model
        self.model = A2C(
            "MlpPolicy",
            self.vec_env,
            learning_rate=delayed_cosine_annealing_schedule,  # v2: Use delayed cosine annealing schedule
            n_steps=self.config['n_steps'],
            gamma=self.config['gamma'],
            gae_lambda=self.config['gae_lambda'],
            ent_coef=self.config['ent_coef'],
            vf_coef=self.config['vf_coef'],
            max_grad_norm=self.config['max_grad_norm'],
            rms_prop_eps=self.config['rms_prop_eps'],
            use_rms_prop=self.config['use_rms_prop'],
            use_sde=self.config['use_sde'],
            normalize_advantage=self.config['normalize_advantage'],
            policy_kwargs=policy_kwargs,           # New: Network architecture configuration
            tensorboard_log=self.config['tensorboard_log'],
            verbose=self.config['verbose'],
            seed=self.config['seed'],
            device='auto'
        )
        
        print(f"SB3 A2C model created with device: {self.model.device}")
        return self.model
    
    def train(self, total_timesteps, eval_freq=10000, save_freq=50000):
        """Train model"""
        if self.model is None:
            self.create_model()
        
        # Create necessary directories
        os.makedirs('./logs/', exist_ok=True)
        os.makedirs('../../../Models/sb3_a2c_best/', exist_ok=True)
        os.makedirs('../../../Models/sb3_a2c_checkpoints/', exist_ok=True)
        
        # Create evaluation environment
        eval_env = DummyVecEnv([lambda: Monitor(
            SB3DictWrapper(DRLOptimizedQueueEnvFixed(max_episode_steps=10000)),
            filename=None
        )])
        
        # Create callbacks
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path='../../../Models/sb3_a2c_best/',
            log_path='./logs/',
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
            n_eval_episodes=10,
            verbose=1
        )
        
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path='../../../Models/sb3_a2c_checkpoints/',
            name_prefix='sb3_a2c'
        )
        
        # Start training
        print(f"Starting SB3 A2C training for {total_timesteps:,} timesteps...")
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=None,  # Remove problematic callbacks
            log_interval=10,
            tb_log_name="SB3_A2C"
        )
        
        print("SB3 A2C training completed!")
        
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
        
        print(f"SB3 A2C results saved to: {path_prefix}")
    
    def save(self, path):
        """Save model"""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        try:
            # Try using exclude parameter to avoid pickle errors
            # Exclude environment objects, only save model parameters
            self.model.save(path, exclude=['env', 'logger', 'ep_info_buffer', 'ep_success_buffer'])
            print(f"SB3 A2C model saved to: {path}")
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
            print(f"SB3 A2C model saved as state_dict to: {path}.pth")
    
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
            print(f"✅ SB3 A2C model loaded from state_dict: {pth_path}")
        else:
            # Standard SB3 format
            self.model = A2C.load(path, env=self.vec_env)
            print(f"SB3 A2C model loaded from: {path}")

        return self.model


def test_sb3_a2c():
    """Test SB3 A2C"""
    print("Testing SB3 A2C...")
    
    # Create baseline
    baseline = SB3A2CBaseline()
    
    # Train
    baseline.train(total_timesteps=50000)
    
    # Evaluate
    results = baseline.evaluate(n_episodes=10)
    print(f"SB3 A2C Results: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    
    # Save
    baseline.save("../../../Models/sb3_a2c_test.zip")


if __name__ == "__main__":
    test_sb3_a2c()
