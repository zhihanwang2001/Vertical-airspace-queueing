"""
Ablation Study Baseline Algorithm

Special PPO baseline created for ablation experiments, supporting:
1. Dynamic configuration modifications (high-layer priority, capacity structure, transfer mechanism, etc.)
2. Single-objective vs multi-objective reward function switching  
3. Component-level switch control
4. Fair comparison experimental setup with complete system

Based on sb3_ppo_baseline.py, with ablation experiment specific features added
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
from ablation_configs import AblationConfigs, AblationEnvironmentFactory


class AblationLearningRateLogger(BaseCallback):
    """Ablation experiment learning rate logger"""
    
    def __init__(self, initial_lr: float = 3e-4, min_lr: float = 1e-6, 
                 ablation_type: str = "full_system", verbose: int = 1):
        super().__init__(verbose)
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.ablation_type = ablation_type
        
    def _on_step(self) -> bool:
        """Log current learning rate and ablation experiment info to TensorBoard"""
        current_lr = self.model.policy.optimizer.param_groups[0]['lr']
        progress_remaining = getattr(self.model, '_current_progress_remaining', 1.0)
        progress = 1.0 - progress_remaining
        
        # Log basic learning rate info
        self.logger.record("train/learning_rate", current_lr)
        self.logger.record("train/lr_progress", progress)
        self.logger.record("train/lr_decay_ratio", current_lr / self.initial_lr)
        
        # Log ablation experiment type
        self.logger.record("ablation/experiment_type", self.ablation_type)
        
        # Periodic printing
        if self.num_timesteps % 10000 == 0 and self.verbose > 0:
            print(f"[{self.ablation_type}] Step {self.num_timesteps:6,}: LR={current_lr:.6f}")
        
        return True


def apply_ablation_config_to_env(env, ablation_config):
    """Apply ablation configuration directly to environment without using wrapper"""
    
    # 1. Modify arrival weights (no high-layer priority experiment)
    env.arrival_weights = np.array(ablation_config.arrival_weights, dtype=np.float32)
    
    # 2. Modify capacity configuration (traditional pyramid experiment)
    env.capacities = np.array(ablation_config.layer_capacities, dtype=np.int32)
    
    # 3. Modify service rates
    env.base_service_rates = np.array(ablation_config.layer_service_rates, dtype=np.float32)
    
    # 4. Handle reward function modification (single-objective experiment)
    if hasattr(ablation_config, '_reward_type') and ablation_config._reward_type == 'throughput_only':
        env._single_objective_mode = True
        
    # 5. Handle transfer mechanism (no transfer experiment)
    if hasattr(ablation_config, '_transfer_enabled') and not ablation_config._transfer_enabled:
        env._transfer_disabled = True
        
    ablation_type = getattr(ablation_config, '_ablation_type', 'unknown')
    print(f"Applied ablation modifications: {ablation_type}")
    if hasattr(ablation_config, '_removed_component'):
        print(f"   Removed component: {ablation_config._removed_component}")
    
    return env


class AblationEnvironmentWrapper:
    """Ablation experiment environment wrapper (deprecated, kept for compatibility)"""
    
    def __init__(self, base_env, ablation_config):
        self.base_env = base_env
        self.ablation_config = ablation_config
        self.ablation_type = getattr(ablation_config, '_ablation_type', 'full_system')
        
        # Apply ablation modifications
        self._apply_ablation_modifications()
    
    def _apply_ablation_modifications(self):
        """Apply ablation experiment modifications"""
        
        # 1. Modify arrival weights (no high-layer priority experiment)
        self.base_env.arrival_weights = np.array(self.ablation_config.arrival_weights, dtype=np.float32)
        
        # 2. Modify capacity configuration (traditional pyramid experiment)
        self.base_env.capacities = np.array(self.ablation_config.layer_capacities, dtype=np.int32)
        
        # 3. Modify service rates
        self.base_env.base_service_rates = np.array(self.ablation_config.layer_service_rates, dtype=np.float32)
        
        # 4. Handle reward function modification (single-objective experiment)
        if hasattr(self.ablation_config, '_reward_type') and self.ablation_config._reward_type == 'throughput_only':
            self.base_env._single_objective_mode = True
            
        # 5. Handle transfer mechanism (no transfer experiment)
        if hasattr(self.ablation_config, '_transfer_enabled') and not self.ablation_config._transfer_enabled:
            self.base_env._transfer_disabled = True
            
        print(f"Applied ablation modifications: {self.ablation_type}")
        if hasattr(self.ablation_config, '_removed_component'):
            print(f"   Removed component: {self.ablation_config._removed_component}")
    
    def __getattr__(self, name):
        """Proxy to base environment"""
        return getattr(self.base_env, name)


class SB3AblationBaseline:
    """Ablation experiment PPO baseline algorithm"""
    
    def __init__(self, ablation_type: str = "full_system", config=None):
        """
        Initialize ablation experiment baseline
        
        Args:
            ablation_type: Ablation experiment type
                - "full_system": Complete system (control group)
                - "no_high_priority": No high-layer priority
                - "single_objective": Single-objective optimization
                - "traditional_pyramid": Traditional pyramid
                - "no_transfer": No transfer mechanism
            config: Additional PPO configuration parameters
        """
        
        self.ablation_type = ablation_type
        
        # Get ablation configuration
        all_configs = AblationConfigs.get_all_ablation_configs()
        if ablation_type not in all_configs:
            raise ValueError(f"Unknown ablation type: {ablation_type}")
        
        self.ablation_config = all_configs[ablation_type]
        
        # PPO configuration
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
        
        print(f"Initializing ablation experiment: {ablation_type}")
        if hasattr(self.ablation_config, '_removed_component'):
            print(f"   Removed component: {self.ablation_config._removed_component}")
        
    def setup_env(self):
        """Setup ablation experiment environment"""
        base_env = DRLOptimizedQueueEnvFixed()
        
        # Apply ablation configuration directly
        apply_ablation_config_to_env(base_env, self.ablation_config)
        
        # Wrap environment
        wrapped_env = SB3DictWrapper(base_env)
        self.env = Monitor(wrapped_env, filename=None)
        
        # Create vectorized environment
        self.vec_env = DummyVecEnv([lambda: self.env])
        
        print(f"Ablation environment setup complete: {self.ablation_type}")
        return self.env
    
    def create_model(self):
        """Create ablation experiment PPO model"""
        if self.env is None:
            self.setup_env()
        
        # Cosine annealing learning rate schedule (consistent with complete system)
        def cosine_annealing_schedule(progress_remaining):
            initial_lr = self.config['learning_rate']
            min_lr = self.config.get('min_lr', 1e-6)
            progress = 1.0 - progress_remaining
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            current_lr = min_lr + (initial_lr - min_lr) * cosine_factor
            return current_lr
        
        # Create PPO model
        self.model = PPO(
            "MlpPolicy",
            self.vec_env,
            learning_rate=cosine_annealing_schedule,
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
        
        print(f"Ablation PPO model creation complete: {self.ablation_type}")
        return self.model
    
    def train(self, total_timesteps, eval_freq=10000, save_freq=50000):
        """Train ablation experiment model - simplified version to avoid pickle errors"""
        if self.model is None:
            self.create_model()
        
        # Start training
        print(f"Starting ablation experiment training: {self.ablation_type}")
        print(f"   Training steps: {total_timesteps:,}")
        if hasattr(self.ablation_config, '_removed_component'):
            print(f"   Removed component: {self.ablation_config._removed_component}")
        
        # Use simplified training without complex callbacks
        self.model.learn(
            total_timesteps=total_timesteps,
            log_interval=10
        )
        
        print(f"Ablation experiment training complete: {self.ablation_type}")
        
        return self.model
    
    def evaluate(self, n_episodes=10):
        """Evaluate ablation experiment model performance - simplified version"""
        if self.model is None:
            raise ValueError("Model not trained, please call train() first")
        
        print(f"Evaluating ablation experiment: {self.ablation_type}")
        
        episode_rewards = []
        
        for episode in range(n_episodes):
            obs = self.vec_env.reset()
            total_reward = 0
            done = False
            step_count = 0
            
            while not done and step_count < 1000:  # Limit maximum steps
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.vec_env.step(action)
                total_reward += reward[0]
                step_count += 1
            
            episode_rewards.append(total_reward)
            
            if episode % 5 == 0:
                print(f"   Episode {episode+1}/{n_episodes}: Reward={total_reward:.2f}")
        
        # Calculate statistics
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        
        results = {
            'ablation_type': self.ablation_type,
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'n_episodes': n_episodes,
            'removed_component': getattr(self.ablation_config, '_removed_component', 'None')
        }
        
        print(f"{self.ablation_type} evaluation results:")
        print(f"   Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
        
        return results


# Ablation experiment manager
class AblationExperimentManager:
    """Ablation experiment manager"""
    
    def __init__(self, total_timesteps=100000):
        self.total_timesteps = total_timesteps
        self.results = {}
        
    def run_all_ablation_experiments(self):
        """Run all ablation experiments"""
        ablation_types = [
            'full_system',
            'no_high_priority', 
            'single_objective',
            'traditional_pyramid',
            'no_transfer'
        ]
        
        print(f"Starting complete ablation study")
        print(f"   Number of experiments: {len(ablation_types)}")
        print(f"   Training steps per experiment: {self.total_timesteps:,}")
        print("=" * 60)
        
        for i, ablation_type in enumerate(ablation_types, 1):
            print(f"\nExecuting experiment {i}/{len(ablation_types)}: {ablation_type}")
            print("-" * 40)
            
            try:
                # Create and train model
                baseline = SB3AblationBaseline(ablation_type)
                baseline.train(self.total_timesteps)
                
                # Evaluate performance
                results = baseline.evaluate(n_episodes=20)
                self.results[ablation_type] = results
                
                print(f"{ablation_type} experiment complete")
                
            except Exception as e:
                print(f"{ablation_type} experiment failed: {str(e)}")
                self.results[ablation_type] = {'error': str(e)}
        
        print(f"\nAblation study complete!")
        self._print_comparison_results()
        
        return self.results
    
    def _print_comparison_results(self):
        """Print comparison results"""
        print(f"\nAblation experiment comparison results:")
        print("=" * 80)
        print(f"{'Experiment Type':<20} {'Mean Reward':<15} {'Std Dev':<10} {'Perf Drop':<10} {'Removed Component'}")
        print("-" * 80)
        
        full_system_reward = self.results.get('full_system', {}).get('mean_reward', 0)
        
        for ablation_type, result in self.results.items():
            if 'error' in result:
                print(f"{ablation_type:<20} {'ERROR':<15} {'-':<10} {'-':<10} {'-'}")
                continue
                
            mean_reward = result.get('mean_reward', 0)
            std_reward = result.get('std_reward', 0)
            removed_component = result.get('removed_component', 'None')
            
            if ablation_type == 'full_system':
                performance_drop = '0.0%'
            else:
                if full_system_reward > 0:
                    drop_percent = (full_system_reward - mean_reward) / full_system_reward * 100
                    performance_drop = f"{drop_percent:.1f}%"
                else:
                    performance_drop = 'N/A'
            
            print(f"{ablation_type:<20} {mean_reward:<15.2f} {std_reward:<10.2f} "
                  f"{performance_drop:<10} {removed_component}")
        
        print("-" * 80)
    
    def save_results(self, filepath="ablation_results.json"):
        """Save experiment results"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Experiment results saved to: {filepath}")


# Test and example usage
if __name__ == "__main__":
    print("Ablation experiment baseline test")
    
    # Test single ablation experiment
    print("\n1. Testing single ablation experiment...")
    baseline = SB3AblationBaseline("no_high_priority")
    
    # Quick training test
    print("   Quick training test...")
    baseline.train(total_timesteps=1000)  # Quick test
    
    # Evaluate
    print("   Evaluation test...")
    results = baseline.evaluate(n_episodes=3)
    
    print(f"Single ablation experiment test complete!")
    print(f"   Results: {results}")
