"""
Base class for baseline algorithms
Base class for all baseline algorithms
"""

from abc import ABC, abstractmethod
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import json
import os


class BaseBaseline(ABC):
    """
    Base class for all baseline algorithms
    """
    
    def __init__(self, 
                 env,
                 algorithm_name: str,
                 config: Optional[Dict] = None):
        """
        Initialize baseline algorithm
        
        Args:
            env: Environment instance
            algorithm_name: Algorithm name
            config: Algorithm configuration parameters
        """
        self.env = env
        self.algorithm_name = algorithm_name
        self.config = config or {}
        
        # Training records
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'avg_rewards': [],
            'training_time': [],
            'loss_values': []
        }
        
        # Evaluation records
        self.evaluation_history = {
            'eval_rewards': [],
            'eval_std': [],
            'eval_episodes': [],
            'system_metrics': []
        }
        
        # Algorithm-specific parameters
        self.total_timesteps = 0
        self.episode_count = 0
        
    @abstractmethod
    def train(self, total_timesteps: int, **kwargs) -> Dict:
        """
        Train algorithm
        
        Args:
            total_timesteps: Total training timesteps
            **kwargs: Other training parameters
            
        Returns:
            Training results dictionary
        """
        pass
    
    @abstractmethod
    def predict(self, observation, deterministic: bool = True) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        Predict action
        
        Args:
            observation: Observation
            deterministic: Whether to use deterministic prediction
            
        Returns:
            (action, extra_info)
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save model
        
        Args:
            path: Save path
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load model
        
        Args:
            path: Model path
        """
        pass
    
    def evaluate(self, 
                 n_episodes: int = 10,
                 deterministic: bool = True,
                 verbose: bool = True) -> Dict:
        """
        Evaluate algorithm performance
        
        Args:
            n_episodes: Number of evaluation episodes
            deterministic: Whether to use deterministic evaluation
            verbose: Whether to print detailed information
            
        Returns:
            Evaluation results dictionary
        """
        episode_rewards = []
        episode_lengths = []
        system_metrics = []
        
        for ep in range(n_episodes):
            obs, info = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                action, _ = self.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Collect system metrics
            if hasattr(info, 'keys') and 'throughput' in info:
                system_metrics.append({
                    'throughput': info.get('throughput', 0),
                    'utilization_rates': info.get('utilization_rates', []),
                    'stability_score': info.get('stability_score', 0)
                })
            
            if verbose and (ep + 1) % 5 == 0:
                print(f"  Episode {ep + 1}/{n_episodes}: Reward = {episode_reward:.2f}")
        
        # Calculate statistics
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_length = np.mean(episode_lengths)
        
        eval_results = {
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'mean_length': mean_length,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'system_metrics': system_metrics
        }
        
        if verbose:
            print(f"Evaluation Results ({self.algorithm_name}):")
            print(f"  Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
            print(f"  Mean Length: {mean_length:.1f}")
        
        return eval_results
    
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """
        Plot training history
        
        Args:
            save_path: Save path
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{self.algorithm_name} Training History', fontsize=16)
        
        # Episode Rewards
        if self.training_history['episode_rewards']:
            axes[0, 0].plot(self.training_history['episode_rewards'])
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].grid(True)
        
        # Average Rewards
        if self.training_history['avg_rewards']:
            axes[0, 1].plot(self.training_history['avg_rewards'])
            axes[0, 1].set_title('Average Rewards (100 episodes)')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Average Reward')
            axes[0, 1].grid(True)
        
        # Episode Lengths
        if self.training_history['episode_lengths']:
            axes[1, 0].plot(self.training_history['episode_lengths'])
            axes[1, 0].set_title('Episode Lengths')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Length')
            axes[1, 0].grid(True)
        
        # Loss Values
        if self.training_history['loss_values']:
            axes[1, 1].plot(self.training_history['loss_values'])
            axes[1, 1].set_title('Loss Values')
            axes[1, 1].set_xlabel('Training Step')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to: {save_path}")
        
        plt.show()
    
    def save_results(self, save_dir: str) -> None:
        """
        Save training and evaluation results
        
        Args:
            save_dir: Save directory
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save training history
        training_file = os.path.join(save_dir, f'{self.algorithm_name}_training_history.json')
        with open(training_file, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Save evaluation history
        eval_file = os.path.join(save_dir, f'{self.algorithm_name}_evaluation_history.json')
        with open(eval_file, 'w') as f:
            json.dump(self.evaluation_history, f, indent=2)
        
        # Save configuration
        config_file = os.path.join(save_dir, f'{self.algorithm_name}_config.json')
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"Results saved to: {save_dir}")
    
    def get_info(self) -> Dict:
        """
        Get algorithm information
        
        Returns:
            Algorithm information dictionary
        """
        return {
            'algorithm_name': self.algorithm_name,
            'total_timesteps': self.total_timesteps,
            'episode_count': self.episode_count,
            'config': self.config,
            'training_completed': len(self.training_history['episode_rewards']) > 0
        }
