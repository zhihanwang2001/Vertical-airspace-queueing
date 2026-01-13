"""
基线算法基础类
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
    所有基线算法的基础类
    """
    
    def __init__(self, 
                 env,
                 algorithm_name: str,
                 config: Optional[Dict] = None):
        """
        初始化基线算法
        
        Args:
            env: 环境实例
            algorithm_name: 算法名称
            config: 算法配置参数
        """
        self.env = env
        self.algorithm_name = algorithm_name
        self.config = config or {}
        
        # 训练记录
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'avg_rewards': [],
            'training_time': [],
            'loss_values': []
        }
        
        # 评估记录
        self.evaluation_history = {
            'eval_rewards': [],
            'eval_std': [],
            'eval_episodes': [],
            'system_metrics': []
        }
        
        # 算法特定参数
        self.total_timesteps = 0
        self.episode_count = 0
        
    @abstractmethod
    def train(self, total_timesteps: int, **kwargs) -> Dict:
        """
        训练算法
        
        Args:
            total_timesteps: 总训练步数
            **kwargs: 其他训练参数
            
        Returns:
            训练结果字典
        """
        pass
    
    @abstractmethod
    def predict(self, observation, deterministic: bool = True) -> Tuple[np.ndarray, Optional[Dict]]:
        """
        预测动作
        
        Args:
            observation: 观察值
            deterministic: 是否确定性预测
            
        Returns:
            (action, extra_info)
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """
        加载模型
        
        Args:
            path: 模型路径
        """
        pass
    
    def evaluate(self, 
                 n_episodes: int = 10,
                 deterministic: bool = True,
                 verbose: bool = True) -> Dict:
        """
        评估算法性能
        
        Args:
            n_episodes: 评估轮数
            deterministic: 是否确定性评估
            verbose: 是否输出详细信息
            
        Returns:
            评估结果字典
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
            
            # 收集系统指标
            if hasattr(info, 'keys') and 'throughput' in info:
                system_metrics.append({
                    'throughput': info.get('throughput', 0),
                    'utilization_rates': info.get('utilization_rates', []),
                    'stability_score': info.get('stability_score', 0)
                })
            
            if verbose and (ep + 1) % 5 == 0:
                print(f"  Episode {ep + 1}/{n_episodes}: Reward = {episode_reward:.2f}")
        
        # 计算统计量
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
        绘制训练历史
        
        Args:
            save_path: 保存路径
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
        保存训练和评估结果
        
        Args:
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存训练历史
        training_file = os.path.join(save_dir, f'{self.algorithm_name}_training_history.json')
        with open(training_file, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # 保存评估历史
        eval_file = os.path.join(save_dir, f'{self.algorithm_name}_evaluation_history.json')
        with open(eval_file, 'w') as f:
            json.dump(self.evaluation_history, f, indent=2)
        
        # 保存配置
        config_file = os.path.join(save_dir, f'{self.algorithm_name}_config.json')
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print(f"Results saved to: {save_dir}")
    
    def get_info(self) -> Dict:
        """
        获取算法信息
        
        Returns:
            算法信息字典
        """
        return {
            'algorithm_name': self.algorithm_name,
            'total_timesteps': self.total_timesteps,
            'episode_count': self.episode_count,
            'config': self.config,
            'training_completed': len(self.training_history['episode_rewards']) > 0
        }