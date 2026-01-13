"""
Random Baseline
随机基线算法 - 用作对比基准
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import time
import random

from .base_baseline import BaseBaseline


class RandomBaseline(BaseBaseline):
    """随机基线算法实现"""
    
    def __init__(self, 
                 env,
                 algorithm_name: str = "Random",
                 config: Optional[Dict] = None):
        
        default_config = {
            'seed': None,  # 随机种子
            'action_bounds': {
                'service_intensities': (0.1, 2.0),
                'arrival_multiplier': (0.5, 5.0),
                'emergency_transfers': (0, 1)
            }
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(env, algorithm_name, default_config)
        
        # 设置随机种子
        if self.config['seed'] is not None:
            random.seed(self.config['seed'])
            np.random.seed(self.config['seed'])
        
        print(f"Random Baseline initialized")
    
    def predict(self, observation, deterministic: bool = True) -> Tuple[np.ndarray, Optional[Dict]]:
        """随机预测动作"""
        # 忽略deterministic参数，总是随机选择
        
        # 根据环境的action_space生成随机动作
        if hasattr(self.env.action_space, 'sample'):
            action = self.env.action_space.sample()
        else:
            # 手动生成随机动作
            action = self._generate_random_action()
        
        return action, None
    
    def _generate_random_action(self) -> Dict:
        """生成随机动作"""
        bounds = self.config['action_bounds']
        
        action = {}
        
        # service_intensities: 5维连续动作
        service_min, service_max = bounds['service_intensities']
        action['service_intensities'] = np.random.uniform(
            service_min, service_max, size=5
        )
        
        # arrival_multiplier: 1维连续动作
        arrival_min, arrival_max = bounds['arrival_multiplier']
        action['arrival_multiplier'] = np.random.uniform(
            arrival_min, arrival_max
        )
        
        # emergency_transfers: 5维二进制动作
        action['emergency_transfers'] = np.random.randint(
            0, 2, size=5
        )
        
        return action
    
    def train(self, total_timesteps: int, **kwargs) -> Dict:
        """训练过程（实际上只是随机运行）"""
        print(f"Running Random Baseline for {total_timesteps} timesteps...")
        
        # 重置训练记录
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'avg_rewards': [],
            'training_time': [],
            'loss_values': []
        }
        
        start_time = time.time()
        
        state, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        episode_count = 0
        
        for timestep in range(total_timesteps):
            # 随机选择动作
            action, _ = self.predict(state, deterministic=False)
            
            # 执行动作
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            if done:
                # 记录episode信息
                self.training_history['episode_rewards'].append(episode_reward)
                self.training_history['episode_lengths'].append(episode_length)
                
                # 计算平均奖励
                if len(self.training_history['episode_rewards']) >= 100:
                    avg_reward = np.mean(self.training_history['episode_rewards'][-100:])
                    self.training_history['avg_rewards'].append(avg_reward)
                
                if episode_count % 100 == 0:
                    print(f"Episode {episode_count}, Timestep {timestep}, Reward: {episode_reward:.2f}")
                
                # 重置环境
                state, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0
                episode_count += 1
        
        end_time = time.time()
        training_time = end_time - start_time
        self.training_history['training_time'].append(training_time)
        
        self.total_timesteps = total_timesteps
        self.episode_count = episode_count
        
        print(f"Random Baseline completed in {training_time:.2f} seconds")
        
        return {
            'total_timesteps': total_timesteps,
            'episodes': episode_count,
            'training_time': training_time,
            'final_reward': self.training_history['episode_rewards'][-1] if self.training_history['episode_rewards'] else 0
        }
    
    def save(self, path: str) -> None:
        """保存模型（随机基线没有参数需要保存）"""
        import json
        
        save_data = {
            'algorithm_name': self.algorithm_name,
            'config': self.config,
            'total_timesteps': self.total_timesteps,
            'episode_count': self.episode_count,
            'training_history': self.training_history
        }
        
        with open(path, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"Random Baseline saved to: {path}")
    
    def load(self, path: str) -> None:
        """加载模型"""
        import json
        
        with open(path, 'r') as f:
            save_data = json.load(f)
        
        self.config.update(save_data['config'])
        self.total_timesteps = save_data.get('total_timesteps', 0)
        self.episode_count = save_data.get('episode_count', 0)
        self.training_history = save_data.get('training_history', {
            'episode_rewards': [],
            'episode_lengths': [],
            'avg_rewards': [],
            'training_time': [],
            'loss_values': []
        })
        
        print(f"Random Baseline loaded from: {path}")
    
    def get_info(self) -> Dict:
        """获取算法信息"""
        info = super().get_info()
        info.update({
            'description': 'Random policy baseline for comparison',
            'deterministic': False,
            'requires_training': False
        })
        return info