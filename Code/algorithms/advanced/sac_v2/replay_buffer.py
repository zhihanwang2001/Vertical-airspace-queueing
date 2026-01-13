"""
SAC v2 Replay Buffer
SAC v2算法的经验回放缓冲区
"""

import numpy as np
import torch
import random
from typing import Dict, Tuple, Any, Optional
from collections import deque


class SAC_ReplayBuffer:
    """SAC经验回放缓冲区"""
    
    def __init__(self,
                 capacity: int = 100000,
                 batch_size: int = 256,
                 device: torch.device = torch.device('cpu')):
        """
        初始化回放缓冲区
        
        Args:
            capacity: 缓冲区容量
            batch_size: 批次大小
            device: 计算设备
        """
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        
        # 使用deque存储经验
        self.buffer = deque(maxlen=capacity)
        
        # 统计信息
        self.total_samples = 0
        
        print(f"📦 SAC Replay Buffer initialized")
        print(f"   Capacity: {capacity:,}")
        print(f"   Batch size: {batch_size}")
    
    def add(self,
            state: np.ndarray,
            action: np.ndarray, 
            reward: float,
            next_state: np.ndarray,
            done: bool):
        """
        添加一个经验到缓冲区
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一状态
            done: 是否结束
        """
        experience = {
            'state': np.array(state, dtype=np.float32),
            'action': np.array(action, dtype=np.float32),
            'reward': float(reward),
            'next_state': np.array(next_state, dtype=np.float32),
            'done': bool(done)
        }
        
        self.buffer.append(experience)
        self.total_samples += 1
    
    def sample(self, batch_size: Optional[int] = None) -> Optional[Dict[str, torch.Tensor]]:
        """
        从缓冲区采样一个批次
        
        Args:
            batch_size: 批次大小，如果为None则使用默认大小
            
        Returns:
            批次字典，如果样本不足返回None
        """
        if batch_size is None:
            batch_size = self.batch_size
        
        if len(self.buffer) < batch_size:
            return None
        
        # 随机采样
        batch = random.sample(self.buffer, batch_size)
        
        # 分离数据
        states = np.array([exp['state'] for exp in batch])
        actions = np.array([exp['action'] for exp in batch])
        rewards = np.array([exp['reward'] for exp in batch])
        next_states = np.array([exp['next_state'] for exp in batch])
        dones = np.array([exp['done'] for exp in batch])
        
        # 转换为张量
        batch_tensors = {
            'states': torch.FloatTensor(states).to(self.device),
            'actions': torch.FloatTensor(actions).to(self.device),
            'rewards': torch.FloatTensor(rewards).unsqueeze(1).to(self.device),
            'next_states': torch.FloatTensor(next_states).to(self.device),
            'dones': torch.FloatTensor(dones.astype(np.float32)).unsqueeze(1).to(self.device)
        }
        
        return batch_tensors
    
    def __len__(self):
        """返回缓冲区当前大小"""
        return len(self.buffer)
    
    @property
    def is_ready(self):
        """检查是否有足够样本进行训练"""
        return len(self.buffer) >= self.batch_size
    
    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()
        self.total_samples = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓冲区统计信息"""
        if len(self.buffer) == 0:
            return {
                'buffer_size': 0,
                'total_samples': self.total_samples,
                'fill_percentage': 0.0,
                'avg_reward': 0.0,
                'avg_episode_length': 0.0
            }
        
        # 计算统计信息
        rewards = [exp['reward'] for exp in self.buffer]
        
        return {
            'buffer_size': len(self.buffer),
            'total_samples': self.total_samples,
            'fill_percentage': len(self.buffer) / self.capacity * 100,
            'avg_reward': np.mean(rewards),
            'reward_std': np.std(rewards),
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards)
        }