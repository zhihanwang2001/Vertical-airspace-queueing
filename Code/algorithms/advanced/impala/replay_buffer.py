"""
IMPALA Replay Buffer
IMPALA算法的经验回放缓冲区，支持序列存储和批量采样
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Any, Optional
from collections import deque
import random


class Episode:
    """单个Episode的存储"""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.next_states = []
    
    def add_step(self, state, action, reward, next_state, done, log_prob, value):
        """添加一个时间步的数据"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def to_tensors(self, device: torch.device):
        """转换为张量格式"""
        return {
            'states': torch.FloatTensor(np.array(self.states)).to(device),
            'actions': torch.FloatTensor(np.array(self.actions)).to(device),
            'rewards': torch.FloatTensor(np.array(self.rewards)).to(device),
            'next_states': torch.FloatTensor(np.array(self.next_states)).to(device),
            'dones': torch.FloatTensor(np.array(self.dones, dtype=np.float32)).to(device),
            'log_probs': torch.FloatTensor(np.array(self.log_probs)).to(device),
            'values': torch.FloatTensor(np.array(self.values)).to(device)
        }
    
    def __len__(self):
        return len(self.states)


class IMPALAReplayBuffer:
    """IMPALA经验回放缓冲区"""
    
    def __init__(self, 
                 capacity: int = 10000,
                 sequence_length: int = 20,
                 device: torch.device = torch.device('cpu')):
        """
        初始化回放缓冲区
        
        Args:
            capacity: 最大episode数量
            sequence_length: 序列长度（用于截断长episode）
            device: 计算设备
        """
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.device = device
        
        self.episodes = deque(maxlen=capacity)
        self.current_episode = Episode()
        
    def add_step(self, state, action, reward, next_state, done, log_prob, value):
        """添加一个时间步"""
        self.current_episode.add_step(state, action, reward, next_state, done, log_prob, value)
        
        # 如果episode结束，保存并开始新episode
        if done:
            if len(self.current_episode) > 0:
                self.episodes.append(self.current_episode)
            self.current_episode = Episode()
    
    def sample_sequences(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        采样序列批次
        
        Args:
            batch_size: 批次大小
            
        Returns:
            包含序列数据的字典
        """
        if len(self.episodes) < batch_size:
            return None
        
        # 随机选择episodes
        selected_episodes = random.sample(list(self.episodes), batch_size)
        
        # 将episodes转换为固定长度的序列
        batch_sequences = []
        
        for episode in selected_episodes:
            episode_data = episode.to_tensors(self.device)
            episode_length = len(episode)
            
            if episode_length >= self.sequence_length:
                # Episode足够长，随机选择一个起始位置
                start_idx = random.randint(0, episode_length - self.sequence_length)
                sequence = {
                    key: value[start_idx:start_idx + self.sequence_length]
                    for key, value in episode_data.items()
                }
            else:
                # Episode不够长，进行padding
                sequence = {}
                for key, value in episode_data.items():
                    if key in ['dones', 'rewards']:
                        # 对于dones和rewards，用0填充
                        padding = torch.zeros(self.sequence_length - episode_length).to(self.device)
                    else:
                        # 对于其他张量，用最后一个值填充
                        last_value = value[-1] if len(value) > 0 else torch.zeros_like(value[0])
                        padding = last_value.unsqueeze(0).repeat(self.sequence_length - episode_length, *([1] * (len(last_value.shape))))
                    
                    sequence[key] = torch.cat([value, padding], dim=0)
            
            batch_sequences.append(sequence)
        
        # 将所有序列堆叠成批次 [batch_size, sequence_length, ...]
        batch = {}
        for key in batch_sequences[0].keys():
            batch[key] = torch.stack([seq[key] for seq in batch_sequences], dim=0)
        
        # 重新排列维度为 [sequence_length, batch_size, ...]
        for key, value in batch.items():
            batch[key] = value.transpose(0, 1)
        
        return batch
    
    def get_recent_trajectories(self, num_episodes: int = None) -> List[Dict[str, torch.Tensor]]:
        """
        获取最近的轨迹
        
        Args:
            num_episodes: 获取的episode数量，None表示获取所有
            
        Returns:
            轨迹列表
        """
        if num_episodes is None:
            episodes_to_return = list(self.episodes)
        else:
            episodes_to_return = list(self.episodes)[-num_episodes:]
        
        trajectories = []
        for episode in episodes_to_return:
            if len(episode) > 0:
                trajectories.append(episode.to_tensors(self.device))
        
        return trajectories
    
    def clear(self):
        """清空缓冲区"""
        self.episodes.clear()
        self.current_episode = Episode()
    
    def __len__(self):
        return len(self.episodes)
    
    @property
    def size(self):
        """缓冲区中的总步数"""
        return sum(len(episode) for episode in self.episodes)
    
    @property
    def is_ready(self):
        """是否有足够的数据进行采样"""
        return len(self.episodes) > 10  # 至少需要10个episode


def collate_sequences(sequences: List[Dict[str, torch.Tensor]], 
                     max_length: int = None) -> Dict[str, torch.Tensor]:
    """
    将不等长的序列打包成批次
    
    Args:
        sequences: 序列列表
        max_length: 最大序列长度
        
    Returns:
        批次字典
    """
    if not sequences:
        return {}
    
    # 确定最大长度
    if max_length is None:
        max_length = max(seq['states'].shape[0] for seq in sequences)
    
    batch_size = len(sequences)
    padded_batch = {}
    
    # 初始化批次张量
    for key in sequences[0].keys():
        example_tensor = sequences[0][key]
        if len(example_tensor.shape) == 1:
            # 一维张量（如rewards, dones）
            padded_batch[key] = torch.zeros(max_length, batch_size, device=example_tensor.device)
        else:
            # 多维张量（如states, actions）
            shape = (max_length, batch_size) + example_tensor.shape[1:]
            padded_batch[key] = torch.zeros(shape, device=example_tensor.device)
    
    # 填充数据
    for i, seq in enumerate(sequences):
        seq_len = min(seq['states'].shape[0], max_length)
        for key, tensor in seq.items():
            if len(tensor.shape) == 1:
                padded_batch[key][:seq_len, i] = tensor[:seq_len]
            else:
                padded_batch[key][:seq_len, i] = tensor[:seq_len]
    
    return padded_batch