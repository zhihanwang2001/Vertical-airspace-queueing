"""
Prioritized Experience Replay for Rainbow DQN
实现基于TD-error的优先级经验回放
"""

import numpy as np
import torch
import random
from collections import namedtuple
from typing import List, Tuple, Optional


# Experience tuple
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class SumTree:
    """Sum Tree数据结构用于高效的优先级采样"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0
        self.size = 0
    
    def add(self, priority: float, data):
        """添加新经验"""
        tree_index = self.data_pointer + self.capacity - 1
        
        self.data[self.data_pointer] = data
        self.update(tree_index, priority)
        
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0
        
        if self.size < self.capacity:
            self.size += 1
    
    def update(self, tree_index: int, priority: float):
        """更新优先级"""
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        
        # 向上传播变化
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change
    
    def get_leaf(self, value: float) -> Tuple[int, float, object]:
        """根据累积概率获取叶节点"""
        parent_index = 0
        
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:
                if value <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    value -= self.tree[left_child_index]
                    parent_index = right_child_index
        
        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], self.data[data_index]
    
    @property
    def total_priority(self) -> float:
        """总优先级"""
        return self.tree[0]


class PrioritizedReplayBuffer:
    """优先级经验回放缓冲区"""
    
    def __init__(self, 
                 capacity: int, 
                 alpha: float = 0.6, 
                 beta: float = 0.4, 
                 beta_increment: float = 0.001,
                 epsilon: float = 1e-6):
        """
        Args:
            capacity: 缓冲区容量
            alpha: 优先级指数 (0=均匀采样, 1=完全优先级采样)
            beta: 重要性采样指数 (0=无校正, 1=完全校正)
            beta_increment: beta的增长率
            epsilon: 最小优先级，防止经验永远不被采样
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        
        self.tree = SumTree(capacity)
        self.max_priority = 1.0
        
    def add(self, state, action, reward, next_state, done):
        """添加经验"""
        experience = Experience(state, action, reward, next_state, done)
        
        # 新经验使用最大优先级
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, experience)
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, List[int]]:
        """采样一批经验"""
        batch = []
        indices = []
        priorities = []
        
        # 计算采样区间
        priority_segment = self.tree.total_priority / batch_size
        
        for i in range(batch_size):
            # 在每个区间内随机采样
            left = priority_segment * i
            right = priority_segment * (i + 1)
            value = random.uniform(left, right)
            
            index, priority, experience = self.tree.get_leaf(value)
            
            batch.append(experience)
            indices.append(index)
            priorities.append(priority)
        
        # 计算重要性权重
        sampling_probabilities = np.array(priorities) / self.tree.total_priority
        weights = np.power(self.tree.size * sampling_probabilities, -self.beta)
        weights = weights / weights.max()  # 归一化
        
        return batch, weights, indices
    
    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """更新经验的优先级"""
        for index, priority in zip(indices, priorities):
            # 确保优先级非负且加上epsilon
            priority = abs(priority) + self.epsilon
            self.tree.update(index, priority ** self.alpha)
            self.max_priority = max(self.max_priority, priority)
    
    def update_beta(self):
        """更新beta值"""
        self.beta = min(1.0, self.beta + self.beta_increment)
    
    def __len__(self) -> int:
        return self.tree.size
    
    @property
    def is_ready(self) -> bool:
        """是否有足够的经验开始训练"""
        return len(self) > 1000  # 至少1000个经验


class UniformReplayBuffer:
    """标准均匀经验回放（用于对比）"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def add(self, state, action, reward, next_state, done):
        """添加经验"""
        experience = Experience(state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, List[int]]:
        """均匀采样"""
        batch = random.sample(self.buffer, batch_size)
        weights = np.ones(batch_size)  # 均匀权重
        indices = list(range(batch_size))  # 虚拟索引
        return batch, weights, indices
    
    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """空实现，保持接口一致"""
        pass
    
    def update_beta(self):
        """空实现"""
        pass
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    @property
    def is_ready(self) -> bool:
        return len(self) > 1000


def create_replay_buffer(buffer_type: str = "prioritized", **kwargs):
    """工厂函数：创建经验回放缓冲区"""
    if buffer_type == "prioritized":
        return PrioritizedReplayBuffer(**kwargs)
    elif buffer_type == "uniform":
        return UniformReplayBuffer(kwargs.get('capacity', 100000))
    else:
        raise ValueError(f"Unknown buffer type: {buffer_type}")


def batch_to_tensors(batch: List[Experience], device: torch.device):
    """将batch转换为tensor"""
    states = torch.FloatTensor([e.state for e in batch]).to(device)
    actions = torch.LongTensor([e.action for e in batch]).to(device)
    rewards = torch.FloatTensor([e.reward for e in batch]).to(device)
    next_states = torch.FloatTensor([e.next_state for e in batch]).to(device)
    dones = torch.BoolTensor([e.done for e in batch]).to(device)
    
    return states, actions, rewards, next_states, dones