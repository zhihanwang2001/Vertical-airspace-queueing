"""
TD7 Replay Buffer with Prioritized Experience Replay
TD7算法的优先级经验回放缓冲区和LAP机制
"""

import numpy as np
import torch
import random
from typing import Dict, Tuple, Any, Optional, List
from collections import deque
import heapq


class SumTree:
    """SumTree数据结构，用于优先级经验回放"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.pending_idx = set()
    
    def _propagate(self, idx: int, change: float):
        """向上传播优先级变化"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, s: float) -> int:
        """检索叶节点索引"""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self) -> float:
        """返回所有优先级总和"""
        return self.tree[0]
    
    def add(self, p: float, data):
        """添加经验"""
        idx = self.n_entries % self.capacity
        data_idx = idx + self.capacity - 1
        
        self.data[idx] = data
        self.update(data_idx, p)
        
        self.n_entries += 1
    
    def update(self, idx: int, p: float):
        """更新优先级"""
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)
    
    def get(self, s: float) -> Tuple[int, float, Any]:
        """根据优先级采样"""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class TD7_PrioritizedReplayBuffer:
    """TD7优先级经验回放缓冲区，包含LAP机制"""
    
    def __init__(self,
                 capacity: int = 1000000,
                 batch_size: int = 256,
                 alpha: float = 0.6,  # 优先级指数
                 beta: float = 0.4,   # 重要性采样指数
                 beta_increment: float = 0.001,
                 device: torch.device = torch.device('cpu')):
        """
        初始化优先级回放缓冲区
        
        Args:
            capacity: 缓冲区容量
            batch_size: 批次大小
            alpha: 优先级指数
            beta: 重要性采样指数
            beta_increment: beta增长率
            device: 计算设备
        """
        self.capacity = capacity
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.device = device
        
        self.tree = SumTree(capacity)
        self.epsilon = 0.01  # 最小优先级
        self.max_priority = 1.0
        
        # LAP (Learned Action Prioritization) 组件
        self.use_lap = True
        self.lap_weight = 0.1
        
        print(f"📦 TD7 Prioritized Replay Buffer initialized")
        print(f"   Capacity: {capacity:,}")
        print(f"   Batch size: {batch_size}")
        print(f"   Alpha: {alpha}, Beta: {beta}")
        print(f"   LAP enabled: {self.use_lap}")
    
    def add(self,
            state: np.ndarray,
            action: np.ndarray,
            reward: float,
            next_state: np.ndarray,
            done: bool):
        """
        添加经验到缓冲区
        
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
        
        # 新经验使用最大优先级
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, experience)
    
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
        
        if self.tree.n_entries < batch_size:
            return None
        
        batch_idx = []
        batch_experiences = []
        priorities = []
        
        # 分段采样
        segment = self.tree.total() / batch_size
        
        # 更新beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            
            idx, priority, experience = self.tree.get(s)
            
            batch_idx.append(idx)
            batch_experiences.append(experience)
            priorities.append(priority)
        
        # 计算重要性采样权重
        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()
        
        # 分离数据
        states = np.array([exp['state'] for exp in batch_experiences])
        actions = np.array([exp['action'] for exp in batch_experiences])
        rewards = np.array([exp['reward'] for exp in batch_experiences])
        next_states = np.array([exp['next_state'] for exp in batch_experiences])
        dones = np.array([exp['done'] for exp in batch_experiences])
        
        # 转换为张量
        batch_tensors = {
            'states': torch.FloatTensor(states).to(self.device),
            'actions': torch.FloatTensor(actions).to(self.device),
            'rewards': torch.FloatTensor(rewards).unsqueeze(1).to(self.device),
            'next_states': torch.FloatTensor(next_states).to(self.device),
            'dones': torch.FloatTensor(dones.astype(np.float32)).unsqueeze(1).to(self.device),
            'is_weights': torch.FloatTensor(is_weights).unsqueeze(1).to(self.device),
            'indices': batch_idx
        }
        
        return batch_tensors
    
    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """
        更新优先级
        
        Args:
            indices: 样本索引
            priorities: 新优先级
        """
        for idx, priority in zip(indices, priorities):
            # 确保优先级为正
            priority = abs(priority) + self.epsilon
            
            # LAP加权
            if self.use_lap:
                priority = priority * (1 + self.lap_weight)
            
            # 更新最大优先级
            self.max_priority = max(self.max_priority, priority)
            
            # 应用alpha指数
            self.tree.update(idx, priority ** self.alpha)
    
    def compute_lap_priority(self, 
                           td_error: torch.Tensor, 
                           action: torch.Tensor) -> torch.Tensor:
        """
        计算LAP优先级
        
        Args:
            td_error: TD误差
            action: 动作
            
        Returns:
            LAP调整的优先级
        """
        # 基础TD误差优先级
        base_priority = torch.abs(td_error)
        
        # LAP组件：基于动作的额外优先级
        action_norm = torch.norm(action, dim=-1, keepdim=True)
        lap_bonus = self.lap_weight * action_norm
        
        # 组合优先级
        lap_priority = base_priority + lap_bonus
        
        return lap_priority.cpu().numpy()
    
    def __len__(self):
        """返回缓冲区当前大小"""
        return min(self.tree.n_entries, self.capacity)
    
    @property
    def is_ready(self):
        """检查是否有足够样本进行训练"""
        return len(self) >= self.batch_size
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓冲区统计信息"""
        if len(self) == 0:
            return {
                'buffer_size': 0,
                'beta': self.beta,
                'max_priority': self.max_priority,
                'total_priority': 0.0
            }
        
        return {
            'buffer_size': len(self),
            'beta': self.beta,
            'max_priority': self.max_priority,
            'total_priority': self.tree.total(),
            'avg_priority': self.tree.total() / len(self) if len(self) > 0 else 0
        }