"""
Prioritized Experience Replay for Rainbow DQN
Implements TD-error based prioritized experience replay
"""

import numpy as np
import torch
import random
from collections import namedtuple
from typing import List, Tuple, Optional


# Experience tuple
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])


class SumTree:
    """Sum Tree data structure for efficient priority sampling"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.data_pointer = 0
        self.size = 0
    
    def add(self, priority: float, data):
        """Add new experience"""
        tree_index = self.data_pointer + self.capacity - 1
        
        self.data[self.data_pointer] = data
        self.update(tree_index, priority)
        
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0
        
        if self.size < self.capacity:
            self.size += 1
    
    def update(self, tree_index: int, priority: float):
        """Update priority"""
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        
        # Propagate change upward
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change
    
    def get_leaf(self, value: float) -> Tuple[int, float, object]:
        """Get leaf node based on cumulative probability"""
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
        """Total priority"""
        return self.tree[0]


class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer"""
    
    def __init__(self, 
                 capacity: int, 
                 alpha: float = 0.6, 
                 beta: float = 0.4, 
                 beta_increment: float = 0.001,
                 epsilon: float = 1e-6):
        """
        Args:
            capacity: Buffer capacity
            alpha: Priority exponent (0=uniform sampling, 1=full prioritization)
            beta: Importance sampling exponent (0=no correction, 1=full correction)
            beta_increment: Beta growth rate
            epsilon: Minimum priority to prevent experiences from never being sampled
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        
        self.tree = SumTree(capacity)
        self.max_priority = 1.0
        
    def add(self, state, action, reward, next_state, done):
        """Add experience"""
        experience = Experience(state, action, reward, next_state, done)
        
        # New experiences use maximum priority
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, experience)
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, List[int]]:
        """Sample a batch of experiences"""
        batch = []
        indices = []
        priorities = []
        
        # Calculate sampling segments
        priority_segment = self.tree.total_priority / batch_size
        
        for i in range(batch_size):
            # Random sample within each segment
            left = priority_segment * i
            right = priority_segment * (i + 1)
            value = random.uniform(left, right)
            
            index, priority, experience = self.tree.get_leaf(value)
            
            batch.append(experience)
            indices.append(index)
            priorities.append(priority)
        
        # Calculate importance weights
        sampling_probabilities = np.array(priorities) / self.tree.total_priority
        weights = np.power(self.tree.size * sampling_probabilities, -self.beta)
        weights = weights / weights.max()  # Normalize
        
        return batch, weights, indices
    
    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update experience priorities"""
        for index, priority in zip(indices, priorities):
            # Ensure priority is non-negative and add epsilon
            priority = abs(priority) + self.epsilon
            self.tree.update(index, priority ** self.alpha)
            self.max_priority = max(self.max_priority, priority)
    
    def update_beta(self):
        """Update beta value"""
        self.beta = min(1.0, self.beta + self.beta_increment)
    
    def __len__(self) -> int:
        return self.tree.size
    
    @property
    def is_ready(self) -> bool:
        """Whether there are enough experiences to start training"""
        return len(self) > 1000  # At least 1000 experiences


class UniformReplayBuffer:
    """Standard uniform experience replay (for comparison)"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def add(self, state, action, reward, next_state, done):
        """Add experience"""
        experience = Experience(state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, List[int]]:
        """Uniform sampling"""
        batch = random.sample(self.buffer, batch_size)
        weights = np.ones(batch_size)  # Uniform weights
        indices = list(range(batch_size))  # Dummy indices
        return batch, weights, indices
    
    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Empty implementation to maintain interface consistency"""
        pass
    
    def update_beta(self):
        """Empty implementation"""
        pass
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    @property
    def is_ready(self) -> bool:
        return len(self) > 1000


def create_replay_buffer(buffer_type: str = "prioritized", **kwargs):
    """Factory function: create experience replay buffer"""
    if buffer_type == "prioritized":
        return PrioritizedReplayBuffer(**kwargs)
    elif buffer_type == "uniform":
        return UniformReplayBuffer(kwargs.get('capacity', 100000))
    else:
        raise ValueError(f"Unknown buffer type: {buffer_type}")


def batch_to_tensors(batch: List[Experience], device: torch.device):
    """Convert batch to tensors"""
    states = torch.FloatTensor([e.state for e in batch]).to(device)
    actions = torch.LongTensor([e.action for e in batch]).to(device)
    rewards = torch.FloatTensor([e.reward for e in batch]).to(device)
    next_states = torch.FloatTensor([e.next_state for e in batch]).to(device)
    dones = torch.BoolTensor([e.done for e in batch]).to(device)
    
    return states, actions, rewards, next_states, dones
