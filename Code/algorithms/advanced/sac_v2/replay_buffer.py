"""
SAC v2 Replay Buffer
Experience replay buffer for SAC v2 algorithm
"""

import numpy as np
import torch
import random
from typing import Dict, Tuple, Any, Optional
from collections import deque


class SAC_ReplayBuffer:
    """SAC experience replay buffer"""

    def __init__(self,
                 capacity: int = 100000,
                 batch_size: int = 256,
                 device: torch.device = torch.device('cpu')):
        """
        Initialize replay buffer

        Args:
            capacity: Buffer capacity
            batch_size: Batch size
            device: Computing device
        """
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device

        # Use deque to store experiences
        self.buffer = deque(maxlen=capacity)

        # Statistics
        self.total_samples = 0
        
        print(f"SAC Replay Buffer initialized")
        print(f"   Capacity: {capacity:,}")
        print(f"   Batch size: {batch_size}")
    
    def add(self,
            state: np.ndarray,
            action: np.ndarray,
            reward: float,
            next_state: np.ndarray,
            done: bool):
        """
        Add an experience to buffer

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
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
        Sample a batch from buffer

        Args:
            batch_size: Batch size, if None use default size

        Returns:
            Batch dictionary, returns None if insufficient samples
        """
        if batch_size is None:
            batch_size = self.batch_size

        if len(self.buffer) < batch_size:
            return None

        # Random sampling
        batch = random.sample(self.buffer, batch_size)

        # Separate data
        states = np.array([exp['state'] for exp in batch])
        actions = np.array([exp['action'] for exp in batch])
        rewards = np.array([exp['reward'] for exp in batch])
        next_states = np.array([exp['next_state'] for exp in batch])
        dones = np.array([exp['done'] for exp in batch])

        # Convert to tensors
        batch_tensors = {
            'states': torch.FloatTensor(states).to(self.device),
            'actions': torch.FloatTensor(actions).to(self.device),
            'rewards': torch.FloatTensor(rewards).unsqueeze(1).to(self.device),
            'next_states': torch.FloatTensor(next_states).to(self.device),
            'dones': torch.FloatTensor(dones.astype(np.float32)).unsqueeze(1).to(self.device)
        }
        
        return batch_tensors
    
    def __len__(self):
        """Return current buffer size"""
        return len(self.buffer)

    @property
    def is_ready(self):
        """Check if there are enough samples for training"""
        return len(self.buffer) >= self.batch_size

    def clear(self):
        """Clear buffer"""
        self.buffer.clear()
        self.total_samples = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        if len(self.buffer) == 0:
            return {
                'buffer_size': 0,
                'total_samples': self.total_samples,
                'fill_percentage': 0.0,
                'avg_reward': 0.0,
                'avg_episode_length': 0.0
            }

        # Calculate statistics
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
