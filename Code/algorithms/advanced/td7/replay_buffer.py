"""
TD7 Replay Buffer with Prioritized Experience Replay
Prioritized experience replay buffer and LAP mechanism for TD7 algorithm
"""

import numpy as np
import torch
import random
from typing import Dict, Tuple, Any, Optional, List
from collections import deque
import heapq


class SumTree:
    """SumTree data structure for prioritized experience replay"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.pending_idx = set()

    def _propagate(self, idx: int, change: float):
        """Propagate priority change upward"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        """Retrieve leaf node index"""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        """Return sum of all priorities"""
        return self.tree[0]

    def add(self, p: float, data):
        """Add experience"""
        idx = self.n_entries % self.capacity
        data_idx = idx + self.capacity - 1

        self.data[idx] = data
        self.update(data_idx, p)

        self.n_entries += 1

    def update(self, idx: int, p: float):
        """Update priority"""
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, Any]:
        """Sample based on priority"""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class TD7_PrioritizedReplayBuffer:
    """TD7 prioritized experience replay buffer with LAP mechanism"""

    def __init__(self,
                 capacity: int = 1000000,
                 batch_size: int = 256,
                 alpha: float = 0.6,  # Priority exponent
                 beta: float = 0.4,   # Importance sampling exponent
                 beta_increment: float = 0.001,
                 device: torch.device = torch.device('cpu')):
        """
        Initialize prioritized replay buffer

        Args:
            capacity: Buffer capacity
            batch_size: Batch size
            alpha: Priority exponent
            beta: Importance sampling exponent
            beta_increment: Beta growth rate
            device: Compute device
        """
        self.capacity = capacity
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.device = device

        self.tree = SumTree(capacity)
        self.epsilon = 0.01  # Minimum priority
        self.max_priority = 1.0

        # LAP (Learned Action Prioritization) component
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
        Add experience to buffer

        Args:
            state: Current state
            action: Executed action
            reward: Received reward
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

        # New experiences use maximum priority
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, experience)

    def sample(self, batch_size: Optional[int] = None) -> Optional[Dict[str, torch.Tensor]]:
        """
        Sample a batch from buffer

        Args:
            batch_size: Batch size, uses default if None

        Returns:
            Batch dictionary, returns None if insufficient samples
        """
        if batch_size is None:
            batch_size = self.batch_size

        if self.tree.n_entries < batch_size:
            return None

        batch_idx = []
        batch_experiences = []
        priorities = []

        # Segment sampling
        segment = self.tree.total() / batch_size

        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)

            idx, priority, experience = self.tree.get(s)

            batch_idx.append(idx)
            batch_experiences.append(experience)
            priorities.append(priority)

        # Compute importance sampling weights
        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weights = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()

        # Separate data
        states = np.array([exp['state'] for exp in batch_experiences])
        actions = np.array([exp['action'] for exp in batch_experiences])
        rewards = np.array([exp['reward'] for exp in batch_experiences])
        next_states = np.array([exp['next_state'] for exp in batch_experiences])
        dones = np.array([exp['done'] for exp in batch_experiences])

        # Convert to tensors
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
        Update priorities

        Args:
            indices: Sample indices
            priorities: New priorities
        """
        for idx, priority in zip(indices, priorities):
            # Ensure priority is positive
            priority = abs(priority) + self.epsilon

            # LAP weighting
            if self.use_lap:
                priority = priority * (1 + self.lap_weight)

            # Update maximum priority
            self.max_priority = max(self.max_priority, priority)

            # Apply alpha exponent
            self.tree.update(idx, priority ** self.alpha)

    def compute_lap_priority(self,
                           td_error: torch.Tensor,
                           action: torch.Tensor) -> torch.Tensor:
        """
        Compute LAP priority

        Args:
            td_error: TD error
            action: Action

        Returns:
            LAP-adjusted priority
        """
        # Base TD error priority
        base_priority = torch.abs(td_error)

        # LAP component: additional priority based on action
        action_norm = torch.norm(action, dim=-1, keepdim=True)
        lap_bonus = self.lap_weight * action_norm

        # Combined priority
        lap_priority = base_priority + lap_bonus

        return lap_priority.cpu().numpy()

    def __len__(self):
        """Return current buffer size"""
        return min(self.tree.n_entries, self.capacity)

    @property
    def is_ready(self):
        """Check if there are enough samples for training"""
        return len(self) >= self.batch_size

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
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