"""
IMPALA Replay Buffer
Experience replay buffer for IMPALA algorithm, supporting sequence storage and batch sampling
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Any, Optional
from collections import deque
import random


class Episode:
    """Storage for a single episode"""

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.next_states = []
    
    def add_step(self, state, action, reward, next_state, done, log_prob, value):
        """Add data for one timestep"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def to_tensors(self, device: torch.device):
        """Convert to tensor format"""
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
    """IMPALA experience replay buffer"""

    def __init__(self, 
                 capacity: int = 10000,
                 sequence_length: int = 20,
                 device: torch.device = torch.device('cpu')):
        """
        Initialize replay buffer

        Args:
            capacity: Maximum number of episodes
            sequence_length: Sequence length (for truncating long episodes)
            device: Computing device
        """
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.device = device
        
        self.episodes = deque(maxlen=capacity)
        self.current_episode = Episode()
        
    def add_step(self, state, action, reward, next_state, done, log_prob, value):
        """Add one timestep"""
        self.current_episode.add_step(state, action, reward, next_state, done, log_prob, value)

        # If episode ends, save and start new episode
        if done:
            if len(self.current_episode) > 0:
                self.episodes.append(self.current_episode)
            self.current_episode = Episode()
    
    def sample_sequences(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample sequence batches

        Args:
            batch_size: Batch size

        Returns:
            Dictionary containing sequence data
        """
        if len(self.episodes) < batch_size:
            return None
        
        # Randomly select episodes
        selected_episodes = random.sample(list(self.episodes), batch_size)

        # Convert episodes to fixed-length sequences
        batch_sequences = []
        
        for episode in selected_episodes:
            episode_data = episode.to_tensors(self.device)
            episode_length = len(episode)
            
            if episode_length >= self.sequence_length:
                # Episode is long enough, randomly select a starting position
                start_idx = random.randint(0, episode_length - self.sequence_length)
                sequence = {
                    key: value[start_idx:start_idx + self.sequence_length]
                    for key, value in episode_data.items()
                }
            else:
                # Episode is not long enough, perform padding
                sequence = {}
                for key, value in episode_data.items():
                    if key in ['dones', 'rewards']:
                        # For dones and rewards, pad with 0
                        padding = torch.zeros(self.sequence_length - episode_length).to(self.device)
                    else:
                        # For other tensors, pad with last value
                        last_value = value[-1] if len(value) > 0 else torch.zeros_like(value[0])
                        padding = last_value.unsqueeze(0).repeat(self.sequence_length - episode_length, *([1] * (len(last_value.shape))))
                    
                    sequence[key] = torch.cat([value, padding], dim=0)
            
            batch_sequences.append(sequence)
        
        # Stack all sequences into batch [batch_size, sequence_length, ...]
        batch = {}
        for key in batch_sequences[0].keys():
            batch[key] = torch.stack([seq[key] for seq in batch_sequences], dim=0)

        # Rearrange dimensions to [sequence_length, batch_size, ...]
        for key, value in batch.items():
            batch[key] = value.transpose(0, 1)
        
        return batch
    
    def get_recent_trajectories(self, num_episodes: int = None) -> List[Dict[str, torch.Tensor]]:
        """
        Get recent trajectories

        Args:
            num_episodes: Number of episodes to get, None means get all

        Returns:
            List of trajectories
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
        """Clear buffer"""
        self.episodes.clear()
        self.current_episode = Episode()
    
    def __len__(self):
        return len(self.episodes)
    
    @property
    def size(self):
        """Total steps in buffer"""
        return sum(len(episode) for episode in self.episodes)

    @property
    def is_ready(self):
        """Whether there is enough data for sampling"""
        return len(self.episodes) > 10  # At least 10 episodes needed


def collate_sequences(sequences: List[Dict[str, torch.Tensor]],
                     max_length: int = None) -> Dict[str, torch.Tensor]:
    """
    Pack sequences of unequal length into batches

    Args:
        sequences: List of sequences
        max_length: Maximum sequence length

    Returns:
        Batch dictionary
    """
    if not sequences:
        return {}
    
    # Determine maximum length
    if max_length is None:
        max_length = max(seq['states'].shape[0] for seq in sequences)

    batch_size = len(sequences)
    padded_batch = {}

    # Initialize batch tensors
    for key in sequences[0].keys():
        example_tensor = sequences[0][key]
        if len(example_tensor.shape) == 1:
            # 1D tensors (like rewards, dones)
            padded_batch[key] = torch.zeros(max_length, batch_size, device=example_tensor.device)
        else:
            # Multi-dimensional tensors (like states, actions)
            shape = (max_length, batch_size) + example_tensor.shape[1:]
            padded_batch[key] = torch.zeros(shape, device=example_tensor.device)
    
    # Fill data
    for i, seq in enumerate(sequences):
        seq_len = min(seq['states'].shape[0], max_length)
        for key, tensor in seq.items():
            if len(tensor.shape) == 1:
                padded_batch[key][:seq_len, i] = tensor[:seq_len]
            else:
                padded_batch[key][:seq_len, i] = tensor[:seq_len]
    
    return padded_batch