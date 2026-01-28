"""
R2D2 Sequence Replay Buffer
Sequence replay buffer for R2D2 algorithm, supporting overlapping sequences and burn-in
"""

import numpy as np
import torch
import random
from typing import List, Dict, Tuple, Any, Optional
from collections import deque
import copy


class SequenceBuffer:
    """Single sequence storage"""

    def __init__(self, max_length: int = 200):
        self.max_length = max_length
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.hidden_states = []  # Store RNN hidden states
        self.priorities = []  # Priority (optional)

        # Sequence metadata
        self.episode_id = None
        self.start_step = 0

    def add_step(self, state, action, reward, done, hidden_state=None, priority=1.0):
        """Add a timestep"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.hidden_states.append(hidden_state)
        self.priorities.append(priority)

        # If exceeds max length, remove earliest data
        if len(self.states) > self.max_length:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.dones.pop(0)
            self.hidden_states.pop(0)
            self.priorities.pop(0)
            self.start_step += 1

    def get_sequence(self, start_idx: int, length: int) -> Dict:
        """Get sequence of specified length"""
        end_idx = min(start_idx + length, len(self.states))
        actual_length = end_idx - start_idx
        
        sequence = {
            'states': np.array(self.states[start_idx:end_idx]),
            'actions': np.array(self.actions[start_idx:end_idx]),
            'rewards': np.array(self.rewards[start_idx:end_idx]),
            'dones': np.array(self.dones[start_idx:end_idx], dtype=np.float32),
            'hidden_states': self.hidden_states[start_idx:end_idx],
            'priorities': np.array(self.priorities[start_idx:end_idx]),
            'length': actual_length,
            'episode_id': self.episode_id,
            'start_step': self.start_step + start_idx
        }
        
        return sequence
    
    def __len__(self):
        return len(self.states)
    
    def is_valid_start(self, start_idx: int, min_length: int) -> bool:
        """Check if sampling can start from this position"""
        return start_idx + min_length <= len(self.states)


class R2D2SequenceReplayBuffer:
    """R2D2 sequence experience replay buffer"""

    def __init__(self,
                 capacity: int = 10000,
                 sequence_length: int = 40,
                 burn_in_length: int = 20,
                 overlap_length: int = 10,
                 device: torch.device = torch.device('cpu')):
        """
        Initialize sequence replay buffer

        Args:
            capacity: Maximum number of sequences
            sequence_length: Training sequence length
            burn_in_length: Burn-in sequence length (for RNN warm-up)
            overlap_length: Sequence overlap length
            device: Computing device
        """
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.burn_in_length = burn_in_length
        self.overlap_length = overlap_length
        self.device = device
        
        self.sequences = deque(maxlen=capacity)
        self.current_sequence = SequenceBuffer()
        self.episode_count = 0

        # Statistics
        self.total_samples = 0
        self.total_episodes = 0

    def add_step(self, state, action, reward, done, hidden_state=None, priority=1.0):
        """Add a timestep to current sequence"""
        self.current_sequence.add_step(state, action, reward, done, hidden_state, priority)
        self.total_samples += 1

        # If episode ends, save current sequence and start new sequence
        if done:
            if len(self.current_sequence) > 0:
                self.current_sequence.episode_id = self.episode_count
                self.sequences.append(copy.deepcopy(self.current_sequence))

            # Start new sequence
            self.current_sequence = SequenceBuffer()
            self.episode_count += 1
            self.total_episodes += 1

    def sample_sequences(self, batch_size: int) -> Optional[Dict]:
        """
        Sample sequence batch

        Args:
            batch_size: Batch size

        Returns:
            Dictionary containing sequence batch, returns None if insufficient samples
        """
        if len(self.sequences) < batch_size:
            return None

        # Randomly select sequences
        selected_sequences = random.sample(list(self.sequences), batch_size)

        # Randomly sample a subsequence from each sequence
        batch_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'burn_in_states': [],
            'burn_in_actions': [],
            'burn_in_hidden_states': [],
            'initial_hidden_states': [],
            'sequence_lengths': [],
            'priorities': []
        }
        
        for seq_buffer in selected_sequences:
            # Determine sampling position
            total_needed = self.burn_in_length + self.sequence_length

            if len(seq_buffer) >= total_needed:
                # Sequence is long enough, randomly select start position
                max_start = len(seq_buffer) - total_needed
                start_idx = random.randint(0, max_start) if max_start > 0 else 0
            else:
                # Sequence too short, start from beginning
                start_idx = 0

            # Get burn-in sequence
            burn_in_end = start_idx + self.burn_in_length
            burn_in_seq = None  # Initialize variable

            if burn_in_end <= len(seq_buffer):
                burn_in_seq = seq_buffer.get_sequence(start_idx, self.burn_in_length)
                batch_data['burn_in_states'].append(burn_in_seq['states'])
                batch_data['burn_in_actions'].append(burn_in_seq['actions'])
                batch_data['burn_in_hidden_states'].append(burn_in_seq['hidden_states'])
            else:
                # Burn-in sequence insufficient, pad with zeros
                actual_burn_in = max(0, len(seq_buffer) - self.sequence_length)
                if actual_burn_in > 0:
                    burn_in_seq = seq_buffer.get_sequence(0, actual_burn_in)
                    # Pad to required length
                    padded_states = self._pad_sequence(burn_in_seq['states'], self.burn_in_length)
                    padded_actions = self._pad_sequence(burn_in_seq['actions'], self.burn_in_length)
                    batch_data['burn_in_states'].append(padded_states)
                    batch_data['burn_in_actions'].append(padded_actions)
                else:
                    # No burn-in data, create zero padding
                    zero_states = np.zeros((self.burn_in_length,) + seq_buffer.states[0].shape)
                    zero_actions = np.zeros((self.burn_in_length,), dtype=np.int32)
                    batch_data['burn_in_states'].append(zero_states)
                    batch_data['burn_in_actions'].append(zero_actions)
                    # Create empty burn_in_seq for later reference
                    burn_in_seq = {'hidden_states': []}
                batch_data['burn_in_hidden_states'].append([None] * self.burn_in_length)

            # Get training sequence
            train_start = max(start_idx, burn_in_end - self.overlap_length)
            train_seq = seq_buffer.get_sequence(train_start, self.sequence_length)

            # Pad sequence to fixed length
            padded_states = self._pad_sequence(train_seq['states'], self.sequence_length)
            padded_actions = self._pad_sequence(train_seq['actions'], self.sequence_length)
            padded_rewards = self._pad_sequence(train_seq['rewards'], self.sequence_length)
            padded_dones = self._pad_sequence(train_seq['dones'], self.sequence_length)
            padded_priorities = self._pad_sequence(train_seq['priorities'], self.sequence_length)

            batch_data['states'].append(padded_states)
            batch_data['actions'].append(padded_actions)
            batch_data['rewards'].append(padded_rewards)
            batch_data['dones'].append(padded_dones)
            batch_data['priorities'].append(padded_priorities)
            batch_data['sequence_lengths'].append(train_seq['length'])

            # Initial hidden state (state after burn-in, or zero state)
            if burn_in_seq and len(burn_in_seq['hidden_states']) > 0:
                initial_hidden = burn_in_seq['hidden_states'][-1]
            else:
                initial_hidden = None
            batch_data['initial_hidden_states'].append(initial_hidden)

        # Convert to tensors
        tensor_batch =
        for key, value in batch_data.items():
            if key in ['burn_in_hidden_states', 'initial_hidden_states']:
                tensor_batch[key] = value  # Keep list format
            elif key == 'sequence_lengths':
                tensor_batch[key] = torch.LongTensor(value).to(self.device)
            else:
                # Ensure correct tensor shape [batch_size, seq_len, ...]
                if key in ['states', 'burn_in_states']:
                    # States need special handling to ensure correct dimension order
                    # Check if all sequences have same shape
                    if len(value) > 0:
                        shapes = [v.shape for v in value]
                        if all(s == shapes[0] for s in shapes):
                            # All shapes same, can stack directly
                            numpy_array = np.stack(value, axis=0)
                        else:
                            # Different shapes, need to manually pad to same size
                            max_seq_len = max(v.shape[0] for v in value)
                            state_dim = value[0].shape[1] if len(value[0].shape) > 1 else 1

                            # Create padded array
                            batch_size = len(value)
                            if len(value[0].shape) > 1:
                                padded_array = np.zeros((batch_size, max_seq_len, state_dim), dtype=value[0].dtype)
                            else:
                                padded_array = np.zeros((batch_size, max_seq_len), dtype=value[0].dtype)

                            # Pad each sequence
                            for i, seq in enumerate(value):
                                seq_len = seq.shape[0]
                                if len(seq.shape) > 1:
                                    padded_array[i, :seq_len, :] = seq
                                else:
                                    padded_array[i, :seq_len] = seq

                            numpy_array = padded_array
                    else:
                        numpy_array = np.array(value)
                else:
                    numpy_array = np.array(value)  # Other data processed normally

                tensor_batch[key] = torch.FloatTensor(numpy_array).to(self.device)

        return tensor_batch

    def _pad_sequence(self, sequence: np.ndarray, target_length: int) -> np.ndarray:
        """Pad sequence to target length"""
        current_length = len(sequence)

        if current_length >= target_length:
            return sequence[:target_length]

        # Need padding
        if len(sequence.shape) == 1:
            # 1D array
            padding = np.zeros(target_length - current_length, dtype=sequence.dtype)
            return np.concatenate([sequence, padding])
        else:
            # Multi-dimensional array
            pad_shape = (target_length - current_length,) + sequence.shape[1:]
            padding = np.zeros(pad_shape, dtype=sequence.dtype)
            return np.concatenate([sequence, padding], axis=0)

    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities (for prioritized experience replay)"""
        # Priority update logic can be implemented here
        # Current simplified version not implemented
        pass

    def clear(self):
        """Clear buffer"""
        self.sequences.clear()
        self.current_sequence = SequenceBuffer()
        self.total_samples = 0
        self.total_episodes = 0

    def __len__(self):
        return len(self.sequences)

    @property
    def size(self):
        """Total steps in buffer"""
        return sum(len(seq) for seq in self.sequences) + len(self.current_sequence)

    @property
    def is_ready(self):
        """Whether there is enough data for sampling"""
        return len(self.sequences) >= 10  # Need at least 10 sequences

    def get_stats(self) -> Dict:
        """Get buffer statistics"""
        return {
            'num_sequences': len(self.sequences),
            'total_samples': self.total_samples,
            'total_episodes': self.total_episodes,
            'current_sequence_length': len(self.current_sequence),
            'average_sequence_length': np.mean([len(seq) for seq in self.sequences]) if self.sequences else 0
        }