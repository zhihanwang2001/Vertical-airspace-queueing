"""
R2D2 Agent Implementation
R2D2 agent implementation integrating recurrent networks and sequence replay
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List, Any
import random
import copy
from collections import deque

from .networks import create_r2d2_network
from .sequence_replay import R2D2SequenceReplayBuffer


class R2D2Agent:
    """R2D2 agent"""

    def __init__(self,
                 state_space,
                 action_space,
                 config: Dict = None):
        """
        Initialize R2D2 agent

        Args:
            state_space: State space
            action_space: Action space
            config: Configuration parameters
        """

        # Default configuration
        default_config = {
            # Network configuration
            'hidden_dim': 512,
            'recurrent_dim': 256,
            'num_layers': 1,
            'recurrent_type': 'LSTM',
            'dueling': True,

            # Learning parameters
            'learning_rate': 1e-4,
            'gamma': 0.99,
            'target_update_freq': 2500,
            'gradient_clip': 40.0,

            # DQN parameters
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay_steps': 250000,
            'double_dqn': True,

            # Sequence replay configuration
            'buffer_size': 5000,
            'sequence_length': 40,
            'burn_in_length': 20,
            'overlap_length': 10,
            'batch_size': 16,

            # Training parameters
            'learning_starts': 5000,
            'train_freq': 4,

            # Action discretization (continuous action space)
            'action_bins': 3,

            # Other
            'seed': 42,
            'device': 'auto'
        }
        
        if config:
            default_config.update(config)
        self.config = default_config

        # Set device
        if self.config['device'] == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.config['device'])

        # Set random seed
        if self.config['seed'] is not None:
            random.seed(self.config['seed'])
            np.random.seed(self.config['seed'])
            torch.manual_seed(self.config['seed'])

        self.state_space = state_space
        self.action_space = action_space

        # Handle action space
        self._setup_action_space()

        # Create networks
        network_config = {
            'hidden_dim': self.config['hidden_dim'],
            'recurrent_dim': self.config['recurrent_dim'],
            'num_layers': self.config['num_layers'],
            'recurrent_type': self.config['recurrent_type'],
            'dueling': self.config['dueling'],
            'action_bins': self.config['action_bins']
        }
        
        self.q_network = create_r2d2_network(
            state_space, action_space, network_config
        ).to(self.device)
        
        self.target_network = create_r2d2_network(
            state_space, action_space, network_config
        ).to(self.device)

        # Synchronize target network
        self.update_target_network()

        # Optimizer
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=self.config['learning_rate']
        )

        # Sequence experience replay buffer
        self.replay_buffer = R2D2SequenceReplayBuffer(
            capacity=self.config['buffer_size'],
            sequence_length=self.config['sequence_length'],
            burn_in_length=self.config['burn_in_length'],
            overlap_length=self.config['overlap_length'],
            device=self.device
        )

        # Training statistics
        self.training_step = 0
        self.episode_rewards = []
        self.losses = []

        # RNN state management
        self.current_hidden_state = None
        self.reset_hidden_state()
        
        print(f"🔄 R2D2 Agent initialized on {self.device}")
        print(f"   State space: {state_space.shape}")
        if self.action_type == 'discrete':
            print(f"   Action space: {self.num_actions} (discrete)")
        else:
            print(f"   Action space: {self.action_dim}D -> {self.num_actions} discrete")
        print(f"   Recurrent: {self.config['recurrent_type']} ({self.config['recurrent_dim']} units)")
        print(f"   Sequence length: {self.config['sequence_length']} + {self.config['burn_in_length']} burn-in")
    
    def _setup_action_space(self):
        """Setup action space"""
        if hasattr(self.action_space, 'n'):
            # Discrete action space
            self.num_actions = self.action_space.n
            self.action_type = 'discrete'
            self.action_dim = None
        else:
            # Continuous action space, needs discretization
            self.action_dim = self.action_space.shape[0]
            self.action_low = self.action_space.low
            self.action_high = self.action_space.high
            self.action_bins = self.config['action_bins']
            self.num_actions = self.action_bins ** self.action_dim
            self.action_type = 'continuous'

            # Create action mapping
            self._create_action_mapping()

    def _create_action_mapping(self):
        """Create intelligent discretization mapping for continuous action space"""
        if self.action_type == 'discrete':
            return

        # Intelligent discretization: only use key action values
        # For most control tasks, {-1, 0, 1} or {-0.5, 0, 0.5} is sufficient
        self.action_grids = []
        for i in range(self.action_dim):
            if self.action_bins == 2:
                # Binary control: only negative and positive values
                grid = np.array([self.action_low[i], self.action_high[i]])
            elif self.action_bins == 3:
                # Three-value control: negative, zero, positive
                grid = np.array([self.action_low[i], 0.0, self.action_high[i]])
            else:
                # Keep original linear distribution
                grid = np.linspace(self.action_low[i], self.action_high[i], self.action_bins)
            self.action_grids.append(grid)
            
        print(f"🎯 R2D2 Action discretization: {self.action_bins}^{self.action_dim} = {self.num_actions} actions")
        print(f"   First dimension grid: {self.action_grids[0]}")
    
    def _discrete_to_continuous_action(self, discrete_action: int) -> np.ndarray:
        """Convert discrete action to continuous action"""
        if self.action_type == 'discrete':
            return discrete_action

        continuous_action = np.zeros(self.action_dim)
        remaining = discrete_action

        for i in range(self.action_dim):
            idx = remaining % self.action_bins
            continuous_action[i] = self.action_grids[i][idx]
            remaining //= self.action_bins

        return continuous_action

    def reset_hidden_state(self):
        """Reset RNN hidden state"""
        self.current_hidden_state = self.q_network.init_hidden_state(1, self.device)

    def get_epsilon(self) -> float:
        """Get current epsilon value"""
        if self.training_step < self.config['epsilon_decay_steps']:
            epsilon = self.config['epsilon_start'] - (
                self.config['epsilon_start'] - self.config['epsilon_end']
            ) * self.training_step / self.config['epsilon_decay_steps']
        else:
            epsilon = self.config['epsilon_end']
        return epsilon
    
    def act(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Select action"""
        if not isinstance(state, np.ndarray):
            state = np.array(state)

        # Convert to tensor and add batch and sequence dimensions
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)

        # epsilon-greedy policy
        if training and random.random() < self.get_epsilon():
            # Random action
            discrete_action = random.randint(0, self.num_actions - 1)
            # Still need to update hidden state
            with torch.no_grad():
                _, self.current_hidden_state = self.q_network(state_tensor, self.current_hidden_state)
        else:
            # Greedy action
            with torch.no_grad():
                q_values, self.current_hidden_state = self.q_network(state_tensor, self.current_hidden_state)
                discrete_action = q_values.squeeze(0).squeeze(0).argmax().item()

        # Convert to action format required by environment
        if self.action_type == 'continuous':
            return self._discrete_to_continuous_action(discrete_action)
        else:
            return discrete_action

    def store_transition(self,
                        state: np.ndarray,
                        action,
                        reward: float,
                        next_state: np.ndarray,
                        done: bool):
        """Store transition to sequence buffer"""

        # If continuous action, convert to discrete action index
        if self.action_type == 'continuous':
            discrete_action = self._continuous_to_discrete_action(action)
        else:
            discrete_action = action

        # Store to sequence replay buffer
        self.replay_buffer.add_step(
            state=state,
            action=discrete_action,
            reward=reward,
            done=done,
            hidden_state=copy.deepcopy(self.current_hidden_state) if self.current_hidden_state else None
        )

        # If episode ends, reset hidden state
        if done:
            self.reset_hidden_state()

    def _continuous_to_discrete_action(self, action: np.ndarray) -> int:
        """Convert continuous action to discrete action index"""
        discrete_action = 0
        multiplier = 1
        
        for i in range(self.action_dim):
            closest_idx = np.argmin(np.abs(self.action_grids[i] - action[i]))
            discrete_action += closest_idx * multiplier
            multiplier *= self.action_bins
        
        return discrete_action
    
    def train(self) -> Optional[Dict]:
        """Train one step"""
        if not self.replay_buffer.is_ready:
            return None

        if self.training_step % self.config['train_freq'] != 0:
            self.training_step += 1
            return None

        # Sample sequence batch
        batch = self.replay_buffer.sample_sequences(self.config['batch_size'])
        if batch is None:
            self.training_step += 1
            return None

        # Unpack batch data
        states = batch['states']  # [batch_size, seq_len, state_dim]
        actions = batch['actions'].long()  # [batch_size, seq_len]
        rewards = batch['rewards']  # [batch_size, seq_len]
        dones = batch['dones']  # [batch_size, seq_len]
        burn_in_states = batch['burn_in_states']  # [batch_size, burn_in_len, state_dim]
        sequence_lengths = batch['sequence_lengths']  # [batch_size]

        batch_size, seq_len = states.shape[:2]

        # Burn-in phase: warm up RNN hidden state
        burn_in_hidden_states = []
        for i in range(batch_size):
            hidden_state = self.q_network.init_hidden_state(1, self.device)

            # If burn-in data exists, perform warm-up
            if burn_in_states.shape[1] > 0:
                burn_in_seq = burn_in_states[i:i+1]  # [1, burn_in_len, state_dim]
                with torch.no_grad():
                    _, hidden_state = self.q_network(burn_in_seq, hidden_state)

            burn_in_hidden_states.append(hidden_state)

        # Merge hidden states into batch format
        if self.config['recurrent_type'].upper() == 'LSTM':
            h_states = torch.cat([h for h, c in burn_in_hidden_states], dim=1)
            c_states = torch.cat([c for h, c in burn_in_hidden_states], dim=1)
            batch_hidden_state = (h_states, c_states)
        else:  # GRU
            h_states = torch.cat([h for h in burn_in_hidden_states], dim=1)
            batch_hidden_state = (h_states,)

        # Forward pass to compute current Q-values
        current_q_values, _ = self.q_network(states, batch_hidden_state)
        current_q_values = current_q_values.gather(2, actions.unsqueeze(2)).squeeze(2)

        # Compute target Q-values
        with torch.no_grad():
            if self.config['double_dqn']:
                # Double DQN: use main network to select actions, target network to evaluate
                next_q_values_main, _ = self.q_network(states, batch_hidden_state)
                next_actions = next_q_values_main.argmax(2)

                next_q_values_target, _ = self.target_network(states, batch_hidden_state)
                next_q_values = next_q_values_target.gather(2, next_actions.unsqueeze(2)).squeeze(2)
            else:
                # Standard DQN
                next_q_values_target, _ = self.target_network(states, batch_hidden_state)
                next_q_values = next_q_values_target.max(2)[0]

            # Compute target values
            target_q_values = rewards + self.config['gamma'] * next_q_values * (1 - dones)

        # Compute loss (only for valid timesteps)
        loss = 0
        valid_steps = 0
        
        for i in range(batch_size):
            seq_len_i = min(sequence_lengths[i].item(), seq_len)
            if seq_len_i > 1:  # 至少需要2个时间步来计算TD误差
                loss += F.mse_loss(
                    current_q_values[i, :seq_len_i-1],
                    target_q_values[i, 1:seq_len_i]
                )
                valid_steps += seq_len_i - 1
        
        if valid_steps > 0:
            loss = loss / batch_size
        else:
            self.training_step += 1
            return None

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.q_network.parameters(),
            self.config['gradient_clip']
        )

        self.optimizer.step()

        # Update target network
        if self.training_step % self.config['target_update_freq'] == 0:
            self.update_target_network()

        self.training_step += 1
        self.losses.append(loss.item())

        # Return training information
        return {
            'loss': loss.item(),
            'epsilon': self.get_epsilon(),
            'buffer_size': len(self.replay_buffer),
            'valid_steps': valid_steps,
            'avg_q_value': current_q_values.mean().item()
        }

    def update_target_network(self):
        """Update target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save(self, filepath: str):
        """Save model"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config,
            'training_step': self.training_step
        }, filepath)

    def load(self, filepath: str):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.training_step = checkpoint['training_step']

        print(f"✅ R2D2 model loaded from {filepath}")

    def get_stats(self) -> Dict:
        """Get training statistics"""
        buffer_stats = self.replay_buffer.get_stats()
        
        return {
            'training_step': self.training_step,
            'epsilon': self.get_epsilon(),
            'buffer_size': len(self.replay_buffer),
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0,
            'episodes_trained': len(self.episode_rewards),
            **buffer_stats
        }