"""
IMPALA Agent Implementation
IMPALA agent implementation integrating V-trace and Actor-Critic architecture
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional, List, Any
import random
from collections import deque

from .networks import create_impala_network
from .replay_buffer import IMPALAReplayBuffer
from .vtrace import VTrace, compute_vtrace_loss


class IMPALAAgent:
    """IMPALA agent"""

    def __init__(self,
                 state_space,
                 action_space,
                 config: Dict = None):
        """
        Initialize IMPALA agent

        Args:
            state_space: State space
            action_space: Action space
            config: Configuration parameters
        """

        # Optimized configuration - conservative V-trace policy to prevent early collapse
        default_config = {
            # Network configuration
            'hidden_dim': 512,
            'num_layers': 2,

            # Learning parameters - further reduced learning rate to prevent late-stage collapse
            'learning_rate': 3e-5,      # Optimization v2: 5e-5 → 3e-5 (prevent 150k step collapse)
            'gamma': 0.99,
            'entropy_coeff': 0.01,
            'value_loss_coeff': 0.5,
            'gradient_clip': 20.0,      # Optimization: 40.0 → 20.0 (stronger gradient clipping)

            # V-trace parameters - extremely conservative to avoid importance sampling explosion
            'rho_bar': 0.7,             # Optimization v2: 0.9 → 0.7 (more conservative IS clipping)
            'c_bar': 0.7,               # Optimization v2: 0.9 → 0.7 (more conservative value clipping)

            # Replay buffer - reduced buffer to decrease policy staleness
            'buffer_size': 30000,       # Optimization v2: 50000 → 30000 (reduce off-policy degree)
            'sequence_length': 10,      # Optimization: 20 → 10 (shorter sequences for stability)
            'batch_size': 32,           # Optimization: 16 → 32 (increase batch size)

            # Training parameters - more frequent updates but delayed start
            'learning_starts': 2000,    # Optimization: 1000 → 2000 (delay learning to accumulate more experience)
            'train_freq': 2,            # Optimization: 4 → 2 (more frequent training)
            'update_freq': 100,

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

        # Create network
        network_config = {
            'hidden_dim': self.config['hidden_dim'],
            'num_layers': self.config['num_layers']
        }

        self.network = create_impala_network(
            state_space, action_space, network_config
        ).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.config['learning_rate']
        )

        # Experience replay buffer
        self.replay_buffer = IMPALAReplayBuffer(
            capacity=self.config['buffer_size'],
            sequence_length=self.config['sequence_length'],
            device=self.device
        )

        # V-trace
        self.vtrace = VTrace(
            rho_bar=self.config['rho_bar'],
            c_bar=self.config['c_bar'],
            gamma=self.config['gamma']
        )

        # Training statistics
        self.training_step = 0
        self.episode_rewards = []
        self.losses = []

        # Current episode behavior policy log_probs (for V-trace)
        self.behavior_log_probs = []

        print(f"🎯 IMPALA Agent initialized on {self.device}")
        print(f"   State space: {state_space.shape}")
        print(f"   Action space: {action_space.shape}")
        print(f"   Network: Actor-Critic with V-trace")
        print(f"   Sequence length: {self.config['sequence_length']}")

    def act(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Select action"""
        if not isinstance(state, np.ndarray):
            state = np.array(state)

        # Convert to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Get action and value
            action, log_prob, value = self.network.get_action_and_value(
                state_tensor, deterministic=not training
            )

            action = action.cpu().numpy()[0]
            log_prob = log_prob.cpu().numpy()[0]
            value = value.cpu().numpy()[0]

        # Store behavior policy log_prob for V-trace
        if training:
            self.behavior_log_probs.append(log_prob[0])  # Extract scalar value

        # Ensure action is within valid range
        action = np.clip(action, self.action_space.low, self.action_space.high)

        return action

    def store_transition(self,
                        state: np.ndarray,
                        action: np.ndarray,
                        reward: float,
                        next_state: np.ndarray,
                        done: bool):
        """Store transition to buffer"""

        # Get value estimate for current state
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            _, _, value = self.network.get_action_and_value(state_tensor)
            value = value.cpu().numpy()[0, 0]

        # Get behavior policy log_prob
        behavior_log_prob = self.behavior_log_probs[-1] if self.behavior_log_probs else 0.0

        # Store to replay buffer
        self.replay_buffer.add_step(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            log_prob=behavior_log_prob,
            value=value
        )

        # If episode ends, clear behavior policy log_probs
        if done:
            self.behavior_log_probs.clear()

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

        # Occasionally check batch shapes (reduce output frequency)
        if self.training_step % 5000 == 0:
            print(f"Debug: Batch shapes - states: {batch['states'].shape}, actions: {batch['actions'].shape}")

        # Unpack batch data
        states = batch['states']  # [T, B, state_dim]
        actions = batch['actions']  # [T, B, action_dim]
        rewards = batch['rewards']  # [T, B]
        dones = batch['dones']  # [T, B]
        behavior_log_probs = batch['log_probs']  # [T, B]

        T, B = states.shape[:2]

        # Forward pass to get current policy outputs
        states_flat = states.reshape(-1, states.shape[-1])
        actions_flat = actions.reshape(-1, actions.shape[-1])

        target_log_probs_flat, values_flat, entropies_flat = self.network.evaluate_action(
            states_flat, actions_flat
        )

        # Reshape
        target_log_probs = target_log_probs_flat.reshape(T, B)
        values = values_flat.reshape(T, B)
        entropies = entropies_flat.reshape(T, B)

        # Compute bootstrap value (value of last state)
        with torch.no_grad():
            last_states = states[-1]  # [B, state_dim]
            _, _, bootstrap_values = self.network.get_action_and_value(last_states)
            bootstrap_values = bootstrap_values.squeeze(-1)  # [B]

        # Compute V-trace loss
        total_loss, loss_info = compute_vtrace_loss(
            self.vtrace,
            behavior_log_probs,
            target_log_probs,
            rewards,
            values,
            bootstrap_values,
            dones,
            entropies,
            entropy_coeff=self.config['entropy_coeff']
        )

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.network.parameters(),
            self.config['gradient_clip']
        )

        self.optimizer.step()

        self.training_step += 1
        self.losses.append(loss_info['total_loss'])

        # Return training information
        return {
            'total_loss': loss_info['total_loss'],
            'pg_loss': loss_info['pg_loss'],
            'value_loss': loss_info['value_loss'],
            'entropy_loss': loss_info['entropy_loss'],
            'mean_advantage': loss_info['mean_advantage'],
            'mean_value': loss_info['mean_value'],
            'buffer_size': len(self.replay_buffer)
        }

    def save(self, filepath: str):
        """Save model"""
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config,
            'training_step': self.training_step
        }, filepath)

    def load(self, filepath: str):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.training_step = checkpoint['training_step']

        print(f"✅ IMPALA model loaded from {filepath}")

    def get_stats(self) -> Dict:
        """Get training statistics"""
        return {
            'training_step': self.training_step,
            'buffer_size': len(self.replay_buffer),
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0,
            'episodes_trained': len(self.episode_rewards)
        }