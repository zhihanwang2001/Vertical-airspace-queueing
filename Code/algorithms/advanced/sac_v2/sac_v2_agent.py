"""
SAC v2 Agent Implementation
SAC v2 agent implementation with automatic entropy tuning and dual Q-networks
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List, Any
import random

from .networks import create_sac_v2_networks
from .replay_buffer import SAC_ReplayBuffer


class SAC_v2_Agent:
    """SAC v2 agent"""

    def __init__(self,
                 state_space,
                 action_space,
                 config: Dict = None):
        """
        Initialize SAC v2 agent

        Args:
            state_space: State space
            action_space: Action space
            config: Configuration parameters
        """

        # Default configuration
        default_config = {
            # Network configuration
            'hidden_dim': 256,
            'max_action': 1.0,

            # Learning parameters
            'actor_lr': 3e-4,
            'critic_lr': 3e-4,
            'alpha_lr': 3e-4,
            'gamma': 0.99,
            'tau': 0.005,  # Soft update coefficient

            # SAC-specific parameters
            'alpha': 0.2,  # Initial entropy coefficient
            'automatic_entropy_tuning': True,
            'target_entropy_scale': 1.0,  # Target entropy scaling factor

            # Replay buffer
            'buffer_size': 100000,
            'batch_size': 256,

            # Training parameters
            'learning_starts': 10000,
            'train_freq': 1,
            'gradient_steps': 1,

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

        # Create networks
        network_config = {
            'hidden_dim': self.config['hidden_dim'],
            'max_action': self.config['max_action']
        }

        self.networks = create_sac_v2_networks(
            state_space, action_space, network_config
        )
        self.networks.device = self.device

        # Move to correct device
        for network in [self.networks.actor, self.networks.critic1, self.networks.critic2,
                       self.networks.target_critic1, self.networks.target_critic2]:
            network.to(self.device)
        self.networks.log_alpha = self.networks.log_alpha.to(self.device)

        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.networks.actor.parameters(),
            lr=self.config['actor_lr']
        )
        
        self.critic1_optimizer = optim.Adam(
            self.networks.critic1.parameters(),
            lr=self.config['critic_lr']
        )
        
        self.critic2_optimizer = optim.Adam(
            self.networks.critic2.parameters(),
            lr=self.config['critic_lr']
        )

        # Automatic entropy tuning optimizer
        if self.config['automatic_entropy_tuning']:
            self.alpha_optimizer = optim.Adam(
                [self.networks.log_alpha],
                lr=self.config['alpha_lr']
            )
            # Adjust target entropy
            self.networks.target_entropy *= self.config['target_entropy_scale']
        else:
            self.alpha_optimizer = None
            # Fixed alpha
            self.networks.log_alpha = torch.log(torch.tensor(self.config['alpha'])).to(self.device)

        # Experience replay buffer
        self.replay_buffer = SAC_ReplayBuffer(
            capacity=self.config['buffer_size'],
            batch_size=self.config['batch_size'],
            device=self.device
        )

        # Training statistics
        self.training_step = 0
        self.episode_rewards = []
        self.losses = {
            'actor_loss': [],
            'critic1_loss': [],
            'critic2_loss': [],
            'alpha_loss': []
        }
        
        print(f"SAC v2 Agent initialized on {self.device}")
        print(f"   State space: {state_space.shape}")
        print(f"   Action space: {action_space.shape}")
        print(f"   Automatic entropy tuning: {self.config['automatic_entropy_tuning']}")
        print(f"   Target entropy: {self.networks.target_entropy}")
        print(f"   Initial alpha: {self.networks.alpha.item():.4f}")
    
    def act(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Select action"""
        if not isinstance(state, np.ndarray):
            state = np.array(state)

        # Convert to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Sample action
            action, _ = self.networks.actor.sample_action(
                state_tensor, deterministic=not training
            )
            action = action.cpu().numpy()[0]

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
        self.replay_buffer.add(state, action, reward, next_state, done)

    def train(self) -> Optional[Dict]:
        """Train one step"""
        if not self.replay_buffer.is_ready:
            return None
        
        if self.replay_buffer.total_samples < self.config['learning_starts']:
            return None
        
        if self.training_step % self.config['train_freq'] != 0:
            self.training_step += 1
            return None

        # Perform multiple gradient update steps
        total_losses = {
            'actor_loss': 0.0,
            'critic1_loss': 0.0,
            'critic2_loss': 0.0,
            'alpha_loss': 0.0
        }

        for _ in range(self.config['gradient_steps']):
            losses = self._update_networks()
            if losses:
                for key, value in losses.items():
                    total_losses[key] += value

        # Average losses
        for key in total_losses:
            total_losses[key] /= self.config['gradient_steps']
            self.losses[key].append(total_losses[key])

        self.training_step += 1

        # Return training information
        return {
            'actor_loss': total_losses['actor_loss'],
            'critic1_loss': total_losses['critic1_loss'],
            'critic2_loss': total_losses['critic2_loss'],
            'alpha_loss': total_losses['alpha_loss'],
            'alpha': self.networks.alpha.item(),
            'buffer_size': len(self.replay_buffer),
            'training_step': self.training_step
        }
    
    def _update_networks(self) -> Optional[Dict]:
        """Update network parameters"""
        # Sample batch
        batch = self.replay_buffer.sample()
        if batch is None:
            return None

        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']

        # Update Critic networks
        critic1_loss, critic2_loss = self._update_critics(
            states, actions, rewards, next_states, dones
        )

        # Update Actor network
        actor_loss = self._update_actor(states)

        # Update Alpha (if automatic entropy tuning is enabled)
        alpha_loss = 0.0
        if self.config['automatic_entropy_tuning']:
            alpha_loss = self._update_alpha(states)

        # Soft update target networks
        self.networks.soft_update_target_networks(self.config['tau'])
        
        return {
            'actor_loss': actor_loss,
            'critic1_loss': critic1_loss,
            'critic2_loss': critic2_loss,
            'alpha_loss': alpha_loss
        }
    
    def _update_critics(self, states, actions, rewards, next_states, dones):
        """Update Critic networks"""
        with torch.no_grad():
            # Compute target Q values
            next_actions, next_log_probs = self.networks.actor.sample_action(next_states)

            target_q1 = self.networks.target_critic1(next_states, next_actions)
            target_q2 = self.networks.target_critic2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.networks.alpha * next_log_probs

            target_q_values = rewards + self.config['gamma'] * (1 - dones) * target_q

        # Current Q values
        current_q1 = self.networks.critic1(states, actions)
        current_q2 = self.networks.critic2(states, actions)

        # Critic losses
        critic1_loss = F.mse_loss(current_q1, target_q_values)
        critic2_loss = F.mse_loss(current_q2, target_q_values)

        # Update Critic 1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        # Update Critic 2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        return critic1_loss.item(), critic2_loss.item()

    def _update_actor(self, states):
        """Update Actor network"""
        # Sample actions
        actions, log_probs = self.networks.actor.sample_action(states)

        # Compute Q values
        q1 = self.networks.critic1(states, actions)
        q2 = self.networks.critic2(states, actions)
        q = torch.min(q1, q2)

        # Actor loss (maximize expected reward and entropy)
        actor_loss = (self.networks.alpha * log_probs - q).mean()

        # Update Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item()

    def _update_alpha(self, states):
        """Update Alpha (entropy coefficient)"""
        with torch.no_grad():
            actions, log_probs = self.networks.actor.sample_action(states)

        # Alpha loss
        alpha_loss = -(self.networks.log_alpha * (log_probs + self.networks.target_entropy)).mean()

        # Update Alpha
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        return alpha_loss.item()

    def save(self, filepath: str):
        """Save model"""
        save_dict = {
            'actor': self.networks.actor.state_dict(),
            'critic1': self.networks.critic1.state_dict(),
            'critic2': self.networks.critic2.state_dict(),
            'target_critic1': self.networks.target_critic1.state_dict(),
            'target_critic2': self.networks.target_critic2.state_dict(),
            'log_alpha': self.networks.log_alpha,
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
            'config': self.config,
            'training_step': self.training_step
        }
        
        if self.alpha_optimizer:
            save_dict['alpha_optimizer'] = self.alpha_optimizer.state_dict()
        
        torch.save(save_dict, filepath)

    def load(self, filepath: str):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.networks.actor.load_state_dict(checkpoint['actor'])
        self.networks.critic1.load_state_dict(checkpoint['critic1'])
        self.networks.critic2.load_state_dict(checkpoint['critic2'])
        self.networks.target_critic1.load_state_dict(checkpoint['target_critic1'])
        self.networks.target_critic2.load_state_dict(checkpoint['target_critic2'])
        self.networks.log_alpha = checkpoint['log_alpha'].to(self.device)

        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer'])

        if self.alpha_optimizer and 'alpha_optimizer' in checkpoint:
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])

        self.training_step = checkpoint['training_step']

        print(f"SAC v2 model loaded from {filepath}")

    def get_stats(self) -> Dict:
        """Get training statistics"""
        buffer_stats = self.replay_buffer.get_stats()

        stats = {
            'training_step': self.training_step,
            'alpha': self.networks.alpha.item(),
            'target_entropy': self.networks.target_entropy,
            'episodes_trained': len(self.episode_rewards),
            **buffer_stats
        }

        # Add loss statistics
        for key, losses in self.losses.items():
            if losses:
                stats[f'avg_{key}'] = np.mean(losses[-100:])
        
        return stats
