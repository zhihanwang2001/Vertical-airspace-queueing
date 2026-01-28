"""
TD7 Agent Implementation
Integrates SALE representation learning, prioritized replay, and checkpoint mechanism
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List, Any
import random
import copy

from .networks import create_td7_networks
from .replay_buffer import TD7_PrioritizedReplayBuffer


class TD7_Agent:
    """TD7 Agent"""

    def __init__(self,
                 state_space,
                 action_space,
                 config: Dict = None):
        """
        Initialize TD7 agent

        Args:
            state_space: State space
            action_space: Action space
            config: Configuration parameters
        """

        # Default configuration
        default_config = {
            # Network configuration
            'embedding_dim': 256,
            'hidden_dim': 256,
            'max_action': 1.0,

            # Learning parameters
            'actor_lr': 3e-4,
            'critic_lr': 3e-4,
            'encoder_lr': 3e-4,
            'gamma': 0.99,
            'tau': 0.005,

            # Learning rate scheduling (prevent late-stage collapse)
            'use_lr_schedule': True,        # Whether to use learning rate scheduling
            'warmup_steps': 75000,          # Keep fixed lr for first 75k steps (stability period)
            'total_steps': 500000,          # Total training steps
            'min_lr_ratio': 0.1,            # Minimum learning rate ratio (final_lr = initial_lr * 0.1)

            # TD3-specific parameters
            'policy_delay': 2,      # Delayed policy update
            'target_noise': 0.2,    # Target smoothing noise
            'noise_clip': 0.5,      # Noise clipping
            'exploration_noise': 0.1, # Exploration noise

            # Prioritized replay
            'buffer_size': 1000000,
            'batch_size': 256,
            'alpha': 0.6,           # Priority exponent
            'beta': 0.4,            # Importance sampling exponent
            'beta_increment': 0.001,

            # SALE-specific parameters
            'embedding_loss_weight': 1.0,  # Embedding loss weight
            'embedding_update_freq': 1,     # Encoder update frequency

            # Checkpoint mechanism
            'use_checkpoints': True,
            'checkpoint_freq': 10000,       # Checkpoint save frequency
            'max_checkpoints': 5,           # Maximum number of checkpoints

            # Training parameters
            'learning_starts': 25000,
            'train_freq': 1,

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
            'embedding_dim': self.config['embedding_dim'],
            'hidden_dim': self.config['hidden_dim'],
            'max_action': self.config['max_action']
        }

        self.networks = create_td7_networks(
            state_space, action_space, network_config
        )
        self.networks.device = self.device

        # Move to correct device
        for network in [self.networks.state_encoder, self.networks.actor, self.networks.critic,
                       self.networks.target_state_encoder, self.networks.target_actor,
                       self.networks.target_critic]:
            network.to(self.device)

        # Optimizers
        self.encoder_optimizer = optim.Adam(
            self.networks.state_encoder.parameters(),
            lr=self.config['encoder_lr']
        )

        self.actor_optimizer = optim.Adam(
            self.networks.actor.parameters(),
            lr=self.config['actor_lr']
        )

        self.critic_optimizer = optim.Adam(
            self.networks.critic.parameters(),
            lr=self.config['critic_lr']
        )

        # Add learning rate schedulers (cosine annealing, prevent late-stage collapse)
        if self.config['use_lr_schedule']:
            from torch.optim.lr_scheduler import LambdaLR

            def lr_lambda(step):
                """
                Delayed cosine annealing learning rate schedule
                - First 75k steps: Keep fixed learning rate 1.0
                - After 75k steps: Cosine annealing to 0.1
                """
                warmup_steps = self.config['warmup_steps']
                total_steps = self.config['total_steps']
                min_ratio = self.config['min_lr_ratio']

                if step < warmup_steps:
                    return 1.0  # Fixed learning rate
                else:
                    # Cosine annealing
                    progress = (step - warmup_steps) / (total_steps - warmup_steps)
                    progress = min(progress, 1.0)
                    cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
                    return min_ratio + (1.0 - min_ratio) * cosine_factor

            self.encoder_scheduler = LambdaLR(self.encoder_optimizer, lr_lambda)
            self.actor_scheduler = LambdaLR(self.actor_optimizer, lr_lambda)
            self.critic_scheduler = LambdaLR(self.critic_optimizer, lr_lambda)
        else:
            self.encoder_scheduler = None
            self.actor_scheduler = None
            self.critic_scheduler = None

        # Prioritized experience replay buffer
        self.replay_buffer = TD7_PrioritizedReplayBuffer(
            capacity=self.config['buffer_size'],
            batch_size=self.config['batch_size'],
            alpha=self.config['alpha'],
            beta=self.config['beta'],
            beta_increment=self.config['beta_increment'],
            device=self.device
        )

        # Training statistics
        self.training_step = 0
        self.episode_rewards = []
        self.losses = {
            'actor_loss': [],
            'critic_loss': [],
            'encoder_loss': []
        }

        # Checkpoint management
        self.checkpoints = []
        
        print(f"🎯 TD7 Agent initialized on {self.device}")
        print(f"   State space: {state_space.shape}")
        print(f"   Action space: {action_space.shape}")
        print(f"   Embedding dim: {self.config['embedding_dim']}")
        print(f"   Use checkpoints: {self.config['use_checkpoints']}")
        print(f"   Policy delay: {self.config['policy_delay']}")
    
    def act(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """Select action"""
        if not isinstance(state, np.ndarray):
            state = np.array(state)

        # Convert to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Get state embedding
            state_embedding = self.networks.state_encoder(state_tensor)

            # Get action
            action = self.networks.actor(state_embedding).cpu().numpy()[0]

            # Add exploration noise
            if training:
                noise = np.random.normal(
                    0, self.config['exploration_noise'], size=action.shape
                )
                action += noise

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
        
        if self.replay_buffer.tree.n_entries < self.config['learning_starts']:
            return None
        
        if self.training_step % self.config['train_freq'] != 0:
            self.training_step += 1
            return None

        # Sample batch
        batch = self.replay_buffer.sample()
        if batch is None:
            self.training_step += 1
            return None

        # Update networks
        losses = self._update_networks(batch)

        # Checkpoint management
        if (self.config['use_checkpoints'] and
            self.training_step % self.config['checkpoint_freq'] == 0):
            self._save_checkpoint()

        # Soft update target networks
        self.networks.soft_update_target_networks(self.config['tau'])

        self.training_step += 1

        # Store losses
        for key, value in losses.items():
            self.losses[key].append(value)

        # Return training information
        return {
            'actor_loss': losses['actor_loss'],
            'critic_loss': losses['critic_loss'],
            'encoder_loss': losses['encoder_loss'],
            'buffer_size': len(self.replay_buffer),
            'training_step': self.training_step,
            'max_priority': self.replay_buffer.max_priority
        }
    
    def _update_networks(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Update network parameters"""
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        is_weights = batch['is_weights']
        indices = batch['indices']

        # Get state embeddings
        state_embeddings = self.networks.state_encoder(states)
        next_state_embeddings = self.networks.target_state_encoder(next_states)

        # Update Critic network
        critic_loss, td_errors = self._update_critic(
            state_embeddings, actions, rewards, next_state_embeddings, dones, is_weights
        )

        # Update priorities
        priorities = self.replay_buffer.compute_lap_priority(td_errors, actions)
        self.replay_buffer.update_priorities(indices, priorities)

        # Delayed policy update
        actor_loss = 0.0
        encoder_loss = 0.0

        if self.training_step % self.config['policy_delay'] == 0:
            # Update Actor network
            actor_loss = self._update_actor(state_embeddings.detach())

            # Update state encoder
            if self.training_step % self.config['embedding_update_freq'] == 0:
                encoder_loss = self._update_encoder(states, actions)
        
        return {
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'encoder_loss': encoder_loss
        }
    
    def _update_critic(self, state_embeddings, actions, rewards, next_state_embeddings, dones, is_weights):
        """Update Critic network"""
        with torch.no_grad():
            # Compute target actions (with noise)
            target_actions = self.networks.target_actor(next_state_embeddings)

            # Add target smoothing noise
            noise = torch.clamp(
                torch.randn_like(target_actions) * self.config['target_noise'],
                -self.config['noise_clip'], self.config['noise_clip']
            )
            target_actions = torch.clamp(
                target_actions + noise,
                -self.networks.max_action, self.networks.max_action
            )

            # Compute target Q-values (take minimum to reduce overestimation)
            target_q1, target_q2 = self.networks.target_critic(next_state_embeddings, target_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + self.config['gamma'] * (1 - dones) * target_q

        # Current Q-values
        current_q1, current_q2 = self.networks.critic(state_embeddings, actions)

        # TD errors
        td_error1 = target_q - current_q1
        td_error2 = target_q - current_q2

        # Huber loss (TD7 feature)
        critic_loss1 = F.smooth_l1_loss(current_q1, target_q, reduction='none')
        critic_loss2 = F.smooth_l1_loss(current_q2, target_q, reduction='none')

        # Importance sampling weighting
        critic_loss1 = (critic_loss1 * is_weights).mean()
        critic_loss2 = (critic_loss2 * is_weights).mean()

        critic_loss = critic_loss1 + critic_loss2

        # Update Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update critic learning rate schedule
        if self.critic_scheduler is not None:
            self.critic_scheduler.step()

        # Return TD errors (for priority update)
        td_errors = torch.max(torch.abs(td_error1), torch.abs(td_error2)).detach()

        return critic_loss.item(), td_errors
    
    def _update_actor(self, state_embeddings):
        """Update Actor network"""
        # Compute policy loss
        actions = self.networks.actor(state_embeddings)
        actor_loss = -self.networks.critic.q1(state_embeddings, actions).mean()

        # Update Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update actor learning rate schedule
        if self.actor_scheduler is not None:
            self.actor_scheduler.step()

        return actor_loss.item()
    
    def _update_encoder(self, states, actions):
        """Update state encoder (SALE mechanism)"""
        # Recompute embeddings
        state_embeddings = self.networks.state_encoder(states)

        # SALE loss: maximize state-action interaction
        # Using simplified SALE loss here
        pred_actions = self.networks.actor(state_embeddings)
        embedding_loss = F.mse_loss(pred_actions, actions)

        # Embedding regularization
        embedding_reg = torch.mean(torch.norm(state_embeddings, dim=-1))

        total_encoder_loss = (
            self.config['embedding_loss_weight'] * embedding_loss +
            0.01 * embedding_reg
        )

        # Update encoder
        self.encoder_optimizer.zero_grad()
        total_encoder_loss.backward()
        self.encoder_optimizer.step()

        # Update encoder learning rate schedule
        if self.encoder_scheduler is not None:
            self.encoder_scheduler.step()

        return total_encoder_loss.item()

    def _save_checkpoint(self):
        """Save checkpoint"""
        checkpoint = {
            'state_encoder': self.networks.state_encoder.state_dict(),
            'actor': self.networks.actor.state_dict(),
            'critic': self.networks.critic.state_dict(),
            'training_step': self.training_step,
            'losses': copy.deepcopy(self.losses)
        }

        self.checkpoints.append(checkpoint)

        # Maintain maximum checkpoint count
        if len(self.checkpoints) > self.config['max_checkpoints']:
            self.checkpoints.pop(0)

        print(f"💾 Checkpoint saved at step {self.training_step} (total: {len(self.checkpoints)})")

    def load_best_checkpoint(self):
        """Load best checkpoint"""
        if not self.checkpoints:
            print("⚠️ No checkpoints available")
            return

        # Simple strategy: load the latest checkpoint
        best_checkpoint = self.checkpoints[-1]

        self.networks.state_encoder.load_state_dict(best_checkpoint['state_encoder'])
        self.networks.actor.load_state_dict(best_checkpoint['actor'])
        self.networks.critic.load_state_dict(best_checkpoint['critic'])

        print(f"✅ Loaded checkpoint from step {best_checkpoint['training_step']}")

    def save(self, filepath: str):
        """Save model"""
        save_dict = {
            'state_encoder': self.networks.state_encoder.state_dict(),
            'actor': self.networks.actor.state_dict(),
            'critic': self.networks.critic.state_dict(),
            'target_state_encoder': self.networks.target_state_encoder.state_dict(),
            'target_actor': self.networks.target_actor.state_dict(),
            'target_critic': self.networks.target_critic.state_dict(),
            'encoder_optimizer': self.encoder_optimizer.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'config': self.config,
            'training_step': self.training_step,
            'checkpoints': self.checkpoints
        }

        torch.save(save_dict, filepath)

    def load(self, filepath: str):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        self.networks.state_encoder.load_state_dict(checkpoint['state_encoder'])
        self.networks.actor.load_state_dict(checkpoint['actor'])
        self.networks.critic.load_state_dict(checkpoint['critic'])
        self.networks.target_state_encoder.load_state_dict(checkpoint['target_state_encoder'])
        self.networks.target_actor.load_state_dict(checkpoint['target_actor'])
        self.networks.target_critic.load_state_dict(checkpoint['target_critic'])

        self.encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

        self.training_step = checkpoint['training_step']

        if 'checkpoints' in checkpoint:
            self.checkpoints = checkpoint['checkpoints']

        print(f"✅ TD7 model loaded from {filepath}")

    def get_stats(self) -> Dict:
        """Get training statistics"""
        buffer_stats = self.replay_buffer.get_stats()

        stats = {
            'training_step': self.training_step,
            'episodes_trained': len(self.episode_rewards),
            'num_checkpoints': len(self.checkpoints),
            **buffer_stats
        }

        # Add loss statistics
        for key, losses in self.losses.items():
            if losses:
                stats[f'avg_{key}'] = np.mean(losses[-100:])

        return stats