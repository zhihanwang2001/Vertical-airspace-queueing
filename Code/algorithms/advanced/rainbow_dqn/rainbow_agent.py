"""
Rainbow DQN Agent Implementation
Integrates all Rainbow DQN components:
1. Double DQN
2. Prioritized Experience Replay
3. Dueling Networks  
4. Multi-step Learning
5. Distributional RL (C51)
6. Noisy Networks
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict, List, Optional
import random
from collections import deque

from .networks import DuelingNoisyNetwork, create_rainbow_network
from .prioritized_replay import PrioritizedReplayBuffer, batch_to_tensors
from .distributional_loss import DistributionalLoss


class RainbowDQNAgent:
    """Rainbow DQN Agent"""
    
    def __init__(self,
                 state_space,
                 action_space,
                 config: Dict = None):
        """
        Initialize Rainbow DQN Agent
        
        Args:
            state_space: State space
            action_space: Action space
            config: Configuration parameters
        """
        
        # Optimized configuration - based on standard Rainbow DQN implementation
        default_config = {
            # Network configuration
            'hidden_dim': 512,
            'num_atoms': 51,
            'v_min': -15.0,  # Adapted to reward range of vertical stratified queue
            'v_max': 15.0,
            'noisy_std': 0.5,
            
            # Learning parameters - fixed critical hyperparameters
            'learning_rate': 1e-4,  # 🔧 Fix: 6.25e-5 → 1e-4 (standard Rainbow learning rate)
            'gamma': 0.99,
            'target_update_freq': 2000,  # 🔧 Fix: 8000 → 2000 (standard Rainbow update frequency)
            'gradient_clip': 10.0,
            
            # Prioritized replay - optimized buffer size
            'buffer_size': 200000,  # 🔧 Fix: 1M → 200k (reduce stale experiences)
            'alpha': 0.5,
            'beta': 0.4,
            'beta_increment': 0.001,
            'epsilon': 1e-6,
            
            # Multi-step learning - enhanced long-term dependencies
            'n_step': 10,  # 🔧 Fix: 3 → 10 (moderate multi-step, capture long-term dependencies)
            
            # Training parameters - early learning opportunities
            'batch_size': 32,
            'learning_starts': 5000,  # 🔧 Fix: 50000 → 5000 (start learning early)
            'train_freq': 4,
            
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
        
        # Handle continuous action space
        if hasattr(action_space, 'n'):
            # Discrete action space
            self.num_actions = action_space.n
            self.action_type = 'discrete'
        else:
            # Continuous action space - discretize
            self.action_dim = action_space.shape[0]
            self.action_low = action_space.low
            self.action_high = action_space.high
            # Create discretization bins for each action dimension
            self.action_bins = 2  # 2 discrete values per dimension
            self.num_actions = self.action_bins ** self.action_dim
            self.action_type = 'continuous'
            
            # Create discretization mapping
            self._create_action_mapping()
        
        # Create networks
        network_config = {
            'hidden_dim': self.config['hidden_dim'],
            'num_atoms': self.config['num_atoms'],
            'v_min': self.config['v_min'],
            'v_max': self.config['v_max']
        }
        
        # If continuous action space, add action_bins parameter
        if self.action_type == 'continuous':
            network_config['action_bins'] = self.action_bins
        
        self.q_network = create_rainbow_network(
            state_space, action_space, network_config
        ).to(self.device)
        
        self.target_network = create_rainbow_network(
            state_space, action_space, network_config
        ).to(self.device)
        
        # Synchronize target network
        self.update_target_network()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=self.config['learning_rate']
        )
        
        # Experience replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=self.config['buffer_size'],
            alpha=self.config['alpha'],
            beta=self.config['beta'],
            beta_increment=self.config['beta_increment'],
            epsilon=self.config['epsilon']
        )
        
        # Distributional loss function
        self.loss_fn = DistributionalLoss(
            num_atoms=self.config['num_atoms'],
            v_min=self.config['v_min'],
            v_max=self.config['v_max'],
            gamma=self.config['gamma']
        )
        
        # Multi-step learning buffer
        self.n_step_buffer = deque(maxlen=self.config['n_step'])
        
        # Training statistics
        self.training_step = 0
        self.episode_rewards = []
        self.losses = []
        
        print(f"🌈 Rainbow DQN Agent initialized on {self.device}")
        print(f"   State space: {state_space.shape}")
        if self.action_type == 'discrete':
            print(f"   Action space: {self.num_actions} (discrete)")
        else:
            print(f"   Action space: {self.action_dim}D continuous -> {self.num_actions} discrete")
        print(f"   Network atoms: {self.config['num_atoms']}")
        print(f"   Value range: [{self.config['v_min']}, {self.config['v_max']}]")
    
    def _create_action_mapping(self):
        """Create discretization mapping for continuous action space"""
        if self.action_type == 'discrete':
            return
        
        # Create discrete values for each action dimension (use moderate values to avoid extreme actions)
        self.action_grids = []
        for i in range(self.action_dim):
            # Don't use extreme values (-1,1), use more moderate range (-0.5,0.5)
            grid = np.linspace(-0.5, 0.5, self.action_bins)
            self.action_grids.append(grid)
        
        print(f"   Action discretization: {self.action_bins}^{self.action_dim} = {self.num_actions} discrete actions")
    
    def _discrete_to_continuous_action(self, discrete_action: int) -> np.ndarray:
        """Convert discrete action to continuous action"""
        if self.action_type == 'discrete':
            return discrete_action
        
        # Convert discrete action index to multi-dimensional coordinates
        continuous_action = np.zeros(self.action_dim)
        remaining = discrete_action
        
        for i in range(self.action_dim):
            idx = remaining % self.action_bins
            continuous_action[i] = self.action_grids[i][idx]
            remaining //= self.action_bins
        
        return continuous_action
    
    def act(self, state: np.ndarray, training: bool = True):
        """Select action"""
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        
        # Convert to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get Q distribution
            q_dist = self.q_network(state_tensor)
            
            # Calculate Q values (expectation of distribution)
            q_values = self.loss_fn.q_values_from_distribution(q_dist)
            
            # Select best action (greedy)
            discrete_action = q_values.argmax(dim=1).item()
        
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
        """Store transition to buffer"""
        
        # If continuous action, need to find corresponding discrete action index
        if self.action_type == 'continuous':
            if isinstance(action, np.ndarray):
                # Convert continuous action to discrete action index
                discrete_action = 0
                multiplier = 1
                
                for i in range(self.action_dim):
                    # Find closest grid point
                    closest_idx = np.argmin(np.abs(self.action_grids[i] - action[i]))
                    discrete_action += closest_idx * multiplier
                    multiplier *= self.action_bins
            else:
                discrete_action = action
        else:
            discrete_action = action
        
        # Add to n-step buffer
        self.n_step_buffer.append((state, discrete_action, reward, next_state, done))
        
        # If n-step buffer is full, calculate n-step return
        if len(self.n_step_buffer) == self.config['n_step']:
            # Calculate n-step reward
            n_step_reward = 0.0
            gamma = 1.0
            
            for i in range(self.config['n_step']):
                n_step_reward += gamma * self.n_step_buffer[i][2]
                gamma *= self.config['gamma']
                if self.n_step_buffer[i][4]:  # If done
                    break
            
            # Get initial state and final state
            initial_state = self.n_step_buffer[0][0]
            initial_action = self.n_step_buffer[0][1]
            final_next_state = self.n_step_buffer[-1][3]
            final_done = any(exp[4] for exp in self.n_step_buffer)
            
            # Store n-step experience
            self.replay_buffer.add(
                initial_state, initial_action, n_step_reward, 
                final_next_state, final_done
            )
    
    def train(self) -> Optional[Dict]:
        """Train one step"""
        if not self.replay_buffer.is_ready:
            return None
        
        if self.training_step % self.config['train_freq'] != 0:
            self.training_step += 1
            return None
        
        # Sample experiences
        batch, weights, indices = self.replay_buffer.sample(self.config['batch_size'])
        
        # Convert to tensors
        states, actions, rewards, next_states, dones = batch_to_tensors(batch, self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Reset noise
        self.q_network.reset_noise()
        self.target_network.reset_noise()
        
        # Current Q distribution
        current_q_dist = self.q_network(states)
        
        # Target network's next state Q distribution
        with torch.no_grad():
            next_q_dist = self.target_network(next_states)
            
            # Double DQN: use current network to select action
            next_q_values = self.loss_fn.q_values_from_distribution(
                self.q_network(next_states)
            )
            next_actions = next_q_values.argmax(dim=1)
        
        # Calculate loss
        loss, td_errors = self.loss_fn.compute_loss(
            current_q_dist, actions, rewards, next_q_dist, next_actions, dones, weights
        )
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.q_network.parameters(), 
            self.config['gradient_clip']
        )
        
        self.optimizer.step()
        
        # Update priorities
        self.replay_buffer.update_priorities(indices, td_errors)
        self.replay_buffer.update_beta()
        
        # Update target network
        if self.training_step % self.config['target_update_freq'] == 0:
            self.update_target_network()
        
        self.training_step += 1
        self.losses.append(loss.item())
        
        return {
            'loss': loss.item(),
            'td_error_mean': np.mean(td_errors),
            'beta': self.replay_buffer.beta,
            'buffer_size': len(self.replay_buffer)
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
        
        print(f"✅ Rainbow DQN model loaded from {filepath}")
    
    def get_stats(self) -> Dict:
        """Get training statistics"""
        return {
            'training_step': self.training_step,
            'buffer_size': len(self.replay_buffer),
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0,
            'beta': self.replay_buffer.beta,
            'episodes_trained': len(self.episode_rewards)
        }
    
    def reset_noise(self):
        """Reset network noise"""
        self.q_network.reset_noise()
        self.target_network.reset_noise()
