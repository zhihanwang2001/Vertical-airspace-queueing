"""
SAC v2 Neural Networks
Neural network architecture for SAC v2 algorithm, including Actor and Critic networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any, Optional
from torch.distributions import Normal


class ActorNetwork(nn.Module):
    """SAC Actor network - stochastic policy network"""

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 max_action: float = 1.0,
                 log_std_min: float = -20,
                 log_std_max: float = 2):
        super(ActorNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Shared feature layers
        self.feature_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Mean and log std outputs
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier initialization
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)

        # Special initialization for last layer
        nn.init.uniform_(self.mean_layer.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.mean_layer.bias, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_layer.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_layer.bias, -3e-3, 3e-3)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            state: State [batch_size, state_dim]

        Returns:
            mean: Action mean [batch_size, action_dim]
            log_std: Action log std [batch_size, action_dim]
        """
        features = self.feature_layers(state)
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)

        # Limit log std range
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def sample_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action

        Args:
            state: State
            deterministic: Whether to use deterministic sampling

        Returns:
            action: Action
            log_prob: Log probability
        """
        mean, log_std = self.forward(state)

        if deterministic:
            # Deterministic action
            action = torch.tanh(mean) * self.max_action
            log_prob = torch.zeros_like(action).sum(dim=-1, keepdim=True)
        else:
            # Stochastic sampling
            std = torch.exp(log_std)
            normal = Normal(mean, std)

            # Reparameterization sampling
            x = normal.rsample()  # Use rsample to support gradient propagation
            action = torch.tanh(x) * self.max_action

            # Compute log probability (considering tanh transformation)
            log_prob = normal.log_prob(x)
            # Correct for tanh transformation jacobian
            log_prob -= torch.log(self.max_action * (1 - torch.tanh(x).pow(2)) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def evaluate_action(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy of action

        Args:
            state: State
            action: Action

        Returns:
            log_prob: Log probability
            entropy: Entropy
        """
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        normal = Normal(mean, std)

        # Inverse tanh transformation
        action_scaled = action / self.max_action
        x = 0.5 * torch.log((1 + action_scaled + 1e-6) / (1 - action_scaled + 1e-6))

        # Compute log probability
        log_prob = normal.log_prob(x)
        # Correct for tanh transformation jacobian
        log_prob -= torch.log(self.max_action * (1 - action_scaled.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        # Compute entropy
        entropy = normal.entropy().sum(dim=-1, keepdim=True)

        return log_prob, entropy


class CriticNetwork(nn.Module):
    """SAC Critic network - Q network"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(CriticNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Q network
        self.q_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            state: State [batch_size, state_dim]
            action: Action [batch_size, action_dim]

        Returns:
            q_value: Q value [batch_size, 1]
        """
        q_input = torch.cat([state, action], dim=-1)
        q_value = self.q_network(q_input)
        return q_value


class SAC_v2_Networks:
    """SAC v2 network collection"""

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 max_action: float = 1.0,
                 device: torch.device = torch.device('cpu')):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.device = device

        # Actor network
        self.actor = ActorNetwork(
            state_dim, action_dim, hidden_dim, max_action
        ).to(device)

        # Two Critic networks (reduce estimation bias)
        self.critic1 = CriticNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic2 = CriticNetwork(state_dim, action_dim, hidden_dim).to(device)

        # Target Critic networks (soft update)
        self.target_critic1 = CriticNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_critic2 = CriticNetwork(state_dim, action_dim, hidden_dim).to(device)

        # Initialize target networks
        self.soft_update_target_networks(tau=1.0)

        # Automatic entropy tuning
        self.target_entropy = -action_dim  # Heuristic target entropy
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)

        print(f"SAC v2 Networks initialized")
        print(f"   State dim: {state_dim}, Action dim: {action_dim}")
        print(f"   Hidden dim: {hidden_dim}, Max action: {max_action}")
        print(f"   Target entropy: {self.target_entropy}")

    @property
    def alpha(self):
        """Get current entropy coefficient"""
        return self.log_alpha.exp()

    def soft_update_target_networks(self, tau: float = 0.005):
        """Soft update target networks"""
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def create_sac_v2_networks(state_space, action_space, network_config: Dict[str, Any] = None):
    """
    Factory function: create SAC v2 networks

    Args:
        state_space: State space
        action_space: Action space
        network_config: Network configuration

    Returns:
        SAC v2 network collection
    """
    default_config = {
        'hidden_dim': 256,
        'max_action': 1.0
    }

    if network_config:
        default_config.update(network_config)

    # Get state and action dimensions
    if len(state_space.shape) == 1:
        state_dim = state_space.shape[0]
    else:
        raise ValueError(f"SAC v2 only supports 1D state space, got {state_space.shape}")

    if len(action_space.shape) == 1:
        action_dim = action_space.shape[0]
        max_action = float(action_space.high[0])  # Assume all dimensions are the same
    else:
        raise ValueError(f"SAC v2 only supports 1D action space, got {action_space.shape}")

    return SAC_v2_Networks(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=default_config['hidden_dim'],
        max_action=max_action
    )
