"""
IMPALA Neural Networks
Neural network architecture for IMPALA algorithm, including Actor-Critic network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any


class IMPALANetwork(nn.Module):
    """IMPALA Actor-Critic network"""

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 512,
                 num_layers: int = 2):
        super(IMPALANetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Shared feature extraction layers
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        self.shared_layers = nn.Sequential(*layers)

        # Actor head: outputs action mean and standard deviation (continuous actions)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_logstd = nn.Linear(hidden_dim, action_dim)

        # Critic head: outputs state value
        self.critic = nn.Linear(hidden_dim, 1)

        # Initialize parameters
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

        # Actor output layer uses small initialization values
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.actor_logstd.weight, gain=0.01)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            state: State tensor [batch_size, state_dim]

        Returns:
            action_mean: Action mean [batch_size, action_dim]
            action_logstd: Action log standard deviation [batch_size, action_dim]
            value: State value [batch_size, 1]
        """
        # Shared feature extraction
        features = self.shared_layers(state)

        # Actor output
        action_mean = self.actor_mean(features)
        action_logstd = self.actor_logstd(features)
        # Limit log_std range
        action_logstd = torch.clamp(action_logstd, -10, 2)

        # Critic output
        value = self.critic(features)

        return action_mean, action_logstd, value

    def get_action_and_value(self, state: torch.Tensor, deterministic: bool = False):
        """
        Get action and value

        Args:
            state: State tensor
            deterministic: Whether to use deterministic policy

        Returns:
            action: Sampled action
            log_prob: Log probability of action
            value: State value
        """
        action_mean, action_logstd, value = self.forward(state)

        if deterministic:
            action = action_mean
            log_prob = torch.zeros_like(action_mean).sum(dim=-1, keepdim=True)
        else:
            # Create normal distribution
            std = torch.exp(action_logstd)
            dist = torch.distributions.Normal(action_mean, std)

            # Sample action
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)

        return action, log_prob, value

    def evaluate_action(self, state: torch.Tensor, action: torch.Tensor):
        """
        Evaluate value and probability of given state and action

        Args:
            state: State tensor
            action: Action tensor

        Returns:
            log_prob: Log probability of action
            value: State value
            entropy: Policy entropy
        """
        action_mean, action_logstd, value = self.forward(state)

        # Create distribution
        std = torch.exp(action_logstd)
        dist = torch.distributions.Normal(action_mean, std)

        # Compute log probability
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)

        # Compute entropy
        entropy = dist.entropy().sum(dim=-1, keepdim=True)

        return log_prob, value, entropy


def create_impala_network(state_space, action_space, network_config: Dict[str, Any] = None):
    """
    Factory function: create IMPALA network

    Args:
        state_space: State space
        action_space: Action space
        network_config: Network configuration

    Returns:
        IMPALA network instance
    """
    default_config = {
        'hidden_dim': 512,
        'num_layers': 2
    }

    if network_config:
        default_config.update(network_config)

    # Get state and action dimensions
    if len(state_space.shape) == 1:
        state_dim = state_space.shape[0]
    else:
        raise ValueError(f"Unsupported state space shape: {state_space.shape}")

    if hasattr(action_space, 'n'):
        raise ValueError("IMPALA currently only supports continuous action spaces")
    else:
        action_dim = action_space.shape[0]

    return IMPALANetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        **default_config
    )
