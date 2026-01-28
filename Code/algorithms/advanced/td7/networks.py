"""
TD7 Neural Networks
Neural network architecture for TD7 algorithm, including Actor, Critic, and SALE encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any, Optional


class StateEncoder(nn.Module):
    """SALE state encoder - learns state representation"""

    def __init__(self,
                 state_dim: int,
                 embedding_dim: int = 256,
                 hidden_dim: int = 256):
        super(StateEncoder, self).__init__()

        self.state_dim = state_dim
        self.embedding_dim = embedding_dim

        # State encoding network
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

        # AvgL1Norm normalization (TD7 feature)
        self.normalize = True

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            state: State [batch_size, state_dim]

        Returns:
            embedding: State embedding [batch_size, embedding_dim]
        """
        embedding = self.encoder(state)

        # AvgL1Norm normalization
        if self.normalize:
            # Compute L1 norm and normalize
            l1_norm = torch.mean(torch.abs(embedding), dim=-1, keepdim=True)
            embedding = embedding / (l1_norm + 1e-8)

        return embedding


class TD7_Actor(nn.Module):
    """TD7 Actor network"""

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 embedding_dim: int = 256,
                 hidden_dim: int = 256,
                 max_action: float = 1.0):
        super(TD7_Actor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim
        self.max_action = max_action

        # Use state embedding as input
        self.policy_network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

        # Small initialization for last layer
        nn.init.uniform_(self.policy_network[-2].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.policy_network[-2].bias, -3e-3, 3e-3)

    def forward(self, state_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            state_embedding: State embedding [batch_size, embedding_dim]

        Returns:
            action: Action [batch_size, action_dim]
        """
        action = self.policy_network(state_embedding)
        return action * self.max_action


class TD7_Critic(nn.Module):
    """TD7 Critic network - dual Q-network"""

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 embedding_dim: int = 256,
                 hidden_dim: int = 256):
        super(TD7_Critic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim

        # Q-network 1 - uses state embedding + action
        self.q1_network = nn.Sequential(
            nn.Linear(embedding_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Q-network 2 - dual Q-network reduces estimation bias
        self.q2_network = nn.Sequential(
            nn.Linear(embedding_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

    def forward(self,
                state_embedding: torch.Tensor,
                action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            state_embedding: State embedding [batch_size, embedding_dim]
            action: Action [batch_size, action_dim]

        Returns:
            q1_value: Q1 value [batch_size, 1]
            q2_value: Q2 value [batch_size, 1]
        """
        q_input = torch.cat([state_embedding, action], dim=-1)

        q1_value = self.q1_network(q_input)
        q2_value = self.q2_network(q_input)

        return q1_value, q2_value

    def q1(self, state_embedding: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return only Q1 value (for Actor update)"""
        q_input = torch.cat([state_embedding, action], dim=-1)
        return self.q1_network(q_input)


class TD7_Networks:
    """TD7 network collection"""

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 embedding_dim: int = 256,
                 hidden_dim: int = 256,
                 max_action: float = 1.0,
                 device: torch.device = torch.device('cpu')):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim
        self.max_action = max_action
        self.device = device

        # State encoder
        self.state_encoder = StateEncoder(
            state_dim, embedding_dim, hidden_dim
        ).to(device)

        # Actor network
        self.actor = TD7_Actor(
            state_dim, action_dim, embedding_dim, hidden_dim, max_action
        ).to(device)

        # Critic network
        self.critic = TD7_Critic(
            state_dim, action_dim, embedding_dim, hidden_dim
        ).to(device)

        # Target networks
        self.target_state_encoder = StateEncoder(
            state_dim, embedding_dim, hidden_dim
        ).to(device)

        self.target_actor = TD7_Actor(
            state_dim, action_dim, embedding_dim, hidden_dim, max_action
        ).to(device)

        self.target_critic = TD7_Critic(
            state_dim, action_dim, embedding_dim, hidden_dim
        ).to(device)

        # Initialize target networks
        self.soft_update_target_networks(tau=1.0)

        print(f"🎯 TD7 Networks initialized")
        print(f"   State dim: {state_dim}, Action dim: {action_dim}")
        print(f"   Embedding dim: {embedding_dim}, Hidden dim: {hidden_dim}")
        print(f"   Max action: {max_action}")

    def soft_update_target_networks(self, tau: float = 0.005):
        """Soft update target networks"""
        # Update state encoder
        for target_param, param in zip(self.target_state_encoder.parameters(),
                                     self.state_encoder.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        # Update Actor
        for target_param, param in zip(self.target_actor.parameters(),
                                     self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        # Update Critic
        for target_param, param in zip(self.target_critic.parameters(),
                                     self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def create_td7_networks(state_space, action_space, network_config: Dict[str, Any] = None):
    """
    Factory function: create TD7 networks

    Args:
        state_space: State space
        action_space: Action space
        network_config: Network configuration

    Returns:
        TD7 network collection
    """
    default_config = {
        'embedding_dim': 256,
        'hidden_dim': 256,
        'max_action': 1.0
    }

    if network_config:
        default_config.update(network_config)

    # Get state and action dimensions
    if len(state_space.shape) == 1:
        state_dim = state_space.shape[0]
    else:
        raise ValueError(f"TD7 only supports 1D state space, got {state_space.shape}")

    if len(action_space.shape) == 1:
        action_dim = action_space.shape[0]
        max_action = float(action_space.high[0])  # Assume all dimensions are the same
    else:
        raise ValueError(f"TD7 only supports 1D action space, got {action_space.shape}")

    return TD7_Networks(
        state_dim=state_dim,
        action_dim=action_dim,
        embedding_dim=default_config['embedding_dim'],
        hidden_dim=default_config['hidden_dim'],
        max_action=max_action
    )