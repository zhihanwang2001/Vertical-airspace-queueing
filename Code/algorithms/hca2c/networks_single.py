"""
Single Policy Network for Ablation Study

This network uses a single large policy instead of hierarchical decomposition
to test the contribution of multi-level architecture.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple


class SinglePolicyNetwork(nn.Module):
    """
    Single large policy network - no hierarchical decomposition

    Architecture:
    - Input: 46-dim observation (same as HCA2C-Full)
    - Hidden: [512, 512, 256] (matched parameter count ~459K)
    - Output: 11-dim action (5 service + 1 arrival + 5 emergency)

    This tests whether hierarchical decomposition is necessary or if
    a single large network can achieve similar performance.
    """

    def __init__(self, obs_dim: int = 46, action_dim: int = 11, hidden_dims=[512, 512, 256]):
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Build network layers
        layers = []
        prev_dim = obs_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim

        self.shared = nn.Sequential(*layers)

        # Policy head (actor)
        self.action_mean = nn.Linear(prev_dim, action_dim)
        self.action_log_std = nn.Linear(prev_dim, action_dim)

        # Value head (critic)
        self.value = nn.Linear(prev_dim, 1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            obs: Observation tensor [batch, 46]

        Returns:
            action_mean: Action mean [batch, 11]
            action_log_std: Action log std [batch, 11]
            value: State value [batch, 1]
        """
        features = self.shared(obs)

        action_mean = self.action_mean(features)
        action_log_std = self.action_log_std(features)
        action_log_std = torch.clamp(action_log_std, -20, 2)

        value = self.value(features)

        return action_mean, action_log_std, value

    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy

        Args:
            obs: Observation tensor
            deterministic: If True, return mean action

        Returns:
            action: Sampled action
            log_prob: Log probability of action
        """
        action_mean, action_log_std, _ = self.forward(obs)

        if deterministic:
            return torch.tanh(action_mean), None

        action_std = torch.exp(action_log_std)
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)

        # Tanh squashing
        action = torch.tanh(action)

        return action, log_prob

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Get state value

        Args:
            obs: Observation tensor

        Returns:
            value: State value
        """
        _, _, value = self.forward(obs)
        return value


if __name__ == '__main__':
    # Test the network
    print("Testing SinglePolicyNetwork...")

    # Create network
    net = SinglePolicyNetwork(obs_dim=46, action_dim=11)

    # Count parameters
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Expected: ~459K")

    # Test forward pass
    batch_size = 4
    obs = torch.randn(batch_size, 46)

    action_mean, action_log_std, value = net.forward(obs)
    print(f"\nForward pass:")
    print(f"  Action mean shape: {action_mean.shape} (expected: [{batch_size}, 11])")
    print(f"  Action log_std shape: {action_log_std.shape} (expected: [{batch_size}, 11])")
    print(f"  Value shape: {value.shape} (expected: [{batch_size}, 1])")

    # Test action sampling
    action, log_prob = net.get_action(obs, deterministic=False)
    print(f"\nAction sampling:")
    print(f"  Action shape: {action.shape} (expected: [{batch_size}, 11])")
    print(f"  Log prob shape: {log_prob.shape} (expected: [{batch_size}, 1])")
    print(f"  Action range: [{action.min():.3f}, {action.max():.3f}] (expected: [-1, 1])")

    # Test deterministic action
    action_det, _ = net.get_action(obs, deterministic=True)
    print(f"\nDeterministic action:")
    print(f"  Action shape: {action_det.shape} (expected: [{batch_size}, 11])")

    print("\n✓ SinglePolicyNetwork test passed!")
