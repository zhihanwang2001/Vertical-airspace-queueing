"""
Distributional Loss for Rainbow DQN (C51 algorithm)
Implementation of distributional reinforcement learning loss function
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class DistributionalLoss:
    """C51 distributional loss function"""

    def __init__(self,
                 num_atoms: int = 51,
                 v_min: float = -10,
                 v_max: float = 10,
                 gamma: float = 0.99):
        """
        Args:
            num_atoms: Number of atoms in value distribution
            v_min: Minimum value of distribution
            v_max: Maximum value of distribution
            gamma: Discount factor
        """
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.gamma = gamma
        self.delta_z = (v_max - v_min) / (num_atoms - 1)

        # Support points
        self.supports = torch.linspace(v_min, v_max, num_atoms)

    def compute_loss(self,
                     q_dist: torch.Tensor,
                     actions: torch.Tensor,
                     rewards: torch.Tensor,
                     next_q_dist: torch.Tensor,
                     next_actions: torch.Tensor,
                     dones: torch.Tensor,
                     weights: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute distributional loss

        Args:
            q_dist: Q distribution of current state [batch_size, action_dim, num_atoms]
            actions: Selected actions [batch_size]
            rewards: Immediate rewards [batch_size]
            next_q_dist: Q distribution of next state [batch_size, action_dim, num_atoms]
            next_actions: Next state actions (Double DQN) [batch_size]
            dones: Whether episode ended [batch_size]
            weights: Importance weights [batch_size]

        Returns:
            loss: Loss value
            td_errors: TD errors (for updating priorities)
        """
        batch_size = q_dist.size(0)
        device = q_dist.device

        # Ensure supports are on correct device
        if self.supports.device != device:
            self.supports = self.supports.to(device)

        # Get Q distribution for current state-action pairs
        current_dist = q_dist[range(batch_size), actions]  # [batch_size, num_atoms]

        # Get Q distribution for next state (Double DQN)
        next_dist = next_q_dist[range(batch_size), next_actions]  # [batch_size, num_atoms]

        # Compute target distribution
        target_dist = self._compute_target_distribution(
            rewards, next_dist, dones, device
        )

        # Compute KL divergence loss
        loss_per_sample = -torch.sum(target_dist * torch.log(current_dist + 1e-8), dim=1)

        # Apply importance weights
        if weights is not None:
            weights = torch.FloatTensor(weights).to(device)
            loss_per_sample = loss_per_sample * weights

        loss = loss_per_sample.mean()

        # Compute TD errors (for updating priorities)
        with torch.no_grad():
            # Use expected value of distribution to compute TD error
            current_q = torch.sum(current_dist * self.supports, dim=1)
            target_q = torch.sum(target_dist * self.supports, dim=1)
            td_errors = torch.abs(current_q - target_q).cpu().numpy()

        return loss, td_errors

    def _compute_target_distribution(self,
                                   rewards: torch.Tensor,
                                   next_dist: torch.Tensor,
                                   dones: torch.Tensor,
                                   device: torch.device) -> torch.Tensor:
        """Compute target distribution"""
        batch_size = rewards.size(0)

        # Compute target support points
        target_supports = rewards.unsqueeze(1) + self.gamma * self.supports.unsqueeze(0) * (~dones).unsqueeze(1).float()

        # Project target support points onto original support points
        target_supports = torch.clamp(target_supports, self.v_min, self.v_max)

        # Compute projection
        target_dist = torch.zeros(batch_size, self.num_atoms, device=device)

        # Compute indices for each target support point
        b = (target_supports - self.v_min) / self.delta_z
        l = b.floor().long()
        u = b.ceil().long()

        # Handle boundary cases
        l[(u > 0) * (l == u)] -= 1
        u[(l < (self.num_atoms - 1)) * (l == u)] += 1

        # Project distribution probabilities
        for i in range(batch_size):
            for j in range(self.num_atoms):
                if not dones[i]:
                    # Not terminal state, perform projection
                    l_idx = l[i, j]
                    u_idx = u[i, j]

                    # Lower bound projection
                    target_dist[i, l_idx] += next_dist[i, j] * (u[i, j].float() - b[i, j])
                    # Upper bound projection
                    target_dist[i, u_idx] += next_dist[i, j] * (b[i, j] - l[i, j].float())
                else:
                    # Terminal state, all probability concentrated at reward value
                    reward_idx = int((rewards[i] - self.v_min) / self.delta_z)
                    reward_idx = max(0, min(self.num_atoms - 1, reward_idx))
                    target_dist[i, reward_idx] = 1.0

        return target_dist

    def q_values_from_distribution(self, q_dist: torch.Tensor) -> torch.Tensor:
        """Compute Q-values from distribution (expected value)"""
        if self.supports.device != q_dist.device:
            self.supports = self.supports.to(q_dist.device)

        # Compute expected Q-value for each action
        q_values = torch.sum(q_dist * self.supports, dim=-1)
        return q_values


class CategoricalDQNLoss:
    """Categorical DQN loss (simplified version of C51)"""

    def __init__(self, num_atoms: int = 51, v_min: float = -10, v_max: float = 10):
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        self.supports = torch.linspace(v_min, v_max, num_atoms)

    def compute_loss(self, pred_dist: torch.Tensor, target_dist: torch.Tensor) -> torch.Tensor:
        """Compute Categorical loss"""
        # Cross-entropy loss
        loss = -torch.sum(target_dist * torch.log(pred_dist + 1e-8), dim=-1)
        return loss.mean()


class QuantileRegressionLoss:
    """Quantile regression loss (for IQN/QR-DQN)"""

    def __init__(self, kappa: float = 1.0):
        self.kappa = kappa

    def compute_loss(self,
                     pred_quantiles: torch.Tensor,
                     target_quantiles: torch.Tensor,
                     tau: torch.Tensor) -> torch.Tensor:
        """
        Compute quantile regression loss

        Args:
            pred_quantiles: Predicted quantiles [batch_size, n_quantiles]
            target_quantiles: Target quantiles [batch_size, n_quantiles]
            tau: Quantile levels [n_quantiles]
        """
        # Compute quantile loss
        u = target_quantiles - pred_quantiles
        loss = (tau - (u < 0).float()) * self._huber_loss(u)

        return loss.mean()

    def _huber_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Huber loss"""
        return torch.where(
            x.abs() <= self.kappa,
            0.5 * x.pow(2),
            self.kappa * (x.abs() - 0.5 * self.kappa)
        )


def create_distributional_loss(loss_type: str = "c51", **kwargs):
    """Factory function: create distributional loss function"""
    if loss_type == "c51":
        return DistributionalLoss(**kwargs)
    elif loss_type == "categorical":
        return CategoricalDQNLoss(**kwargs)
    elif loss_type == "quantile":
        return QuantileRegressionLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")