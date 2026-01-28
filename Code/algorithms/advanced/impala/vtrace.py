"""
V-trace Implementation
Implementation of V-trace importance sampling correction algorithm
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, NamedTuple


class VTraceReturns(NamedTuple):
    """V-trace return values"""
    vs: torch.Tensor  # V-trace value targets
    pg_advantages: torch.Tensor  # Policy gradient advantages


class VTrace:
    """V-trace algorithm implementation"""

    def __init__(self,
                 rho_bar: float = 1.0,
                 c_bar: float = 1.0,
                 gamma: float = 0.99):
        """
        Initialize V-trace

        Args:
            rho_bar: Importance weight clipping threshold
            c_bar: Temporal difference error weight clipping threshold
            gamma: Discount factor
        """
        self.rho_bar = rho_bar
        self.c_bar = c_bar
        self.gamma = gamma

    def compute_vtrace_targets(self,
                               behavior_log_probs: torch.Tensor,
                               target_log_probs: torch.Tensor,
                               rewards: torch.Tensor,
                               values: torch.Tensor,
                               bootstrap_value: torch.Tensor,
                               dones: torch.Tensor) -> VTraceReturns:
        """
        Compute V-trace targets

        Args:
            behavior_log_probs: Log probabilities of behavior policy [T, B]
            target_log_probs: Log probabilities of target policy [T, B]
            rewards: Reward sequence [T, B]
            values: Value estimates [T, B]
            bootstrap_value: Bootstrap value [B]
            dones: Whether episode ended [T, B]

        Returns:
            VTraceReturns containing vs and pg_advantages
        """
        # Compute importance weights
        log_rhos = target_log_probs - behavior_log_probs
        rhos = torch.exp(log_rhos)

        # Clip importance weights
        clipped_rhos = torch.clamp(rhos, max=self.rho_bar)
        cs = torch.clamp(rhos, max=self.c_bar)

        # Adjust dimensions for computation
        T, B = rewards.shape

        # Compute temporal difference errors
        values_t_plus_1 = torch.cat([values[1:], bootstrap_value.unsqueeze(0)], dim=0)
        deltas = clipped_rhos * (rewards + self.gamma * values_t_plus_1 * (1 - dones) - values)

        # Backward compute V-trace values
        vs_minus_v_xs = []
        vs_minus_v_x = torch.zeros_like(bootstrap_value)

        # Compute backward from last timestep
        for t in reversed(range(T)):
            vs_minus_v_x = deltas[t] + self.gamma * cs[t] * (1 - dones[t]) * vs_minus_v_x
            vs_minus_v_xs.append(vs_minus_v_x)

        # Reverse list to get forward order
        vs_minus_v_xs.reverse()
        vs_minus_v_xs = torch.stack(vs_minus_v_xs, dim=0)

        # Compute V-trace value targets
        vs = values + vs_minus_v_xs

        # Compute policy gradient advantages
        vs_t_plus_1 = torch.cat([vs[1:], bootstrap_value.unsqueeze(0)], dim=0)
        pg_advantages = clipped_rhos * (rewards + self.gamma * vs_t_plus_1 * (1 - dones) - values)

        return VTraceReturns(vs=vs, pg_advantages=pg_advantages)

    def compute_policy_gradient_loss(self,
                                     log_probs: torch.Tensor,
                                     advantages: torch.Tensor) -> torch.Tensor:
        """
        Compute policy gradient loss

        Args:
            log_probs: Action log probabilities [T, B]
            advantages: V-trace advantages [T, B]

        Returns:
            Policy gradient loss
        """
        return -(log_probs * advantages.detach()).mean()

    def compute_value_loss(self,
                           values: torch.Tensor,
                           vs_targets: torch.Tensor) -> torch.Tensor:
        """
        Compute value function loss

        Args:
            values: Predicted values [T, B]
            vs_targets: V-trace target values [T, B]

        Returns:
            Value function loss (MSE)
        """
        return F.mse_loss(values, vs_targets.detach())

    def compute_entropy_loss(self,
                             entropies: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy loss (for exploration)

        Args:
            entropies: Policy entropy [T, B]

        Returns:
            Entropy loss (negative entropy, to maximize entropy)
        """
        return -entropies.mean()


def compute_vtrace_loss(vtrace: VTrace,
                        behavior_log_probs: torch.Tensor,
                        target_log_probs: torch.Tensor,
                        rewards: torch.Tensor,
                        values: torch.Tensor,
                        bootstrap_value: torch.Tensor,
                        dones: torch.Tensor,
                        entropies: torch.Tensor,
                        entropy_coeff: float = 0.01) -> Tuple[torch.Tensor, dict]:
    """
    Compute complete IMPALA loss

    Args:
        vtrace: V-trace instance
        behavior_log_probs: Behavior policy log probabilities
        target_log_probs: Target policy log probabilities
        rewards: Reward sequence
        values: Value estimates
        bootstrap_value: Bootstrap value
        dones: Whether episode ended
        entropies: Policy entropy
        entropy_coeff: Entropy coefficient

    Returns:
        total_loss: Total loss
        loss_info: Loss information dictionary
    """
    # Compute V-trace targets
    vtrace_returns = vtrace.compute_vtrace_targets(
        behavior_log_probs, target_log_probs, rewards, values, bootstrap_value, dones
    )

    # Compute individual losses
    pg_loss = vtrace.compute_policy_gradient_loss(target_log_probs, vtrace_returns.pg_advantages)
    value_loss = vtrace.compute_value_loss(values, vtrace_returns.vs)
    entropy_loss = vtrace.compute_entropy_loss(entropies)

    # Total loss
    total_loss = pg_loss + 0.5 * value_loss + entropy_coeff * entropy_loss

    # Loss information
    loss_info = {
        'total_loss': total_loss.item(),
        'pg_loss': pg_loss.item(),
        'value_loss': value_loss.item(),
        'entropy_loss': entropy_loss.item(),
        'mean_advantage': vtrace_returns.pg_advantages.mean().item(),
        'mean_value': values.mean().item()
    }

    return total_loss, loss_info
