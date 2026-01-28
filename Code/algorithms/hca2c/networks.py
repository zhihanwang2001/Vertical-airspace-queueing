"""
Neural Network Architectures for HCA2C Algorithm

This module defines the hierarchical network structure:
- GlobalPolicyNetwork: System-level decisions (arrival control, emergency mode)
- LayerPolicyNetwork: Layer-specific decisions (service intensity, transfers)
- CoordinationModule: Inter-layer message passing
- HierarchicalActorCritic: Combined actor-critic architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from typing import Dict, Tuple, List, Optional


class GlobalPolicyNetwork(nn.Module):
    """
    Global Policy Network - Responsible for system-level decisions

    Inputs:
        - System state (6 dimensions): system_metrics(3) + total_queue(1) + avg_util(1) + prev_reward(1)

    Outputs:
        - arrival_multiplier: Controls overall arrival rate [0.5, 5.0]
        - emergency_mode: Binary flag for emergency response
        - layer_attention: Attention weights for each layer (5 dimensions)
    """

    def __init__(self, state_dim: int = 6, hidden_dim: int = 256):
        super().__init__()

        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # Arrival rate control head (outputs mean and log_std for Gaussian)
        self.arrival_mean = nn.Linear(hidden_dim, 1)
        self.arrival_log_std = nn.Linear(hidden_dim, 1)

        # Emergency mode head (binary classification)
        self.emergency_head = nn.Linear(hidden_dim, 1)

        # Layer attention head (softmax over 5 layers)
        self.attention_head = nn.Linear(hidden_dim, 5)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using orthogonal initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)

        # Special initialization for output heads
        nn.init.orthogonal_(self.arrival_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.arrival_log_std.weight, gain=0.01)
        nn.init.orthogonal_(self.emergency_head.weight, gain=0.01)
        nn.init.orthogonal_(self.attention_head.weight, gain=0.01)

    def forward(self, global_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            global_state: [batch_size, state_dim] tensor

        Returns:
            Dictionary containing:
                - arrival_mean: Mean of arrival multiplier distribution
                - arrival_log_std: Log std of arrival multiplier distribution
                - emergency_logit: Logit for emergency mode
                - layer_attention: Softmax attention weights for layers
        """
        # Handle NaN in input
        global_state = torch.nan_to_num(global_state, nan=0.0, posinf=1.0, neginf=-1.0)

        # Extract shared features
        features = self.shared(global_state)
        features = torch.nan_to_num(features, nan=0.0)

        # Arrival rate control
        arrival_mean = self.arrival_mean(features)
        arrival_log_std = self.arrival_log_std(features)
        arrival_log_std = torch.clamp(arrival_log_std, min=-5, max=2)

        # Ensure no NaN
        arrival_mean = torch.nan_to_num(arrival_mean, nan=0.0)
        arrival_log_std = torch.nan_to_num(arrival_log_std, nan=-1.0)

        # Emergency mode
        emergency_logit = self.emergency_head(features)
        emergency_logit = torch.nan_to_num(emergency_logit, nan=0.0)

        # Layer attention (softmax to get probability distribution)
        attention_raw = self.attention_head(features)
        attention_raw = torch.nan_to_num(attention_raw, nan=0.0)
        layer_attention = F.softmax(attention_raw, dim=-1)

        return {
            'arrival_mean': arrival_mean,
            'arrival_log_std': arrival_log_std,
            'emergency_logit': emergency_logit,
            'layer_attention': layer_attention,
            'features': features
        }


class LayerPolicyNetwork(nn.Module):
    """
    Layer Policy Network - Responsible for single layer decisions

    Inputs (8 dimensions):
        - queue_length, capacity, utilization (3)
        - service_rate, load_rate (2)
        - neighbor_pressure_up, neighbor_pressure_down (2)
        - global_attention_weight (1)

    Outputs:
        - service_intensity: Service rate multiplier [0.1, 2.0]
        - transfer_decision: Transfer direction and magnitude [-1, 1]
    """

    def __init__(self, layer_state_dim: int = 8, hidden_dim: int = 128, layer_idx: int = 0):
        super().__init__()

        self.layer_idx = layer_idx
        self.layer_state_dim = layer_state_dim
        self.hidden_dim = hidden_dim

        # Feature extractor
        self.network = nn.Sequential(
            nn.Linear(layer_state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # Service intensity head (Gaussian)
        self.service_mean = nn.Linear(hidden_dim, 1)
        self.service_log_std = nn.Linear(hidden_dim, 1)

        # Transfer decision head (Gaussian, negative=up, positive=down)
        self.transfer_mean = nn.Linear(hidden_dim, 1)
        self.transfer_log_std = nn.Linear(hidden_dim, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)

        # Small initialization for output heads
        for head in [self.service_mean, self.service_log_std,
                     self.transfer_mean, self.transfer_log_std]:
            nn.init.orthogonal_(head.weight, gain=0.01)

    def forward(self, layer_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            layer_state: [batch_size, layer_state_dim] tensor

        Returns:
            Dictionary containing action distribution parameters
        """
        # Handle NaN in input
        layer_state = torch.nan_to_num(layer_state, nan=0.0, posinf=1.0, neginf=-1.0)

        features = self.network(layer_state)

        # Service intensity
        service_mean = self.service_mean(features)
        service_log_std = self.service_log_std(features)
        service_log_std = torch.clamp(service_log_std, min=-5, max=2)

        # Transfer decision
        transfer_mean = self.transfer_mean(features)
        transfer_log_std = self.transfer_log_std(features)
        transfer_log_std = torch.clamp(transfer_log_std, min=-5, max=2)

        # Ensure no NaN
        service_mean = torch.nan_to_num(service_mean, nan=0.0)
        service_log_std = torch.nan_to_num(service_log_std, nan=-1.0)
        transfer_mean = torch.nan_to_num(transfer_mean, nan=0.0)
        transfer_log_std = torch.nan_to_num(transfer_log_std, nan=-1.0)

        return {
            'service_mean': service_mean,
            'service_log_std': service_log_std,
            'transfer_mean': transfer_mean,
            'transfer_log_std': transfer_log_std,
            'features': features
        }


class CoordinationModule(nn.Module):
    """
    Coordination Module - Enables inter-layer communication

    Uses 1D convolution for message passing between adjacent layers,
    allowing each layer to consider its neighbors' states.
    """

    def __init__(self, layer_dim: int = 128, n_layers: int = 5, n_heads: int = 4):
        super().__init__()

        self.layer_dim = layer_dim
        self.n_layers = n_layers

        # 1D Convolution for local message passing (kernel_size=3 for adjacent layers)
        self.local_conv = nn.Conv1d(
            in_channels=layer_dim,
            out_channels=layer_dim,
            kernel_size=3,
            padding=1,
            groups=1
        )

        # Self-attention for global coordination
        self.attention = nn.MultiheadAttention(
            embed_dim=layer_dim,
            num_heads=n_heads,
            dropout=0.1,
            batch_first=True
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(layer_dim)
        self.norm2 = nn.LayerNorm(layer_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(layer_dim, layer_dim * 2),
            nn.ReLU(),
            nn.Linear(layer_dim * 2, layer_dim)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        nn.init.xavier_uniform_(self.local_conv.weight)
        nn.init.constant_(self.local_conv.bias, 0)

    def forward(self, layer_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with inter-layer coordination

        Args:
            layer_features: [batch_size, n_layers, layer_dim] tensor

        Returns:
            Coordinated features: [batch_size, n_layers, layer_dim]
        """
        batch_size = layer_features.shape[0]

        # Local message passing via 1D convolution
        # Reshape for conv1d: [batch, channels, length]
        x = layer_features.transpose(1, 2)  # [batch, layer_dim, n_layers]
        x = self.local_conv(x)
        x = F.relu(x)
        x = x.transpose(1, 2)  # [batch, n_layers, layer_dim]

        # Residual connection
        x = self.norm1(layer_features + x)

        # Self-attention for global coordination
        attn_out, _ = self.attention(x, x, x)
        x = self.norm2(x + attn_out)

        # Feed-forward
        x = x + self.ffn(x)

        return x


class HierarchicalValueNetwork(nn.Module):
    """
    Hierarchical Value Network for critic

    Estimates state value using both global and layer-specific information.
    """

    def __init__(self, global_dim: int = 6, layer_dim: int = 8,
                 hidden_dim: int = 256, n_layers: int = 5):
        super().__init__()

        self.n_layers = n_layers

        # Global value stream
        self.global_stream = nn.Sequential(
            nn.Linear(global_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        # Layer value stream (processes all layers together)
        self.layer_stream = nn.Sequential(
            nn.Linear(layer_dim * n_layers, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        # Combined value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)

        # Small initialization for value head
        nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)

    def forward(self, global_state: torch.Tensor,
                layer_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            global_state: [batch_size, global_dim]
            layer_states: [batch_size, n_layers, layer_dim]

        Returns:
            State value: [batch_size, 1]
        """
        # Process global state
        global_features = self.global_stream(global_state)

        # Process layer states (flatten all layers)
        batch_size = layer_states.shape[0]
        layer_flat = layer_states.view(batch_size, -1)
        layer_features = self.layer_stream(layer_flat)

        # Combine and compute value
        combined = torch.cat([global_features, layer_features], dim=-1)
        value = self.value_head(combined)

        return value


class HierarchicalActorCritic(nn.Module):
    """
    Complete Hierarchical Actor-Critic Network for HCA2C

    Combines:
    - GlobalPolicyNetwork for system-level decisions
    - LayerPolicyNetwork (x5) for layer-specific decisions
    - CoordinationModule for inter-layer communication
    - HierarchicalValueNetwork for value estimation
    """

    def __init__(self,
                 global_state_dim: int = 6,
                 layer_state_dim: int = 8,
                 global_hidden_dim: int = 256,
                 layer_hidden_dim: int = 128,
                 n_layers: int = 5):
        super().__init__()

        self.n_layers = n_layers
        self.global_state_dim = global_state_dim
        self.layer_state_dim = layer_state_dim

        # Global policy network
        self.global_policy = GlobalPolicyNetwork(
            state_dim=global_state_dim,
            hidden_dim=global_hidden_dim
        )

        # Layer policy networks (one per layer)
        self.layer_policies = nn.ModuleList([
            LayerPolicyNetwork(
                layer_state_dim=layer_state_dim + 1,  # +1 for attention weight
                hidden_dim=layer_hidden_dim,
                layer_idx=i
            )
            for i in range(n_layers)
        ])

        # Coordination module
        self.coordinator = CoordinationModule(
            layer_dim=layer_hidden_dim,
            n_layers=n_layers
        )

        # Layer feature extractor (to get features for coordination)
        self.layer_encoder = nn.Sequential(
            nn.Linear(layer_state_dim, layer_hidden_dim),
            nn.ReLU()
        )

        # Value network
        self.value_net = HierarchicalValueNetwork(
            global_dim=global_state_dim,
            layer_dim=layer_state_dim,
            hidden_dim=global_hidden_dim,
            n_layers=n_layers
        )

    def forward(self, global_state: torch.Tensor,
                layer_states: torch.Tensor,
                deterministic: bool = False) -> Tuple[Dict, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the entire hierarchical network

        Args:
            global_state: [batch_size, global_state_dim]
            layer_states: [batch_size, n_layers, layer_state_dim]
            deterministic: If True, use mean actions instead of sampling

        Returns:
            actions: Dictionary of action tensors
            log_probs: Log probabilities of actions
            values: State value estimates
        """
        batch_size = global_state.shape[0]
        device = global_state.device

        # 1. Global policy forward
        global_out = self.global_policy(global_state)

        # 2. Encode layer states for coordination
        layer_features = []
        for i in range(self.n_layers):
            feat = self.layer_encoder(layer_states[:, i, :])
            layer_features.append(feat)
        layer_features = torch.stack(layer_features, dim=1)  # [batch, n_layers, hidden]

        # 3. Inter-layer coordination
        coordinated_features = self.coordinator(layer_features)

        # 4. Layer policy forward (with attention weights and coordinated features)
        layer_outputs = []
        for i in range(self.n_layers):
            # Concatenate layer state with attention weight
            attention_weight = global_out['layer_attention'][:, i:i+1]
            layer_input = torch.cat([
                layer_states[:, i, :],
                attention_weight
            ], dim=-1)

            layer_out = self.layer_policies[i](layer_input)
            layer_outputs.append(layer_out)

        # 5. Sample actions
        actions, log_probs = self._sample_actions(
            global_out, layer_outputs, deterministic
        )

        # 6. Compute value
        values = self.value_net(global_state, layer_states)

        return actions, log_probs, values

    def _sample_actions(self, global_out: Dict, layer_outputs: List[Dict],
                        deterministic: bool) -> Tuple[Dict, torch.Tensor]:
        """
        Sample actions from the policy distributions

        Returns:
            actions: Dictionary containing all action components
            log_probs: Total log probability of the action
        """
        log_probs = []

        # Sample arrival multiplier
        arrival_dist = Normal(
            global_out['arrival_mean'],
            global_out['arrival_log_std'].exp()
        )
        if deterministic:
            arrival_action = global_out['arrival_mean']
        else:
            arrival_action = arrival_dist.rsample()

        # Transform to [0.5, 5.0] range
        arrival_action = torch.sigmoid(arrival_action) * 4.5 + 0.5
        log_probs.append(arrival_dist.log_prob(arrival_action).sum(dim=-1))

        # Sample emergency mode (Bernoulli)
        emergency_prob = torch.sigmoid(global_out['emergency_logit'])
        # Clamp to valid range for Bernoulli
        emergency_prob = torch.clamp(emergency_prob, min=1e-6, max=1.0-1e-6)
        if deterministic:
            emergency_action = (emergency_prob > 0.5).float()
        else:
            emergency_action = torch.bernoulli(emergency_prob)

        # Sample layer actions
        service_actions = []
        transfer_actions = []

        for i, layer_out in enumerate(layer_outputs):
            # Service intensity
            service_dist = Normal(
                layer_out['service_mean'],
                layer_out['service_log_std'].exp()
            )
            if deterministic:
                service_action = layer_out['service_mean']
            else:
                service_action = service_dist.rsample()

            # Transform to [0.1, 2.0] range
            service_action = torch.sigmoid(service_action) * 1.9 + 0.1
            service_actions.append(service_action)
            log_probs.append(service_dist.log_prob(service_action).sum(dim=-1))

            # Transfer decision
            transfer_dist = Normal(
                layer_out['transfer_mean'],
                layer_out['transfer_log_std'].exp()
            )
            if deterministic:
                transfer_action = layer_out['transfer_mean']
            else:
                transfer_action = transfer_dist.rsample()

            # Transform to [-1, 1] range
            transfer_action = torch.tanh(transfer_action)
            transfer_actions.append(transfer_action)
            log_probs.append(transfer_dist.log_prob(transfer_action).sum(dim=-1))

        # Combine actions
        actions = {
            'arrival_multiplier': arrival_action,
            'emergency_mode': emergency_action,
            'service_intensities': torch.cat(service_actions, dim=-1),
            'transfer_decisions': torch.cat(transfer_actions, dim=-1),
            'layer_attention': global_out['layer_attention']
        }

        # Sum log probabilities
        total_log_prob = torch.stack(log_probs, dim=-1).sum(dim=-1)

        return actions, total_log_prob

    def evaluate_actions(self, global_state: torch.Tensor,
                        layer_states: torch.Tensor,
                        actions: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate given actions (for PPO-style updates)

        Returns:
            log_probs: Log probabilities of the actions
            values: State value estimates
            entropy: Policy entropy
        """
        # Forward pass to get distributions
        global_out = self.global_policy(global_state)

        # Encode and coordinate layer features
        layer_features = []
        for i in range(self.n_layers):
            feat = self.layer_encoder(layer_states[:, i, :])
            layer_features.append(feat)
        layer_features = torch.stack(layer_features, dim=1)
        coordinated_features = self.coordinator(layer_features)

        # Get layer outputs
        layer_outputs = []
        for i in range(self.n_layers):
            attention_weight = global_out['layer_attention'][:, i:i+1]
            layer_input = torch.cat([
                layer_states[:, i, :],
                attention_weight
            ], dim=-1)
            layer_out = self.layer_policies[i](layer_input)
            layer_outputs.append(layer_out)

        # Compute log probs and entropy
        log_probs, entropy = self._compute_log_probs_and_entropy(
            global_out, layer_outputs, actions
        )

        # Compute values
        values = self.value_net(global_state, layer_states)

        return log_probs, values, entropy

    def _compute_log_probs_and_entropy(self, global_out: Dict,
                                       layer_outputs: List[Dict],
                                       actions: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute log probabilities and entropy for given actions"""
        log_probs = []
        entropies = []

        # Arrival multiplier
        arrival_dist = Normal(
            global_out['arrival_mean'],
            global_out['arrival_log_std'].exp()
        )
        # Inverse transform from [0.5, 5.0] to unbounded
        arrival_raw = torch.logit((actions['arrival_multiplier'] - 0.5) / 4.5)
        log_probs.append(arrival_dist.log_prob(arrival_raw).sum(dim=-1))
        entropies.append(arrival_dist.entropy().sum(dim=-1))

        # Layer actions
        for i, layer_out in enumerate(layer_outputs):
            # Service intensity
            service_dist = Normal(
                layer_out['service_mean'],
                layer_out['service_log_std'].exp()
            )
            service_raw = torch.logit(
                (actions['service_intensities'][:, i:i+1] - 0.1) / 1.9
            )
            log_probs.append(service_dist.log_prob(service_raw).sum(dim=-1))
            entropies.append(service_dist.entropy().sum(dim=-1))

            # Transfer decision
            transfer_dist = Normal(
                layer_out['transfer_mean'],
                layer_out['transfer_log_std'].exp()
            )
            transfer_raw = torch.atanh(
                actions['transfer_decisions'][:, i:i+1].clamp(-0.999, 0.999)
            )
            log_probs.append(transfer_dist.log_prob(transfer_raw).sum(dim=-1))
            entropies.append(transfer_dist.entropy().sum(dim=-1))

        total_log_prob = torch.stack(log_probs, dim=-1).sum(dim=-1)
        total_entropy = torch.stack(entropies, dim=-1).sum(dim=-1)

        return total_log_prob, total_entropy


if __name__ == "__main__":
    # Test the networks
    print("Testing HCA2C Networks...")

    batch_size = 4
    n_layers = 5
    global_dim = 6
    layer_dim = 8

    # Create dummy inputs
    global_state = torch.randn(batch_size, global_dim)
    layer_states = torch.randn(batch_size, n_layers, layer_dim)

    # Test HierarchicalActorCritic
    model = HierarchicalActorCritic(
        global_state_dim=global_dim,
        layer_state_dim=layer_dim
    )

    actions, log_probs, values = model(global_state, layer_states)

    print(f"✅ Global state shape: {global_state.shape}")
    print(f"✅ Layer states shape: {layer_states.shape}")
    print(f"✅ Arrival multiplier shape: {actions['arrival_multiplier'].shape}")
    print(f"✅ Service intensities shape: {actions['service_intensities'].shape}")
    print(f"✅ Transfer decisions shape: {actions['transfer_decisions'].shape}")
    print(f"✅ Log probs shape: {log_probs.shape}")
    print(f"✅ Values shape: {values.shape}")

    # Test action evaluation
    log_probs_eval, values_eval, entropy = model.evaluate_actions(
        global_state, layer_states, actions
    )
    print(f"✅ Evaluated log probs shape: {log_probs_eval.shape}")
    print(f"✅ Entropy shape: {entropy.shape}")

    print("\n✅ All network tests passed!")
