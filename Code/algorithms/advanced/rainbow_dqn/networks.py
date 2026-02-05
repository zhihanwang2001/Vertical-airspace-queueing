"""
Rainbow DQN Network Architectures
Implementation of Dueling networks and Noisy Networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class NoisyLinear(nn.Module):
    """Noisy Linear Layer for exploration"""
    
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # Parameter layers
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        """Forward pass"""
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        """Initialize parameters"""
        mu_range = 1 / math.sqrt(self.in_features)

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        """Reset noise"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        """Generate scaled noise"""
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x


class DuelingNoisyNetwork(nn.Module):
    """Dueling + Noisy Networks for Rainbow DQN"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=512, num_atoms=51, v_min=-10, v_max=10, noisy_std=0.5, **kwargs):
        super(DuelingNoisyNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max

        # Support points
        self.register_buffer('supports', torch.linspace(v_min, v_max, num_atoms))
        self.delta_z = (v_max - v_min) / (num_atoms - 1)

        # Feature extraction layer
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Dueling architecture
        # Value stream
        self.value_stream = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim//2, noisy_std),
            nn.ReLU(),
            NoisyLinear(hidden_dim//2, num_atoms, noisy_std)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim//2, noisy_std),
            nn.ReLU(),
            NoisyLinear(hidden_dim//2, action_dim * num_atoms, noisy_std)
        )

    def forward(self, state):
        """Forward pass"""
        batch_size = state.size(0)

        # Feature extraction
        features = self.feature_layer(state)

        # Value stream
        value = self.value_stream(features)  # [batch_size, num_atoms]
        value = value.view(batch_size, 1, self.num_atoms)

        # Advantage stream
        advantage = self.advantage_stream(features)  # [batch_size, action_dim * num_atoms]
        advantage = advantage.view(batch_size, self.action_dim, self.num_atoms)

        # Dueling combination
        advantage_mean = advantage.mean(dim=1, keepdim=True)
        q_dist = value + advantage - advantage_mean

        # Apply softmax to get probability distribution
        q_dist = F.softmax(q_dist, dim=-1)
        q_dist = q_dist.clamp(min=1e-3)  # Prevent numerical instability

        return q_dist

    def reset_noise(self):
        """Reset all noise layers"""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class RainbowCNN(nn.Module):
    """Rainbow network for image input"""

    def __init__(self, input_channels, action_dim, hidden_dim=512, num_atoms=51, v_min=-10, v_max=10):
        super(RainbowCNN, self).__init__()

        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max

        self.register_buffer('supports', torch.linspace(v_min, v_max, num_atoms))

        # CNN feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Calculate conv output dimension
        conv_out_dim = self._get_conv_out_dim(input_channels, 84, 84)

        # Dueling architecture
        self.value_stream = nn.Sequential(
            NoisyLinear(conv_out_dim, hidden_dim),
            nn.ReLU(),
            NoisyLinear(hidden_dim, num_atoms)
        )

        self.advantage_stream = nn.Sequential(
            NoisyLinear(conv_out_dim, hidden_dim),
            nn.ReLU(),
            NoisyLinear(hidden_dim, action_dim * num_atoms)
        )

    def _get_conv_out_dim(self, channels, height, width):
        """Calculate convolutional output dimension"""
        dummy_input = torch.zeros(1, channels, height, width)
        conv_out = self.conv_layers(dummy_input)
        return int(np.prod(conv_out.size()))

    def forward(self, state):
        """Forward pass"""
        batch_size = state.size(0)

        # CNN feature extraction
        conv_out = self.conv_layers(state)
        features = conv_out.view(batch_size, -1)

        # Dueling network
        value = self.value_stream(features).view(batch_size, 1, self.num_atoms)
        advantage = self.advantage_stream(features).view(batch_size, self.action_dim, self.num_atoms)

        # Combination
        advantage_mean = advantage.mean(dim=1, keepdim=True)
        q_dist = value + advantage - advantage_mean

        # Softmax
        q_dist = F.softmax(q_dist, dim=-1)
        q_dist = q_dist.clamp(min=1e-3)

        return q_dist

    def reset_noise(self):
        """Reset noise"""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


def create_rainbow_network(state_space, action_space, network_config=None):
    """Factory function: create Rainbow network"""

    default_config = {
        'hidden_dim': 512,
        'num_atoms': 51,
        'v_min': -10,
        'v_max': 10,
        'noisy_std': 0.5
    }

    if network_config:
        default_config.update(network_config)

    # Determine action dimension
    if hasattr(action_space, 'n'):
        action_dim = action_space.n
    else:
        # Continuous action space, use discretized action count
        action_bins = network_config.get('action_bins', 2) if network_config else 2
        action_dim = action_bins ** action_space.shape[0]

    # Select network architecture based on state space
    if len(state_space.shape) == 1:  # Vector input
        return DuelingNoisyNetwork(
            state_dim=state_space.shape[0],
            action_dim=action_dim,
            **default_config
        )
    elif len(state_space.shape) == 3:  # Image input
        return RainbowCNN(
            input_channels=state_space.shape[0],
            action_dim=action_dim,
            **default_config
        )
    else:
        raise ValueError(f"Unsupported state space shape: {state_space.shape}")