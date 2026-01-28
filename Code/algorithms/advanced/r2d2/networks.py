"""
R2D2 Neural Networks
Recurrent neural network architecture for R2D2 algorithm, including LSTM/GRU networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any, Optional


class R2D2Network(nn.Module):
    """R2D2 recurrent Q-network"""

    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 512,
                 recurrent_dim: int = 256,
                 num_layers: int = 1,
                 recurrent_type: str = 'LSTM',
                 dueling: bool = True):
        super(R2D2Network, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.recurrent_dim = recurrent_dim
        self.num_layers = num_layers
        self.recurrent_type = recurrent_type
        self.dueling = dueling

        # Feature extraction layers
        self.feature_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Recurrent layer
        if recurrent_type.upper() == 'LSTM':
            self.recurrent = nn.LSTM(
                input_size=hidden_dim,
                hidden_size=recurrent_dim,
                num_layers=num_layers,
                batch_first=True
            )
        elif recurrent_type.upper() == 'GRU':
            self.recurrent = nn.GRU(
                input_size=hidden_dim,
                hidden_size=recurrent_dim,
                num_layers=num_layers,
                batch_first=True
            )
        else:
            raise ValueError(f"Unsupported recurrent type: {recurrent_type}")

        # Q-value output layer
        if dueling:
            # Dueling architecture
            self.value_head = nn.Linear(recurrent_dim, 1)
            self.advantage_head = nn.Linear(recurrent_dim, action_dim)
        else:
            # Standard DQN architecture
            self.q_head = nn.Linear(recurrent_dim, action_dim)

        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self,
                states: torch.Tensor,
                hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass

        Args:
            states: State sequence [batch_size, seq_len, state_dim]
            hidden_state: Hidden state (h, c) for LSTM or h for GRU

        Returns:
            q_values: Q-values [batch_size, seq_len, action_dim]
            new_hidden_state: New hidden state
        """
        # Handle different input shapes
        if len(states.shape) == 2:
            # Single timestep input: [batch_size, state_dim] -> [batch_size, 1, state_dim]
            states = states.unsqueeze(1)
            batch_size, seq_len, _ = states.shape
        elif len(states.shape) == 3:
            # Sequence input: [batch_size, seq_len, state_dim]
            batch_size, seq_len, _ = states.shape
        else:
            raise ValueError(f"Unsupported states shape: {states.shape}. Expected 2D or 3D tensor.")

        # Feature extraction
        # Flatten sequence dimension for processing
        states_flat = states.view(-1, self.state_dim)
        features_flat = self.feature_layers(states_flat)
        features = features_flat.view(batch_size, seq_len, -1)

        # Recurrent layer processing
        if hidden_state is None:
            recurrent_out, new_hidden_state = self.recurrent(features)
        else:
            recurrent_out, new_hidden_state = self.recurrent(features, hidden_state)

        # Q-value calculation
        if self.dueling:
            # Dueling DQN
            # Flatten sequence dimension
            recurrent_flat = recurrent_out.contiguous().view(-1, self.recurrent_dim)

            values = self.value_head(recurrent_flat)  # [batch_size * seq_len, 1]
            advantages = self.advantage_head(recurrent_flat)  # [batch_size * seq_len, action_dim]

            # Dueling aggregation
            q_values_flat = values + advantages - advantages.mean(dim=1, keepdim=True)
            q_values = q_values_flat.view(batch_size, seq_len, self.action_dim)
        else:
            # Standard DQN
            recurrent_flat = recurrent_out.contiguous().view(-1, self.recurrent_dim)
            q_values_flat = self.q_head(recurrent_flat)
            q_values = q_values_flat.view(batch_size, seq_len, self.action_dim)
        
        return q_values, new_hidden_state
    
    def init_hidden_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state"""
        if self.recurrent_type.upper() == 'LSTM':
            h = torch.zeros(self.num_layers, batch_size, self.recurrent_dim, device=device)
            c = torch.zeros(self.num_layers, batch_size, self.recurrent_dim, device=device)
            return (h, c)
        else:  # GRU
            h = torch.zeros(self.num_layers, batch_size, self.recurrent_dim, device=device)
            return (h,)
    
    def get_q_values(self,
                     states: torch.Tensor,
                     hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Get Q-values (for inference)"""
        with torch.no_grad():
            return self.forward(states, hidden_state)


class R2D2ConvNetwork(nn.Module):
    """R2D2 convolutional + recurrent network (for image input)"""
    
    def __init__(self,
                 input_channels: int,
                 action_dim: int,
                 recurrent_dim: int = 256,
                 num_layers: int = 1,
                 recurrent_type: str = 'LSTM'):
        super(R2D2ConvNetwork, self).__init__()
        
        self.action_dim = action_dim
        self.recurrent_dim = recurrent_dim
        self.num_layers = num_layers
        self.recurrent_type = recurrent_type

        # Convolutional feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU()
        )

        # Calculate convolutional output size (assuming 84x84 input)
        conv_output_size = self._get_conv_output_size(input_channels, 84, 84)

        # Fully connected layer
        self.fc = nn.Linear(conv_output_size, 512)

        # Recurrent layer
        if recurrent_type.upper() == 'LSTM':
            self.recurrent = nn.LSTM(512, recurrent_dim, num_layers, batch_first=True)
        else:
            self.recurrent = nn.GRU(512, recurrent_dim, num_layers, batch_first=True)

        # Q-value head
        self.q_head = nn.Linear(recurrent_dim, action_dim)
    
    def _get_conv_output_size(self, channels, height, width):
        """Calculate convolutional layer output size"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, channels, height, width)
            dummy_output = self.conv_layers(dummy_input)
            return int(np.prod(dummy_output.shape[1:]))
    
    def forward(self,
                images: torch.Tensor,
                hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """Forward pass"""
        batch_size, seq_len = images.shape[:2]

        # Reshape input for convolutional processing
        images_flat = images.view(-1, *images.shape[2:])
        conv_features = self.conv_layers(images_flat)
        conv_features = conv_features.view(conv_features.size(0), -1)
        fc_features = F.relu(self.fc(conv_features))

        # Reshape back to sequence format
        features = fc_features.view(batch_size, seq_len, -1)

        # Recurrent processing
        if hidden_state is None:
            recurrent_out, new_hidden_state = self.recurrent(features)
        else:
            recurrent_out, new_hidden_state = self.recurrent(features, hidden_state)

        # Q-value calculation
        recurrent_flat = recurrent_out.contiguous().view(-1, self.recurrent_dim)
        q_values_flat = self.q_head(recurrent_flat)
        q_values = q_values_flat.view(batch_size, seq_len, self.action_dim)
        
        return q_values, new_hidden_state


def create_r2d2_network(state_space, action_space, network_config: Dict[str, Any] = None):
    """
    Factory function: create R2D2 network

    Args:
        state_space: State space
        action_space: Action space
        network_config: Network configuration

    Returns:
        R2D2 network instance
    """
    default_config = {
        'hidden_dim': 512,
        'recurrent_dim': 256,
        'num_layers': 1,
        'recurrent_type': 'LSTM',
        'dueling': True
    }
    
    if network_config:
        default_config.update(network_config)

    # Determine action dimension - R2D2 only supports discrete actions
    if not hasattr(action_space, 'n'):
        # If continuous action space, discretize it
        action_bins = network_config.get('action_bins', 5) if network_config else 5
        action_dim = action_bins ** action_space.shape[0]
        print(f"Warning: R2D2 discretizing continuous action space to {action_dim} actions")
    else:
        action_dim = action_space.n

    # Select network based on state space type
    if len(state_space.shape) == 1:  # Vector state
        state_dim = state_space.shape[0]
        # Filter out parameters not applicable to R2D2Network
        network_kwargs = {k: v for k, v in default_config.items() if k != 'action_bins'}
        return R2D2Network(
            state_dim=state_dim,
            action_dim=action_dim,
            **network_kwargs
        )
    elif len(state_space.shape) == 3:  # Image state
        input_channels = state_space.shape[0]
        return R2D2ConvNetwork(
            input_channels=input_channels,
            action_dim=action_dim,
            recurrent_dim=default_config['recurrent_dim'],
            num_layers=default_config['num_layers'],
            recurrent_type=default_config['recurrent_type']
        )
    else:
        raise ValueError(f"Unsupported state space shape: {state_space.shape}")