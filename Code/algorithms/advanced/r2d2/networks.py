"""
R2D2 Neural Networks
R2D2算法的循环神经网络架构，包括LSTM/GRU网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any, Optional


class R2D2Network(nn.Module):
    """R2D2循环Q网络"""
    
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
        
        # 特征提取层
        self.feature_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 循环层
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
        
        # Q值输出层
        if dueling:
            # Dueling架构
            self.value_head = nn.Linear(recurrent_dim, 1)
            self.advantage_head = nn.Linear(recurrent_dim, action_dim)
        else:
            # 普通DQN架构
            self.q_head = nn.Linear(recurrent_dim, action_dim)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, 
                states: torch.Tensor,
                hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播
        
        Args:
            states: 状态序列 [batch_size, seq_len, state_dim]
            hidden_state: 隐藏状态 (h, c) for LSTM or h for GRU
            
        Returns:
            q_values: Q值 [batch_size, seq_len, action_dim]
            new_hidden_state: 新的隐藏状态
        """
        # 处理不同的输入形状
        if len(states.shape) == 2:
            # 单时间步输入: [batch_size, state_dim] -> [batch_size, 1, state_dim]
            states = states.unsqueeze(1)
            batch_size, seq_len, _ = states.shape
        elif len(states.shape) == 3:
            # 序列输入: [batch_size, seq_len, state_dim]
            batch_size, seq_len, _ = states.shape
        else:
            raise ValueError(f"Unsupported states shape: {states.shape}. Expected 2D or 3D tensor.")
        
        # 特征提取
        # 将序列维度展平处理
        states_flat = states.view(-1, self.state_dim)
        features_flat = self.feature_layers(states_flat)
        features = features_flat.view(batch_size, seq_len, -1)
        
        # 循环层处理
        if hidden_state is None:
            recurrent_out, new_hidden_state = self.recurrent(features)
        else:
            recurrent_out, new_hidden_state = self.recurrent(features, hidden_state)
        
        # Q值计算
        if self.dueling:
            # Dueling DQN
            # 将序列维度展平
            recurrent_flat = recurrent_out.contiguous().view(-1, self.recurrent_dim)
            
            values = self.value_head(recurrent_flat)  # [batch_size * seq_len, 1]
            advantages = self.advantage_head(recurrent_flat)  # [batch_size * seq_len, action_dim]
            
            # Dueling aggregation
            q_values_flat = values + advantages - advantages.mean(dim=1, keepdim=True)
            q_values = q_values_flat.view(batch_size, seq_len, self.action_dim)
        else:
            # 普通DQN
            recurrent_flat = recurrent_out.contiguous().view(-1, self.recurrent_dim)
            q_values_flat = self.q_head(recurrent_flat)
            q_values = q_values_flat.view(batch_size, seq_len, self.action_dim)
        
        return q_values, new_hidden_state
    
    def init_hidden_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """初始化隐藏状态"""
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
        """获取Q值（用于推理）"""
        with torch.no_grad():
            return self.forward(states, hidden_state)


class R2D2ConvNetwork(nn.Module):
    """R2D2卷积+循环网络（用于图像输入）"""
    
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
        
        # 卷积特征提取
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU()
        )
        
        # 计算卷积输出大小（假设84x84输入）
        conv_output_size = self._get_conv_output_size(input_channels, 84, 84)
        
        # 全连接层
        self.fc = nn.Linear(conv_output_size, 512)
        
        # 循环层
        if recurrent_type.upper() == 'LSTM':
            self.recurrent = nn.LSTM(512, recurrent_dim, num_layers, batch_first=True)
        else:
            self.recurrent = nn.GRU(512, recurrent_dim, num_layers, batch_first=True)
        
        # Q值头
        self.q_head = nn.Linear(recurrent_dim, action_dim)
    
    def _get_conv_output_size(self, channels, height, width):
        """计算卷积层输出尺寸"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, channels, height, width)
            dummy_output = self.conv_layers(dummy_input)
            return int(np.prod(dummy_output.shape[1:]))
    
    def forward(self, 
                images: torch.Tensor,
                hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """前向传播"""
        batch_size, seq_len = images.shape[:2]
        
        # 重塑输入用于卷积处理
        images_flat = images.view(-1, *images.shape[2:])
        conv_features = self.conv_layers(images_flat)
        conv_features = conv_features.view(conv_features.size(0), -1)
        fc_features = F.relu(self.fc(conv_features))
        
        # 重塑回序列格式
        features = fc_features.view(batch_size, seq_len, -1)
        
        # 循环处理
        if hidden_state is None:
            recurrent_out, new_hidden_state = self.recurrent(features)
        else:
            recurrent_out, new_hidden_state = self.recurrent(features, hidden_state)
        
        # Q值计算
        recurrent_flat = recurrent_out.contiguous().view(-1, self.recurrent_dim)
        q_values_flat = self.q_head(recurrent_flat)
        q_values = q_values_flat.view(batch_size, seq_len, self.action_dim)
        
        return q_values, new_hidden_state


def create_r2d2_network(state_space, action_space, network_config: Dict[str, Any] = None):
    """
    工厂函数：创建R2D2网络
    
    Args:
        state_space: 状态空间
        action_space: 动作空间  
        network_config: 网络配置
        
    Returns:
        R2D2网络实例
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
    
    # 确定动作维度 - R2D2只支持离散动作
    if not hasattr(action_space, 'n'):
        # 如果是连续动作空间，进行离散化
        action_bins = network_config.get('action_bins', 5) if network_config else 5
        action_dim = action_bins ** action_space.shape[0]
        print(f"Warning: R2D2 discretizing continuous action space to {action_dim} actions")
    else:
        action_dim = action_space.n
    
    # 根据状态空间类型选择网络
    if len(state_space.shape) == 1:  # 向量状态
        state_dim = state_space.shape[0]
        # 过滤掉不适用于R2D2Network的参数
        network_kwargs = {k: v for k, v in default_config.items() if k != 'action_bins'}
        return R2D2Network(
            state_dim=state_dim,
            action_dim=action_dim,
            **network_kwargs
        )
    elif len(state_space.shape) == 3:  # 图像状态
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