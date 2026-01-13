"""
Rainbow DQN Network Architectures
包含Dueling网络和Noisy Networks的实现
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
        
        # 参数层
        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def forward(self, x):
        """前向传播"""
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def reset_parameters(self):
        """初始化参数"""
        mu_range = 1 / math.sqrt(self.in_features)
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
    
    def reset_noise(self):
        """重置噪声"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size):
        """生成缩放噪声"""
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
        
        # 支撑点
        self.register_buffer('supports', torch.linspace(v_min, v_max, num_atoms))
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        
        # 特征提取层
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Dueling架构
        # 价值流
        self.value_stream = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim//2, noisy_std),
            nn.ReLU(),
            NoisyLinear(hidden_dim//2, num_atoms, noisy_std)
        )
        
        # 优势流
        self.advantage_stream = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim//2, noisy_std),
            nn.ReLU(),
            NoisyLinear(hidden_dim//2, action_dim * num_atoms, noisy_std)
        )
    
    def forward(self, state):
        """前向传播"""
        batch_size = state.size(0)
        
        # 特征提取
        features = self.feature_layer(state)
        
        # 价值流
        value = self.value_stream(features)  # [batch_size, num_atoms]
        value = value.view(batch_size, 1, self.num_atoms)
        
        # 优势流
        advantage = self.advantage_stream(features)  # [batch_size, action_dim * num_atoms]
        advantage = advantage.view(batch_size, self.action_dim, self.num_atoms)
        
        # Dueling组合
        advantage_mean = advantage.mean(dim=1, keepdim=True)
        q_dist = value + advantage - advantage_mean
        
        # 应用softmax获得概率分布
        q_dist = F.softmax(q_dist, dim=-1)
        q_dist = q_dist.clamp(min=1e-3)  # 防止数值不稳定
        
        return q_dist
    
    def reset_noise(self):
        """重置所有噪声层"""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class RainbowCNN(nn.Module):
    """用于图像输入的Rainbow网络"""
    
    def __init__(self, input_channels, action_dim, hidden_dim=512, num_atoms=51, v_min=-10, v_max=10):
        super(RainbowCNN, self).__init__()
        
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        
        self.register_buffer('supports', torch.linspace(v_min, v_max, num_atoms))
        
        # CNN特征提取
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # 计算conv输出维度
        conv_out_dim = self._get_conv_out_dim(input_channels, 84, 84)
        
        # Dueling架构
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
        """计算卷积输出维度"""
        dummy_input = torch.zeros(1, channels, height, width)
        conv_out = self.conv_layers(dummy_input)
        return int(np.prod(conv_out.size()))
    
    def forward(self, state):
        """前向传播"""
        batch_size = state.size(0)
        
        # CNN特征提取
        conv_out = self.conv_layers(state)
        features = conv_out.view(batch_size, -1)
        
        # Dueling网络
        value = self.value_stream(features).view(batch_size, 1, self.num_atoms)
        advantage = self.advantage_stream(features).view(batch_size, self.action_dim, self.num_atoms)
        
        # 组合
        advantage_mean = advantage.mean(dim=1, keepdim=True)
        q_dist = value + advantage - advantage_mean
        
        # Softmax
        q_dist = F.softmax(q_dist, dim=-1)
        q_dist = q_dist.clamp(min=1e-3)
        
        return q_dist
    
    def reset_noise(self):
        """重置噪声"""
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


def create_rainbow_network(state_space, action_space, network_config=None):
    """工厂函数：创建Rainbow网络"""
    
    default_config = {
        'hidden_dim': 512,
        'num_atoms': 51,
        'v_min': -10,
        'v_max': 10,
        'noisy_std': 0.5
    }
    
    if network_config:
        default_config.update(network_config)
    
    # 确定动作维度
    if hasattr(action_space, 'n'):
        action_dim = action_space.n
    else:
        # 连续动作空间，使用离散化后的动作数量
        action_bins = network_config.get('action_bins', 2) if network_config else 2
        action_dim = action_bins ** action_space.shape[0]
    
    # 根据状态空间选择网络架构
    if len(state_space.shape) == 1:  # 向量输入
        return DuelingNoisyNetwork(
            state_dim=state_space.shape[0],
            action_dim=action_dim,
            **default_config
        )
    elif len(state_space.shape) == 3:  # 图像输入
        return RainbowCNN(
            input_channels=state_space.shape[0],
            action_dim=action_dim,
            **default_config
        )
    else:
        raise ValueError(f"Unsupported state space shape: {state_space.shape}")