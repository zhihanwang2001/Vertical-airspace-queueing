"""
SAC v2 Neural Networks
SAC v2算法的神经网络架构，包括Actor和Critic网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any, Optional
from torch.distributions import Normal


class ActorNetwork(nn.Module):
    """SAC Actor网络 - 随机策略网络"""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int, 
                 hidden_dim: int = 256,
                 max_action: float = 1.0,
                 log_std_min: float = -20,
                 log_std_max: float = 2):
        super(ActorNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # 共享特征层
        self.feature_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 均值和对数标准差输出
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier初始化
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
        
        # 最后一层特殊初始化
        nn.init.uniform_(self.mean_layer.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.mean_layer.bias, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_layer.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_layer.bias, -3e-3, 3e-3)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            state: 状态 [batch_size, state_dim]
            
        Returns:
            mean: 动作均值 [batch_size, action_dim]
            log_std: 动作对数标准差 [batch_size, action_dim]
        """
        features = self.feature_layers(state)
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        
        # 限制对数标准差范围
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        采样动作
        
        Args:
            state: 状态
            deterministic: 是否确定性采样
            
        Returns:
            action: 动作
            log_prob: 对数概率
        """
        mean, log_std = self.forward(state)
        
        if deterministic:
            # 确定性动作
            action = torch.tanh(mean) * self.max_action
            log_prob = torch.zeros_like(action).sum(dim=-1, keepdim=True)
        else:
            # 随机采样
            std = torch.exp(log_std)
            normal = Normal(mean, std)
            
            # 重参数化采样
            x = normal.rsample()  # 使用rsample以支持梯度传播
            action = torch.tanh(x) * self.max_action
            
            # 计算对数概率（考虑tanh变换）
            log_prob = normal.log_prob(x)
            # 修正tanh变换的jacobian
            log_prob -= torch.log(self.max_action * (1 - torch.tanh(x).pow(2)) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob
    
    def evaluate_action(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        评估动作的对数概率和熵
        
        Args:
            state: 状态
            action: 动作
            
        Returns:
            log_prob: 对数概率
            entropy: 熵
        """
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        normal = Normal(mean, std)
        
        # 反tanh变换
        action_scaled = action / self.max_action
        x = 0.5 * torch.log((1 + action_scaled + 1e-6) / (1 - action_scaled + 1e-6))
        
        # 计算对数概率
        log_prob = normal.log_prob(x)
        # 修正tanh变换的jacobian
        log_prob -= torch.log(self.max_action * (1 - action_scaled.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        # 计算熵
        entropy = normal.entropy().sum(dim=-1, keepdim=True)
        
        return log_prob, entropy


class CriticNetwork(nn.Module):
    """SAC Critic网络 - Q网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(CriticNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Q网络
        self.q_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state: 状态 [batch_size, state_dim]
            action: 动作 [batch_size, action_dim]
            
        Returns:
            q_value: Q值 [batch_size, 1]
        """
        q_input = torch.cat([state, action], dim=-1)
        q_value = self.q_network(q_input)
        return q_value


class SAC_v2_Networks:
    """SAC v2网络集合"""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int, 
                 hidden_dim: int = 256,
                 max_action: float = 1.0,
                 device: torch.device = torch.device('cpu')):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.device = device
        
        # Actor网络
        self.actor = ActorNetwork(
            state_dim, action_dim, hidden_dim, max_action
        ).to(device)
        
        # 两个Critic网络（减少估计偏差）
        self.critic1 = CriticNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.critic2 = CriticNetwork(state_dim, action_dim, hidden_dim).to(device)
        
        # 目标Critic网络（软更新）
        self.target_critic1 = CriticNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_critic2 = CriticNetwork(state_dim, action_dim, hidden_dim).to(device)
        
        # 初始化目标网络
        self.soft_update_target_networks(tau=1.0)
        
        # 自动熵调节
        self.target_entropy = -action_dim  # 启发式目标熵
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        
        print(f"🎭 SAC v2 Networks initialized")
        print(f"   State dim: {state_dim}, Action dim: {action_dim}")
        print(f"   Hidden dim: {hidden_dim}, Max action: {max_action}")
        print(f"   Target entropy: {self.target_entropy}")
    
    @property
    def alpha(self):
        """获取当前熵系数"""
        return self.log_alpha.exp()
    
    def soft_update_target_networks(self, tau: float = 0.005):
        """软更新目标网络"""
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def create_sac_v2_networks(state_space, action_space, network_config: Dict[str, Any] = None):
    """
    工厂函数：创建SAC v2网络
    
    Args:
        state_space: 状态空间
        action_space: 动作空间
        network_config: 网络配置
        
    Returns:
        SAC v2网络集合
    """
    default_config = {
        'hidden_dim': 256,
        'max_action': 1.0
    }
    
    if network_config:
        default_config.update(network_config)
    
    # 获取状态和动作维度
    if len(state_space.shape) == 1:
        state_dim = state_space.shape[0]
    else:
        raise ValueError(f"SAC v2 only supports 1D state space, got {state_space.shape}")
    
    if len(action_space.shape) == 1:
        action_dim = action_space.shape[0]
        max_action = float(action_space.high[0])  # 假设所有维度相同
    else:
        raise ValueError(f"SAC v2 only supports 1D action space, got {action_space.shape}")
    
    return SAC_v2_Networks(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=default_config['hidden_dim'],
        max_action=max_action
    )