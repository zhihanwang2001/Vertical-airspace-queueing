"""
IMPALA Neural Networks
IMPALA算法的神经网络架构，包括Actor-Critic网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any


class IMPALANetwork(nn.Module):
    """IMPALA Actor-Critic网络"""
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int, 
                 hidden_dim: int = 512,
                 num_layers: int = 2):
        super(IMPALANetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # 共享特征提取层
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        self.shared_layers = nn.Sequential(*layers)
        
        # Actor头：输出动作的均值和标准差（连续动作）
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_logstd = nn.Linear(hidden_dim, action_dim)
        
        # Critic头：输出状态价值
        self.critic = nn.Linear(hidden_dim, 1)
        
        # 初始化参数
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        
        # Actor输出层使用小的初始化值
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.actor_logstd.weight, gain=0.01)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            state: 状态张量 [batch_size, state_dim]
            
        Returns:
            action_mean: 动作均值 [batch_size, action_dim]
            action_logstd: 动作log标准差 [batch_size, action_dim] 
            value: 状态价值 [batch_size, 1]
        """
        # 共享特征提取
        features = self.shared_layers(state)
        
        # Actor输出
        action_mean = self.actor_mean(features)
        action_logstd = self.actor_logstd(features)
        # 限制log_std的范围
        action_logstd = torch.clamp(action_logstd, -10, 2)
        
        # Critic输出
        value = self.critic(features)
        
        return action_mean, action_logstd, value
    
    def get_action_and_value(self, state: torch.Tensor, deterministic: bool = False):
        """
        获取动作和价值
        
        Args:
            state: 状态张量
            deterministic: 是否使用确定性策略
            
        Returns:
            action: 采样的动作
            log_prob: 动作的对数概率
            value: 状态价值
        """
        action_mean, action_logstd, value = self.forward(state)
        
        if deterministic:
            action = action_mean
            log_prob = torch.zeros_like(action_mean).sum(dim=-1, keepdim=True)
        else:
            # 创建正态分布
            std = torch.exp(action_logstd)
            dist = torch.distributions.Normal(action_mean, std)
            
            # 采样动作
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        return action, log_prob, value
    
    def evaluate_action(self, state: torch.Tensor, action: torch.Tensor):
        """
        评估给定状态和动作的价值和概率
        
        Args:
            state: 状态张量
            action: 动作张量
            
        Returns:
            log_prob: 动作的对数概率
            value: 状态价值
            entropy: 策略熵
        """
        action_mean, action_logstd, value = self.forward(state)
        
        # 创建分布
        std = torch.exp(action_logstd)
        dist = torch.distributions.Normal(action_mean, std)
        
        # 计算对数概率
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        # 计算熵
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return log_prob, value, entropy


def create_impala_network(state_space, action_space, network_config: Dict[str, Any] = None):
    """
    工厂函数：创建IMPALA网络
    
    Args:
        state_space: 状态空间
        action_space: 动作空间
        network_config: 网络配置
        
    Returns:
        IMPALA网络实例
    """
    default_config = {
        'hidden_dim': 512,
        'num_layers': 2
    }
    
    if network_config:
        default_config.update(network_config)
    
    # 获取状态和动作维度
    if len(state_space.shape) == 1:
        state_dim = state_space.shape[0]
    else:
        raise ValueError(f"Unsupported state space shape: {state_space.shape}")
    
    if hasattr(action_space, 'n'):
        raise ValueError("IMPALA currently only supports continuous action spaces")
    else:
        action_dim = action_space.shape[0]
    
    return IMPALANetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        **default_config
    )