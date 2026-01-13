"""
TD7 Neural Networks
TD7算法的神经网络架构，包括Actor、Critic和SALE编码器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any, Optional


class StateEncoder(nn.Module):
    """SALE状态编码器 - 学习状态表示"""
    
    def __init__(self,
                 state_dim: int,
                 embedding_dim: int = 256,
                 hidden_dim: int = 256):
        super(StateEncoder, self).__init__()
        
        self.state_dim = state_dim
        self.embedding_dim = embedding_dim
        
        # 状态编码网络
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # AvgL1Norm归一化（TD7特色）
        self.normalize = True
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state: 状态 [batch_size, state_dim]
            
        Returns:
            embedding: 状态嵌入 [batch_size, embedding_dim]
        """
        embedding = self.encoder(state)
        
        # AvgL1Norm归一化
        if self.normalize:
            # 计算L1范数并归一化
            l1_norm = torch.mean(torch.abs(embedding), dim=-1, keepdim=True)
            embedding = embedding / (l1_norm + 1e-8)
        
        return embedding


class TD7_Actor(nn.Module):
    """TD7 Actor网络"""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 embedding_dim: int = 256,
                 hidden_dim: int = 256,
                 max_action: float = 1.0):
        super(TD7_Actor, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim
        self.max_action = max_action
        
        # 使用状态嵌入作为输入
        self.policy_network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        
        # 最后一层小初始化
        nn.init.uniform_(self.policy_network[-2].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.policy_network[-2].bias, -3e-3, 3e-3)
    
    def forward(self, state_embedding: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state_embedding: 状态嵌入 [batch_size, embedding_dim]
            
        Returns:
            action: 动作 [batch_size, action_dim]
        """
        action = self.policy_network(state_embedding)
        return action * self.max_action


class TD7_Critic(nn.Module):
    """TD7 Critic网络 - 双Q网络"""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 embedding_dim: int = 256,
                 hidden_dim: int = 256):
        super(TD7_Critic, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim
        
        # Q网络1 - 使用状态嵌入+动作
        self.q1_network = nn.Sequential(
            nn.Linear(embedding_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Q网络2 - 双Q网络减少估计偏差
        self.q2_network = nn.Sequential(
            nn.Linear(embedding_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, 
                state_embedding: torch.Tensor, 
                action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            state_embedding: 状态嵌入 [batch_size, embedding_dim]
            action: 动作 [batch_size, action_dim]
            
        Returns:
            q1_value: Q1值 [batch_size, 1]
            q2_value: Q2值 [batch_size, 1]
        """
        q_input = torch.cat([state_embedding, action], dim=-1)
        
        q1_value = self.q1_network(q_input)
        q2_value = self.q2_network(q_input)
        
        return q1_value, q2_value
    
    def q1(self, state_embedding: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """只返回Q1值（用于Actor更新）"""
        q_input = torch.cat([state_embedding, action], dim=-1)
        return self.q1_network(q_input)


class TD7_Networks:
    """TD7网络集合"""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 embedding_dim: int = 256,
                 hidden_dim: int = 256,
                 max_action: float = 1.0,
                 device: torch.device = torch.device('cpu')):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim
        self.max_action = max_action
        self.device = device
        
        # 状态编码器
        self.state_encoder = StateEncoder(
            state_dim, embedding_dim, hidden_dim
        ).to(device)
        
        # Actor网络
        self.actor = TD7_Actor(
            state_dim, action_dim, embedding_dim, hidden_dim, max_action
        ).to(device)
        
        # Critic网络
        self.critic = TD7_Critic(
            state_dim, action_dim, embedding_dim, hidden_dim
        ).to(device)
        
        # 目标网络
        self.target_state_encoder = StateEncoder(
            state_dim, embedding_dim, hidden_dim
        ).to(device)
        
        self.target_actor = TD7_Actor(
            state_dim, action_dim, embedding_dim, hidden_dim, max_action
        ).to(device)
        
        self.target_critic = TD7_Critic(
            state_dim, action_dim, embedding_dim, hidden_dim
        ).to(device)
        
        # 初始化目标网络
        self.soft_update_target_networks(tau=1.0)
        
        print(f"🎯 TD7 Networks initialized")
        print(f"   State dim: {state_dim}, Action dim: {action_dim}")
        print(f"   Embedding dim: {embedding_dim}, Hidden dim: {hidden_dim}")
        print(f"   Max action: {max_action}")
    
    def soft_update_target_networks(self, tau: float = 0.005):
        """软更新目标网络"""
        # 更新状态编码器
        for target_param, param in zip(self.target_state_encoder.parameters(), 
                                     self.state_encoder.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        # 更新Actor
        for target_param, param in zip(self.target_actor.parameters(), 
                                     self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        # 更新Critic
        for target_param, param in zip(self.target_critic.parameters(), 
                                     self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def create_td7_networks(state_space, action_space, network_config: Dict[str, Any] = None):
    """
    工厂函数：创建TD7网络
    
    Args:
        state_space: 状态空间
        action_space: 动作空间
        network_config: 网络配置
        
    Returns:
        TD7网络集合
    """
    default_config = {
        'embedding_dim': 256,
        'hidden_dim': 256,
        'max_action': 1.0
    }
    
    if network_config:
        default_config.update(network_config)
    
    # 获取状态和动作维度
    if len(state_space.shape) == 1:
        state_dim = state_space.shape[0]
    else:
        raise ValueError(f"TD7 only supports 1D state space, got {state_space.shape}")
    
    if len(action_space.shape) == 1:
        action_dim = action_space.shape[0]
        max_action = float(action_space.high[0])  # 假设所有维度相同
    else:
        raise ValueError(f"TD7 only supports 1D action space, got {action_space.shape}")
    
    return TD7_Networks(
        state_dim=state_dim,
        action_dim=action_dim,
        embedding_dim=default_config['embedding_dim'],
        hidden_dim=default_config['hidden_dim'],
        max_action=max_action
    )