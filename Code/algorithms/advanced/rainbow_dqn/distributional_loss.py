"""
Distributional Loss for Rainbow DQN (C51算法)
实现分布式强化学习的损失函数
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class DistributionalLoss:
    """C51分布式损失函数"""
    
    def __init__(self, 
                 num_atoms: int = 51, 
                 v_min: float = -10, 
                 v_max: float = 10, 
                 gamma: float = 0.99):
        """
        Args:
            num_atoms: 价值分布的原子数量
            v_min: 价值分布的最小值
            v_max: 价值分布的最大值  
            gamma: 折扣因子
        """
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.gamma = gamma
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        
        # 支撑点
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
        计算分布式损失
        
        Args:
            q_dist: 当前状态的Q分布 [batch_size, action_dim, num_atoms]
            actions: 选择的动作 [batch_size]
            rewards: 即时奖励 [batch_size] 
            next_q_dist: 下一状态的Q分布 [batch_size, action_dim, num_atoms]
            next_actions: 下一状态的动作（Double DQN）[batch_size]
            dones: 是否结束 [batch_size]
            weights: 重要性权重 [batch_size]
            
        Returns:
            loss: 损失值
            td_errors: TD错误（用于更新优先级）
        """
        batch_size = q_dist.size(0)
        device = q_dist.device
        
        # 确保supports在正确的设备上
        if self.supports.device != device:
            self.supports = self.supports.to(device)
        
        # 获取当前状态-动作对的Q分布
        current_dist = q_dist[range(batch_size), actions]  # [batch_size, num_atoms]
        
        # 获取下一状态的Q分布（Double DQN）
        next_dist = next_q_dist[range(batch_size), next_actions]  # [batch_size, num_atoms]
        
        # 计算目标分布
        target_dist = self._compute_target_distribution(
            rewards, next_dist, dones, device
        )
        
        # 计算KL散度损失
        loss_per_sample = -torch.sum(target_dist * torch.log(current_dist + 1e-8), dim=1)
        
        # 应用重要性权重
        if weights is not None:
            weights = torch.FloatTensor(weights).to(device)
            loss_per_sample = loss_per_sample * weights
        
        loss = loss_per_sample.mean()
        
        # 计算TD错误（用于更新优先级）
        with torch.no_grad():
            # 使用分布的期望值计算TD错误
            current_q = torch.sum(current_dist * self.supports, dim=1)
            target_q = torch.sum(target_dist * self.supports, dim=1)
            td_errors = torch.abs(current_q - target_q).cpu().numpy()
        
        return loss, td_errors
    
    def _compute_target_distribution(self, 
                                   rewards: torch.Tensor,
                                   next_dist: torch.Tensor, 
                                   dones: torch.Tensor,
                                   device: torch.device) -> torch.Tensor:
        """计算目标分布"""
        batch_size = rewards.size(0)
        
        # 计算目标支撑点
        target_supports = rewards.unsqueeze(1) + self.gamma * self.supports.unsqueeze(0) * (~dones).unsqueeze(1).float()
        
        # 将目标支撑点投影到原始支撑点上
        target_supports = torch.clamp(target_supports, self.v_min, self.v_max)
        
        # 计算投影
        target_dist = torch.zeros(batch_size, self.num_atoms, device=device)
        
        # 计算每个目标支撑点对应的索引
        b = (target_supports - self.v_min) / self.delta_z
        l = b.floor().long()
        u = b.ceil().long()
        
        # 处理边界情况
        l[(u > 0) * (l == u)] -= 1
        u[(l < (self.num_atoms - 1)) * (l == u)] += 1
        
        # 分布概率投影
        for i in range(batch_size):
            for j in range(self.num_atoms):
                if not dones[i]:
                    # 不是终止状态，进行投影
                    l_idx = l[i, j]
                    u_idx = u[i, j]
                    
                    # 下界投影
                    target_dist[i, l_idx] += next_dist[i, j] * (u[i, j].float() - b[i, j])
                    # 上界投影  
                    target_dist[i, u_idx] += next_dist[i, j] * (b[i, j] - l[i, j].float())
                else:
                    # 终止状态，所有概率集中在奖励值上
                    reward_idx = int((rewards[i] - self.v_min) / self.delta_z)
                    reward_idx = max(0, min(self.num_atoms - 1, reward_idx))
                    target_dist[i, reward_idx] = 1.0
        
        return target_dist
    
    def q_values_from_distribution(self, q_dist: torch.Tensor) -> torch.Tensor:
        """从分布计算Q值（期望值）"""
        if self.supports.device != q_dist.device:
            self.supports = self.supports.to(q_dist.device)
        
        # 计算每个动作的期望Q值
        q_values = torch.sum(q_dist * self.supports, dim=-1)
        return q_values


class CategoricalDQNLoss:
    """Categorical DQN损失（C51的简化版本）"""
    
    def __init__(self, num_atoms: int = 51, v_min: float = -10, v_max: float = 10):
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (num_atoms - 1)
        self.supports = torch.linspace(v_min, v_max, num_atoms)
    
    def compute_loss(self, pred_dist: torch.Tensor, target_dist: torch.Tensor) -> torch.Tensor:
        """计算Categorical损失"""
        # 交叉熵损失
        loss = -torch.sum(target_dist * torch.log(pred_dist + 1e-8), dim=-1)
        return loss.mean()


class QuantileRegressionLoss:
    """分位数回归损失（用于IQN/QR-DQN）"""
    
    def __init__(self, kappa: float = 1.0):
        self.kappa = kappa
    
    def compute_loss(self, 
                     pred_quantiles: torch.Tensor,
                     target_quantiles: torch.Tensor, 
                     tau: torch.Tensor) -> torch.Tensor:
        """
        计算分位数回归损失
        
        Args:
            pred_quantiles: 预测的分位数 [batch_size, n_quantiles]
            target_quantiles: 目标分位数 [batch_size, n_quantiles]
            tau: 分位数水平 [n_quantiles]
        """
        # 计算分位数损失
        u = target_quantiles - pred_quantiles
        loss = (tau - (u < 0).float()) * self._huber_loss(u)
        
        return loss.mean()
    
    def _huber_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Huber损失"""
        return torch.where(
            x.abs() <= self.kappa,
            0.5 * x.pow(2),
            self.kappa * (x.abs() - 0.5 * self.kappa)
        )


def create_distributional_loss(loss_type: str = "c51", **kwargs):
    """工厂函数：创建分布式损失函数"""
    if loss_type == "c51":
        return DistributionalLoss(**kwargs)
    elif loss_type == "categorical":
        return CategoricalDQNLoss(**kwargs) 
    elif loss_type == "quantile":
        return QuantileRegressionLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")