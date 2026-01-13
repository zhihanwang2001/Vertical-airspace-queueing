"""
V-trace Implementation
V-trace重要性采样校正算法的实现
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, NamedTuple


class VTraceReturns(NamedTuple):
    """V-trace返回值"""
    vs: torch.Tensor  # V-trace价值目标
    pg_advantages: torch.Tensor  # 策略梯度优势


class VTrace:
    """V-trace算法实现"""
    
    def __init__(self, 
                 rho_bar: float = 1.0,
                 c_bar: float = 1.0,
                 gamma: float = 0.99):
        """
        初始化V-trace
        
        Args:
            rho_bar: 重要性权重截断阈值
            c_bar: 时间差分误差权重截断阈值  
            gamma: 折扣因子
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
        计算V-trace目标
        
        Args:
            behavior_log_probs: 行为策略的对数概率 [T, B]
            target_log_probs: 目标策略的对数概率 [T, B]
            rewards: 奖励序列 [T, B] 
            values: 价值估计 [T, B]
            bootstrap_value: 引导价值 [B]
            dones: 是否结束 [T, B]
            
        Returns:
            VTraceReturns包含vs和pg_advantages
        """
        # 计算重要性权重
        log_rhos = target_log_probs - behavior_log_probs
        rhos = torch.exp(log_rhos)
        
        # 截断重要性权重
        clipped_rhos = torch.clamp(rhos, max=self.rho_bar)
        cs = torch.clamp(rhos, max=self.c_bar)
        
        # 调整维度以便计算
        T, B = rewards.shape
        
        # 计算时间差分误差
        values_t_plus_1 = torch.cat([values[1:], bootstrap_value.unsqueeze(0)], dim=0)
        deltas = clipped_rhos * (rewards + self.gamma * values_t_plus_1 * (1 - dones) - values)
        
        # 反向计算V-trace价值
        vs_minus_v_xs = []
        vs_minus_v_x = torch.zeros_like(bootstrap_value)
        
        # 从最后一个时间步开始反向计算
        for t in reversed(range(T)):
            vs_minus_v_x = deltas[t] + self.gamma * cs[t] * (1 - dones[t]) * vs_minus_v_x
            vs_minus_v_xs.append(vs_minus_v_x)
        
        # 反转列表得到正向顺序
        vs_minus_v_xs.reverse()
        vs_minus_v_xs = torch.stack(vs_minus_v_xs, dim=0)
        
        # 计算V-trace价值目标
        vs = values + vs_minus_v_xs
        
        # 计算策略梯度优势
        vs_t_plus_1 = torch.cat([vs[1:], bootstrap_value.unsqueeze(0)], dim=0)
        pg_advantages = clipped_rhos * (rewards + self.gamma * vs_t_plus_1 * (1 - dones) - values)
        
        return VTraceReturns(vs=vs, pg_advantages=pg_advantages)
    
    def compute_policy_gradient_loss(self,
                                     log_probs: torch.Tensor,
                                     advantages: torch.Tensor) -> torch.Tensor:
        """
        计算策略梯度损失
        
        Args:
            log_probs: 动作对数概率 [T, B]
            advantages: V-trace优势 [T, B]
            
        Returns:
            策略梯度损失
        """
        return -(log_probs * advantages.detach()).mean()
    
    def compute_value_loss(self,
                           values: torch.Tensor,
                           vs_targets: torch.Tensor) -> torch.Tensor:
        """
        计算价值函数损失
        
        Args:
            values: 预测价值 [T, B]
            vs_targets: V-trace目标价值 [T, B]
            
        Returns:
            价值函数损失（MSE）
        """
        return F.mse_loss(values, vs_targets.detach())
    
    def compute_entropy_loss(self,
                             entropies: torch.Tensor) -> torch.Tensor:
        """
        计算熵损失（用于探索）
        
        Args:
            entropies: 策略熵 [T, B]
            
        Returns:
            熵损失（负熵，用于最大化熵）
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
    计算完整的IMPALA损失
    
    Args:
        vtrace: V-trace实例
        behavior_log_probs: 行为策略对数概率
        target_log_probs: 目标策略对数概率
        rewards: 奖励序列
        values: 价值估计
        bootstrap_value: 引导价值
        dones: 是否结束
        entropies: 策略熵
        entropy_coeff: 熵系数
        
    Returns:
        total_loss: 总损失
        loss_info: 损失信息字典
    """
    # 计算V-trace目标
    vtrace_returns = vtrace.compute_vtrace_targets(
        behavior_log_probs, target_log_probs, rewards, values, bootstrap_value, dones
    )
    
    # 计算各项损失
    pg_loss = vtrace.compute_policy_gradient_loss(target_log_probs, vtrace_returns.pg_advantages)
    value_loss = vtrace.compute_value_loss(values, vtrace_returns.vs)
    entropy_loss = vtrace.compute_entropy_loss(entropies)
    
    # 总损失
    total_loss = pg_loss + 0.5 * value_loss + entropy_coeff * entropy_loss
    
    # 损失信息
    loss_info = {
        'total_loss': total_loss.item(),
        'pg_loss': pg_loss.item(),
        'value_loss': value_loss.item(),
        'entropy_loss': entropy_loss.item(),
        'mean_advantage': vtrace_returns.pg_advantages.mean().item(),
        'mean_value': values.mean().item()
    }
    
    return total_loss, loss_info