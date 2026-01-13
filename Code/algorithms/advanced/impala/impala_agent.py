"""
IMPALA Agent Implementation
IMPALA智能体实现，集成V-trace和Actor-Critic架构
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional, List, Any
import random
from collections import deque

from .networks import create_impala_network
from .replay_buffer import IMPALAReplayBuffer
from .vtrace import VTrace, compute_vtrace_loss


class IMPALAAgent:
    """IMPALA智能体"""
    
    def __init__(self,
                 state_space,
                 action_space,
                 config: Dict = None):
        """
        初始化IMPALA智能体
        
        Args:
            state_space: 状态空间
            action_space: 动作空间
            config: 配置参数
        """
        
        # 优化配置 - 保守V-trace策略防止早期崩溃
        default_config = {
            # 网络配置
            'hidden_dim': 512,
            'num_layers': 2,

            # 学习参数 - 进一步降低学习率防止late-stage崩溃
            'learning_rate': 3e-5,      # 🔧 优化v2: 5e-5 → 3e-5 (防止150k步崩溃)
            'gamma': 0.99,
            'entropy_coeff': 0.01,
            'value_loss_coeff': 0.5,
            'gradient_clip': 20.0,      # 🔧 优化: 40.0 → 20.0 (更强梯度裁剪)

            # V-trace参数 - 极度保守避免重要性采样爆炸
            'rho_bar': 0.7,             # 🔧 优化v2: 0.9 → 0.7 (更保守的IS裁剪)
            'c_bar': 0.7,               # 🔧 优化v2: 0.9 → 0.7 (更保守的value裁剪)

            # 回放缓冲区 - 减小缓冲区降低策略陈旧性
            'buffer_size': 30000,       # 🔧 优化v2: 50000 → 30000 (减少off-policy程度)
            'sequence_length': 10,      # 🔧 优化: 20 → 10 (缩短序列长度提高稳定性)
            'batch_size': 32,           # 🔧 优化: 16 → 32 (增加批次大小)

            # 训练参数 - 更频繁更新但延后启动
            'learning_starts': 2000,    # 🔧 优化: 1000 → 2000 (延后学习积累更多经验)
            'train_freq': 2,            # 🔧 优化: 4 → 2 (更频繁训练)
            'update_freq': 100,

            # 其他
            'seed': 42,
            'device': 'auto'
        }
        
        if config:
            default_config.update(config)
        self.config = default_config
        
        # 设置设备
        if self.config['device'] == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.config['device'])
        
        # 设置随机种子
        if self.config['seed'] is not None:
            random.seed(self.config['seed'])
            np.random.seed(self.config['seed'])
            torch.manual_seed(self.config['seed'])
        
        self.state_space = state_space
        self.action_space = action_space
        
        # 创建网络
        network_config = {
            'hidden_dim': self.config['hidden_dim'],
            'num_layers': self.config['num_layers']
        }
        
        self.network = create_impala_network(
            state_space, action_space, network_config
        ).to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.config['learning_rate']
        )
        
        # 经验回放缓冲区
        self.replay_buffer = IMPALAReplayBuffer(
            capacity=self.config['buffer_size'],
            sequence_length=self.config['sequence_length'],
            device=self.device
        )
        
        # V-trace
        self.vtrace = VTrace(
            rho_bar=self.config['rho_bar'],
            c_bar=self.config['c_bar'],
            gamma=self.config['gamma']
        )
        
        # 训练统计
        self.training_step = 0
        self.episode_rewards = []
        self.losses = []
        
        # 当前episode的行为策略log_probs（用于V-trace）
        self.behavior_log_probs = []
        
        print(f"🎯 IMPALA Agent initialized on {self.device}")
        print(f"   State space: {state_space.shape}")
        print(f"   Action space: {action_space.shape}")
        print(f"   Network: Actor-Critic with V-trace")
        print(f"   Sequence length: {self.config['sequence_length']}")
    
    def act(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """选择动作"""
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        
        # 转换为tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 获取动作和价值
            action, log_prob, value = self.network.get_action_and_value(
                state_tensor, deterministic=not training
            )
            
            action = action.cpu().numpy()[0]
            log_prob = log_prob.cpu().numpy()[0]
            value = value.cpu().numpy()[0]
        
        # 存储行为策略的log_prob用于V-trace
        if training:
            self.behavior_log_probs.append(log_prob[0])  # 取出标量值
        
        # 确保动作在有效范围内
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        return action
    
    def store_transition(self,
                        state: np.ndarray,
                        action: np.ndarray,
                        reward: float,
                        next_state: np.ndarray,
                        done: bool):
        """存储转换到缓冲区"""
        
        # 获取当前状态的价值估计
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            _, _, value = self.network.get_action_and_value(state_tensor)
            value = value.cpu().numpy()[0, 0]
        
        # 获取行为策略的log_prob
        behavior_log_prob = self.behavior_log_probs[-1] if self.behavior_log_probs else 0.0
        
        # 存储到回放缓冲区
        self.replay_buffer.add_step(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            log_prob=behavior_log_prob,
            value=value
        )
        
        # 如果episode结束，清空行为策略log_probs
        if done:
            self.behavior_log_probs.clear()
    
    def train(self) -> Optional[Dict]:
        """训练一步"""
        if not self.replay_buffer.is_ready:
            return None
        
        if self.training_step % self.config['train_freq'] != 0:
            self.training_step += 1
            return None
        
        # 采样序列批次
        batch = self.replay_buffer.sample_sequences(self.config['batch_size'])
        if batch is None:
            self.training_step += 1
            return None
        
        # 偶尔检查批次形状（减少输出频率）
        if self.training_step % 5000 == 0:
            print(f"Debug: Batch shapes - states: {batch['states'].shape}, actions: {batch['actions'].shape}")
        
        # 解包批次数据
        states = batch['states']  # [T, B, state_dim]
        actions = batch['actions']  # [T, B, action_dim]
        rewards = batch['rewards']  # [T, B]
        dones = batch['dones']  # [T, B]
        behavior_log_probs = batch['log_probs']  # [T, B]
        
        T, B = states.shape[:2]
        
        # 前向传播获取当前策略的输出
        states_flat = states.reshape(-1, states.shape[-1])
        actions_flat = actions.reshape(-1, actions.shape[-1])
        
        target_log_probs_flat, values_flat, entropies_flat = self.network.evaluate_action(
            states_flat, actions_flat
        )
        
        # 重新整形
        target_log_probs = target_log_probs_flat.reshape(T, B)
        values = values_flat.reshape(T, B)
        entropies = entropies_flat.reshape(T, B)
        
        # 计算bootstrap价值（最后一个状态的价值）
        with torch.no_grad():
            last_states = states[-1]  # [B, state_dim]
            _, _, bootstrap_values = self.network.get_action_and_value(last_states)
            bootstrap_values = bootstrap_values.squeeze(-1)  # [B]
        
        # 计算V-trace损失
        total_loss, loss_info = compute_vtrace_loss(
            self.vtrace,
            behavior_log_probs,
            target_log_probs,
            rewards,
            values,
            bootstrap_values,
            dones,
            entropies,
            entropy_coeff=self.config['entropy_coeff']
        )
        
        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            self.network.parameters(),
            self.config['gradient_clip']
        )
        
        self.optimizer.step()
        
        self.training_step += 1
        self.losses.append(loss_info['total_loss'])
        
        # 返回训练信息
        return {
            'total_loss': loss_info['total_loss'],
            'pg_loss': loss_info['pg_loss'],
            'value_loss': loss_info['value_loss'],
            'entropy_loss': loss_info['entropy_loss'],
            'mean_advantage': loss_info['mean_advantage'],
            'mean_value': loss_info['mean_value'],
            'buffer_size': len(self.replay_buffer)
        }
    
    def save(self, filepath: str):
        """保存模型"""
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config,
            'training_step': self.training_step
        }, filepath)
    
    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.training_step = checkpoint['training_step']
        
        print(f"✅ IMPALA model loaded from {filepath}")
    
    def get_stats(self) -> Dict:
        """获取训练统计"""
        return {
            'training_step': self.training_step,
            'buffer_size': len(self.replay_buffer),
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0,
            'episodes_trained': len(self.episode_rewards)
        }