"""
R2D2 Agent Implementation
R2D2智能体实现，集成循环网络和序列经验回放
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List, Any
import random
import copy
from collections import deque

from .networks import create_r2d2_network
from .sequence_replay import R2D2SequenceReplayBuffer


class R2D2Agent:
    """R2D2智能体"""
    
    def __init__(self,
                 state_space,
                 action_space,
                 config: Dict = None):
        """
        初始化R2D2智能体
        
        Args:
            state_space: 状态空间
            action_space: 动作空间
            config: 配置参数
        """
        
        # 默认配置
        default_config = {
            # 网络配置
            'hidden_dim': 512,
            'recurrent_dim': 256,
            'num_layers': 1,
            'recurrent_type': 'LSTM',
            'dueling': True,
            
            # 学习参数
            'learning_rate': 1e-4,
            'gamma': 0.99,
            'target_update_freq': 2500,
            'gradient_clip': 40.0,
            
            # DQN参数
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay_steps': 250000,
            'double_dqn': True,
            
            # 序列回放配置
            'buffer_size': 5000,
            'sequence_length': 40,
            'burn_in_length': 20,
            'overlap_length': 10,
            'batch_size': 16,
            
            # 训练参数
            'learning_starts': 5000,
            'train_freq': 4,
            
            # 动作离散化（连续动作空间）
            'action_bins': 3,
            
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
        
        # 处理动作空间
        self._setup_action_space()
        
        # 创建网络
        network_config = {
            'hidden_dim': self.config['hidden_dim'],
            'recurrent_dim': self.config['recurrent_dim'],
            'num_layers': self.config['num_layers'],
            'recurrent_type': self.config['recurrent_type'],
            'dueling': self.config['dueling'],
            'action_bins': self.config['action_bins']
        }
        
        self.q_network = create_r2d2_network(
            state_space, action_space, network_config
        ).to(self.device)
        
        self.target_network = create_r2d2_network(
            state_space, action_space, network_config  
        ).to(self.device)
        
        # 同步目标网络
        self.update_target_network()
        
        # 优化器
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=self.config['learning_rate']
        )
        
        # 序列经验回放缓冲区
        self.replay_buffer = R2D2SequenceReplayBuffer(
            capacity=self.config['buffer_size'],
            sequence_length=self.config['sequence_length'],
            burn_in_length=self.config['burn_in_length'],
            overlap_length=self.config['overlap_length'],
            device=self.device
        )
        
        # 训练统计
        self.training_step = 0
        self.episode_rewards = []
        self.losses = []
        
        # RNN状态管理
        self.current_hidden_state = None
        self.reset_hidden_state()
        
        print(f"🔄 R2D2 Agent initialized on {self.device}")
        print(f"   State space: {state_space.shape}")
        if self.action_type == 'discrete':
            print(f"   Action space: {self.num_actions} (discrete)")
        else:
            print(f"   Action space: {self.action_dim}D -> {self.num_actions} discrete")
        print(f"   Recurrent: {self.config['recurrent_type']} ({self.config['recurrent_dim']} units)")
        print(f"   Sequence length: {self.config['sequence_length']} + {self.config['burn_in_length']} burn-in")
    
    def _setup_action_space(self):
        """设置动作空间"""
        if hasattr(self.action_space, 'n'):
            # 离散动作空间
            self.num_actions = self.action_space.n
            self.action_type = 'discrete'
            self.action_dim = None
        else:
            # 连续动作空间，需要离散化
            self.action_dim = self.action_space.shape[0]
            self.action_low = self.action_space.low
            self.action_high = self.action_space.high
            self.action_bins = self.config['action_bins']
            self.num_actions = self.action_bins ** self.action_dim
            self.action_type = 'continuous'
            
            # 创建动作映射
            self._create_action_mapping()
    
    def _create_action_mapping(self):
        """为连续动作空间创建智能离散化映射"""
        if self.action_type == 'discrete':
            return
        
        # 智能离散化：只使用关键动作值
        # 对于大多数控制任务，{-1, 0, 1} 或者 {-0.5, 0, 0.5} 就够用了
        self.action_grids = []
        for i in range(self.action_dim):
            if self.action_bins == 2:
                # 二进制控制：只有负值和正值
                grid = np.array([self.action_low[i], self.action_high[i]])
            elif self.action_bins == 3:
                # 三值控制：负值、零、正值
                grid = np.array([self.action_low[i], 0.0, self.action_high[i]])
            else:
                # 保持原来的线性分布
                grid = np.linspace(self.action_low[i], self.action_high[i], self.action_bins)
            self.action_grids.append(grid)
            
        print(f"🎯 R2D2 Action discretization: {self.action_bins}^{self.action_dim} = {self.num_actions} actions")
        print(f"   First dimension grid: {self.action_grids[0]}")
    
    def _discrete_to_continuous_action(self, discrete_action: int) -> np.ndarray:
        """将离散动作转换为连续动作"""
        if self.action_type == 'discrete':
            return discrete_action
        
        continuous_action = np.zeros(self.action_dim)
        remaining = discrete_action
        
        for i in range(self.action_dim):
            idx = remaining % self.action_bins
            continuous_action[i] = self.action_grids[i][idx]
            remaining //= self.action_bins
        
        return continuous_action
    
    def reset_hidden_state(self):
        """重置RNN隐藏状态"""
        self.current_hidden_state = self.q_network.init_hidden_state(1, self.device)
    
    def get_epsilon(self) -> float:
        """获取当前epsilon值"""
        if self.training_step < self.config['epsilon_decay_steps']:
            epsilon = self.config['epsilon_start'] - (
                self.config['epsilon_start'] - self.config['epsilon_end']
            ) * self.training_step / self.config['epsilon_decay_steps']
        else:
            epsilon = self.config['epsilon_end']
        return epsilon
    
    def act(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """选择动作"""
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        
        # 转换为tensor并添加批次维度和序列维度
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # epsilon-greedy策略
        if training and random.random() < self.get_epsilon():
            # 随机动作
            discrete_action = random.randint(0, self.num_actions - 1)
            # 仍需要更新隐藏状态
            with torch.no_grad():
                _, self.current_hidden_state = self.q_network(state_tensor, self.current_hidden_state)
        else:
            # 贪心动作
            with torch.no_grad():
                q_values, self.current_hidden_state = self.q_network(state_tensor, self.current_hidden_state)
                discrete_action = q_values.squeeze(0).squeeze(0).argmax().item()
        
        # 转换为环境所需的动作格式
        if self.action_type == 'continuous':
            return self._discrete_to_continuous_action(discrete_action)
        else:
            return discrete_action
    
    def store_transition(self,
                        state: np.ndarray,
                        action,
                        reward: float,
                        next_state: np.ndarray,
                        done: bool):
        """存储转换到序列缓冲区"""
        
        # 如果是连续动作，转换为离散动作索引
        if self.action_type == 'continuous':
            discrete_action = self._continuous_to_discrete_action(action)
        else:
            discrete_action = action
        
        # 存储到序列回放缓冲区
        self.replay_buffer.add_step(
            state=state,
            action=discrete_action,
            reward=reward,
            done=done,
            hidden_state=copy.deepcopy(self.current_hidden_state) if self.current_hidden_state else None
        )
        
        # 如果episode结束，重置隐藏状态
        if done:
            self.reset_hidden_state()
    
    def _continuous_to_discrete_action(self, action: np.ndarray) -> int:
        """将连续动作转换为离散动作索引"""
        discrete_action = 0
        multiplier = 1
        
        for i in range(self.action_dim):
            closest_idx = np.argmin(np.abs(self.action_grids[i] - action[i]))
            discrete_action += closest_idx * multiplier
            multiplier *= self.action_bins
        
        return discrete_action
    
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
        
        # 解包批次数据
        states = batch['states']  # [batch_size, seq_len, state_dim]
        actions = batch['actions'].long()  # [batch_size, seq_len]
        rewards = batch['rewards']  # [batch_size, seq_len]
        dones = batch['dones']  # [batch_size, seq_len]
        burn_in_states = batch['burn_in_states']  # [batch_size, burn_in_len, state_dim]
        sequence_lengths = batch['sequence_lengths']  # [batch_size]
        
        batch_size, seq_len = states.shape[:2]
        
        # Burn-in阶段：预热RNN隐藏状态
        burn_in_hidden_states = []
        for i in range(batch_size):
            hidden_state = self.q_network.init_hidden_state(1, self.device)
            
            # 如果有burn-in数据，进行预热
            if burn_in_states.shape[1] > 0:
                burn_in_seq = burn_in_states[i:i+1]  # [1, burn_in_len, state_dim]
                with torch.no_grad():
                    _, hidden_state = self.q_network(burn_in_seq, hidden_state)
            
            burn_in_hidden_states.append(hidden_state)
        
        # 合并隐藏状态为批次格式
        if self.config['recurrent_type'].upper() == 'LSTM':
            h_states = torch.cat([h for h, c in burn_in_hidden_states], dim=1)
            c_states = torch.cat([c for h, c in burn_in_hidden_states], dim=1)
            batch_hidden_state = (h_states, c_states)
        else:  # GRU
            h_states = torch.cat([h for h in burn_in_hidden_states], dim=1)
            batch_hidden_state = (h_states,)
        
        # 前向传播计算当前Q值
        current_q_values, _ = self.q_network(states, batch_hidden_state)
        current_q_values = current_q_values.gather(2, actions.unsqueeze(2)).squeeze(2)
        
        # 计算目标Q值
        with torch.no_grad():
            if self.config['double_dqn']:
                # Double DQN：用主网络选择动作，用目标网络评估
                next_q_values_main, _ = self.q_network(states, batch_hidden_state)
                next_actions = next_q_values_main.argmax(2)
                
                next_q_values_target, _ = self.target_network(states, batch_hidden_state)
                next_q_values = next_q_values_target.gather(2, next_actions.unsqueeze(2)).squeeze(2)
            else:
                # 普通DQN
                next_q_values_target, _ = self.target_network(states, batch_hidden_state)
                next_q_values = next_q_values_target.max(2)[0]
            
            # 计算目标值
            target_q_values = rewards + self.config['gamma'] * next_q_values * (1 - dones)
        
        # 计算损失（只对有效时间步计算）
        loss = 0
        valid_steps = 0
        
        for i in range(batch_size):
            seq_len_i = min(sequence_lengths[i].item(), seq_len)
            if seq_len_i > 1:  # 至少需要2个时间步来计算TD误差
                loss += F.mse_loss(
                    current_q_values[i, :seq_len_i-1],
                    target_q_values[i, 1:seq_len_i]
                )
                valid_steps += seq_len_i - 1
        
        if valid_steps > 0:
            loss = loss / batch_size
        else:
            self.training_step += 1
            return None
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            self.q_network.parameters(),
            self.config['gradient_clip']
        )
        
        self.optimizer.step()
        
        # 更新目标网络
        if self.training_step % self.config['target_update_freq'] == 0:
            self.update_target_network()
        
        self.training_step += 1
        self.losses.append(loss.item())
        
        # 返回训练信息
        return {
            'loss': loss.item(),
            'epsilon': self.get_epsilon(),
            'buffer_size': len(self.replay_buffer),
            'valid_steps': valid_steps,
            'avg_q_value': current_q_values.mean().item()
        }
    
    def update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, filepath: str):
        """保存模型"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config,
            'training_step': self.training_step
        }, filepath)
    
    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.training_step = checkpoint['training_step']
        
        print(f"✅ R2D2 model loaded from {filepath}")
    
    def get_stats(self) -> Dict:
        """获取训练统计"""
        buffer_stats = self.replay_buffer.get_stats()
        
        return {
            'training_step': self.training_step,
            'epsilon': self.get_epsilon(),
            'buffer_size': len(self.replay_buffer),
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0,
            'episodes_trained': len(self.episode_rewards),
            **buffer_stats
        }