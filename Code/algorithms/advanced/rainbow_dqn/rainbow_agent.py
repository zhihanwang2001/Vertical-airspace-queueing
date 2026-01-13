"""
Rainbow DQN Agent Implementation
整合所有Rainbow DQN的组件：
1. Double DQN
2. Prioritized Experience Replay
3. Dueling Networks  
4. Multi-step Learning
5. Distributional RL (C51)
6. Noisy Networks
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict, List, Optional
import random
from collections import deque

from .networks import DuelingNoisyNetwork, create_rainbow_network
from .prioritized_replay import PrioritizedReplayBuffer, batch_to_tensors
from .distributional_loss import DistributionalLoss


class RainbowDQNAgent:
    """Rainbow DQN智能体"""
    
    def __init__(self,
                 state_space,
                 action_space,
                 config: Dict = None):
        """
        初始化Rainbow DQN智能体
        
        Args:
            state_space: 状态空间
            action_space: 动作空间
            config: 配置参数
        """
        
        # 优化后配置 - 基于标准Rainbow DQN实现
        default_config = {
            # 网络配置
            'hidden_dim': 512,
            'num_atoms': 51,
            'v_min': -15.0,  # 适应垂直分层队列的奖励范围
            'v_max': 15.0,
            'noisy_std': 0.5,
            
            # 学习参数 - 修复关键超参数
            'learning_rate': 1e-4,  # 🔧 修复: 6.25e-5 → 1e-4 (标准Rainbow学习率)
            'gamma': 0.99,
            'target_update_freq': 2000,  # 🔧 修复: 8000 → 2000 (标准Rainbow更新频率)
            'gradient_clip': 10.0,
            
            # 优先级回放 - 优化缓冲区大小
            'buffer_size': 200000,  # 🔧 修复: 1M → 200k (减少过时经验)
            'alpha': 0.5,
            'beta': 0.4,
            'beta_increment': 0.001,
            'epsilon': 1e-6,
            
            # 多步学习 - 增强长期依赖
            'n_step': 10,  # 🔧 修复: 3 → 10 (适中的multi-step，捕获长期依赖)
            
            # 训练参数 - 早期学习机会
            'batch_size': 32,
            'learning_starts': 5000,  # 🔧 修复: 50000 → 5000 (早期开始学习)
            'train_freq': 4,
            
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
        
        # 处理连续动作空间
        if hasattr(action_space, 'n'):
            # 离散动作空间
            self.num_actions = action_space.n
            self.action_type = 'discrete'
        else:
            # 连续动作空间 - 进行离散化
            self.action_dim = action_space.shape[0]
            self.action_low = action_space.low
            self.action_high = action_space.high
            # 为每个动作维度创建离散化区间
            self.action_bins = 2  # 每个维度2个离散值
            self.num_actions = self.action_bins ** self.action_dim
            self.action_type = 'continuous'
            
            # 创建离散化映射
            self._create_action_mapping()
        
        # 创建网络
        network_config = {
            'hidden_dim': self.config['hidden_dim'],
            'num_atoms': self.config['num_atoms'],
            'v_min': self.config['v_min'],
            'v_max': self.config['v_max']
        }
        
        # 如果是连续动作空间，添加action_bins参数
        if self.action_type == 'continuous':
            network_config['action_bins'] = self.action_bins
        
        self.q_network = create_rainbow_network(
            state_space, action_space, network_config
        ).to(self.device)
        
        self.target_network = create_rainbow_network(
            state_space, action_space, network_config
        ).to(self.device)
        
        # 同步目标网络
        self.update_target_network()
        
        # 优化器
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=self.config['learning_rate']
        )
        
        # 经验回放缓冲区
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=self.config['buffer_size'],
            alpha=self.config['alpha'],
            beta=self.config['beta'],
            beta_increment=self.config['beta_increment'],
            epsilon=self.config['epsilon']
        )
        
        # 分布式损失函数
        self.loss_fn = DistributionalLoss(
            num_atoms=self.config['num_atoms'],
            v_min=self.config['v_min'],
            v_max=self.config['v_max'],
            gamma=self.config['gamma']
        )
        
        # 多步学习缓冲区
        self.n_step_buffer = deque(maxlen=self.config['n_step'])
        
        # 训练统计
        self.training_step = 0
        self.episode_rewards = []
        self.losses = []
        
        print(f"🌈 Rainbow DQN Agent initialized on {self.device}")
        print(f"   State space: {state_space.shape}")
        if self.action_type == 'discrete':
            print(f"   Action space: {self.num_actions} (discrete)")
        else:
            print(f"   Action space: {self.action_dim}D continuous -> {self.num_actions} discrete")
        print(f"   Network atoms: {self.config['num_atoms']}")
        print(f"   Value range: [{self.config['v_min']}, {self.config['v_max']}]")
    
    def _create_action_mapping(self):
        """为连续动作空间创建离散化映射"""
        if self.action_type == 'discrete':
            return
        
        # 为每个动作维度创建离散值 (使用更温和的值避免极端动作)
        self.action_grids = []
        for i in range(self.action_dim):
            # 不使用极端值(-1,1)，而使用更温和的范围(-0.5,0.5)
            grid = np.linspace(-0.5, 0.5, self.action_bins)
            self.action_grids.append(grid)
        
        print(f"   Action discretization: {self.action_bins}^{self.action_dim} = {self.num_actions} discrete actions")
    
    def _discrete_to_continuous_action(self, discrete_action: int) -> np.ndarray:
        """将离散动作转换为连续动作"""
        if self.action_type == 'discrete':
            return discrete_action
        
        # 将离散动作索引转换为多维坐标
        continuous_action = np.zeros(self.action_dim)
        remaining = discrete_action
        
        for i in range(self.action_dim):
            idx = remaining % self.action_bins
            continuous_action[i] = self.action_grids[i][idx]
            remaining //= self.action_bins
        
        return continuous_action
    
    def act(self, state: np.ndarray, training: bool = True):
        """选择动作"""
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        
        # 转换为tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 获取Q分布
            q_dist = self.q_network(state_tensor)
            
            # 计算Q值（分布的期望）
            q_values = self.loss_fn.q_values_from_distribution(q_dist)
            
            # 选择最佳动作（贪心）
            discrete_action = q_values.argmax(dim=1).item()
        
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
        """存储转换到缓冲区"""
        
        # 如果是连续动作，需要找到对应的离散动作索引
        if self.action_type == 'continuous':
            if isinstance(action, np.ndarray):
                # 将连续动作转换为离散动作索引
                discrete_action = 0
                multiplier = 1
                
                for i in range(self.action_dim):
                    # 找到最接近的网格点
                    closest_idx = np.argmin(np.abs(self.action_grids[i] - action[i]))
                    discrete_action += closest_idx * multiplier
                    multiplier *= self.action_bins
            else:
                discrete_action = action
        else:
            discrete_action = action
        
        # 添加到n-step缓冲区
        self.n_step_buffer.append((state, discrete_action, reward, next_state, done))
        
        # 如果n-step缓冲区满了，计算n-step回报
        if len(self.n_step_buffer) == self.config['n_step']:
            # 计算n-step奖励
            n_step_reward = 0.0
            gamma = 1.0
            
            for i in range(self.config['n_step']):
                n_step_reward += gamma * self.n_step_buffer[i][2]
                gamma *= self.config['gamma']
                if self.n_step_buffer[i][4]:  # 如果done
                    break
            
            # 获取初始状态和最终状态
            initial_state = self.n_step_buffer[0][0]
            initial_action = self.n_step_buffer[0][1]
            final_next_state = self.n_step_buffer[-1][3]
            final_done = any(exp[4] for exp in self.n_step_buffer)
            
            # 存储n-step经验
            self.replay_buffer.add(
                initial_state, initial_action, n_step_reward, 
                final_next_state, final_done
            )
    
    def train(self) -> Optional[Dict]:
        """训练一步"""
        if not self.replay_buffer.is_ready:
            return None
        
        if self.training_step % self.config['train_freq'] != 0:
            self.training_step += 1
            return None
        
        # 采样经验
        batch, weights, indices = self.replay_buffer.sample(self.config['batch_size'])
        
        # 转换为tensor
        states, actions, rewards, next_states, dones = batch_to_tensors(batch, self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # 重置噪声
        self.q_network.reset_noise()
        self.target_network.reset_noise()
        
        # 当前Q分布
        current_q_dist = self.q_network(states)
        
        # 目标网络的下一状态Q分布
        with torch.no_grad():
            next_q_dist = self.target_network(next_states)
            
            # Double DQN：使用当前网络选择动作
            next_q_values = self.loss_fn.q_values_from_distribution(
                self.q_network(next_states)
            )
            next_actions = next_q_values.argmax(dim=1)
        
        # 计算损失
        loss, td_errors = self.loss_fn.compute_loss(
            current_q_dist, actions, rewards, next_q_dist, next_actions, dones, weights
        )
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(
            self.q_network.parameters(), 
            self.config['gradient_clip']
        )
        
        self.optimizer.step()
        
        # 更新优先级
        self.replay_buffer.update_priorities(indices, td_errors)
        self.replay_buffer.update_beta()
        
        # 更新目标网络
        if self.training_step % self.config['target_update_freq'] == 0:
            self.update_target_network()
        
        self.training_step += 1
        self.losses.append(loss.item())
        
        return {
            'loss': loss.item(),
            'td_error_mean': np.mean(td_errors),
            'beta': self.replay_buffer.beta,
            'buffer_size': len(self.replay_buffer)
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
        
        print(f"✅ Rainbow DQN model loaded from {filepath}")
    
    def get_stats(self) -> Dict:
        """获取训练统计"""
        return {
            'training_step': self.training_step,
            'buffer_size': len(self.replay_buffer),
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0,
            'beta': self.replay_buffer.beta,
            'episodes_trained': len(self.episode_rewards)
        }
    
    def reset_noise(self):
        """重置网络噪声"""
        self.q_network.reset_noise()
        self.target_network.reset_noise()