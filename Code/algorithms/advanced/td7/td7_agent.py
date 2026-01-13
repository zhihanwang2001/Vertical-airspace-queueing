"""
TD7 Agent Implementation
TD7智能体实现，集成SALE表示学习、优先级回放和检查点机制
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List, Any
import random
import copy

from .networks import create_td7_networks
from .replay_buffer import TD7_PrioritizedReplayBuffer


class TD7_Agent:
    """TD7智能体"""
    
    def __init__(self,
                 state_space,
                 action_space,
                 config: Dict = None):
        """
        初始化TD7智能体
        
        Args:
            state_space: 状态空间
            action_space: 动作空间
            config: 配置参数
        """
        
        # 默认配置
        default_config = {
            # 网络配置
            'embedding_dim': 256,
            'hidden_dim': 256,
            'max_action': 1.0,
            
            # 学习参数
            'actor_lr': 3e-4,
            'critic_lr': 3e-4,
            'encoder_lr': 3e-4,
            'gamma': 0.99,
            'tau': 0.005,

            # 🔧 学习率调度（防止后期崩溃）
            'use_lr_schedule': True,        # 是否使用学习率调度
            'warmup_steps': 75000,          # 前75k步保持固定lr（稳定期）
            'total_steps': 500000,          # 总训练步数
            'min_lr_ratio': 0.1,            # 最小学习率比例（final_lr = initial_lr * 0.1）
            
            # TD3特定参数
            'policy_delay': 2,      # 延迟策略更新
            'target_noise': 0.2,    # 目标平滑噪声
            'noise_clip': 0.5,      # 噪声裁剪
            'exploration_noise': 0.1, # 探索噪声
            
            # 优先级回放
            'buffer_size': 1000000,
            'batch_size': 256,
            'alpha': 0.6,           # 优先级指数
            'beta': 0.4,            # 重要性采样指数
            'beta_increment': 0.001,
            
            # SALE特定参数
            'embedding_loss_weight': 1.0,  # 嵌入损失权重
            'embedding_update_freq': 1,     # 编码器更新频率
            
            # 检查点机制
            'use_checkpoints': True,
            'checkpoint_freq': 10000,       # 检查点保存频率
            'max_checkpoints': 5,           # 最大检查点数量
            
            # 训练参数
            'learning_starts': 25000,
            'train_freq': 1,
            
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
            'embedding_dim': self.config['embedding_dim'],
            'hidden_dim': self.config['hidden_dim'],
            'max_action': self.config['max_action']
        }
        
        self.networks = create_td7_networks(
            state_space, action_space, network_config
        )
        self.networks.device = self.device
        
        # 移动到正确设备
        for network in [self.networks.state_encoder, self.networks.actor, self.networks.critic,
                       self.networks.target_state_encoder, self.networks.target_actor, 
                       self.networks.target_critic]:
            network.to(self.device)
        
        # 优化器
        self.encoder_optimizer = optim.Adam(
            self.networks.state_encoder.parameters(),
            lr=self.config['encoder_lr']
        )

        self.actor_optimizer = optim.Adam(
            self.networks.actor.parameters(),
            lr=self.config['actor_lr']
        )

        self.critic_optimizer = optim.Adam(
            self.networks.critic.parameters(),
            lr=self.config['critic_lr']
        )

        # 🔧 添加学习率调度器（余弦退火，防止后期崩溃）
        if self.config['use_lr_schedule']:
            from torch.optim.lr_scheduler import LambdaLR

            def lr_lambda(step):
                """
                延迟余弦退火学习率调度
                - 前75k步: 保持固定学习率1.0
                - 75k步后: 余弦退火至0.1
                """
                warmup_steps = self.config['warmup_steps']
                total_steps = self.config['total_steps']
                min_ratio = self.config['min_lr_ratio']

                if step < warmup_steps:
                    return 1.0  # 固定学习率
                else:
                    # 余弦退火
                    progress = (step - warmup_steps) / (total_steps - warmup_steps)
                    progress = min(progress, 1.0)
                    cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
                    return min_ratio + (1.0 - min_ratio) * cosine_factor

            self.encoder_scheduler = LambdaLR(self.encoder_optimizer, lr_lambda)
            self.actor_scheduler = LambdaLR(self.actor_optimizer, lr_lambda)
            self.critic_scheduler = LambdaLR(self.critic_optimizer, lr_lambda)
        else:
            self.encoder_scheduler = None
            self.actor_scheduler = None
            self.critic_scheduler = None
        
        # 优先级经验回放缓冲区
        self.replay_buffer = TD7_PrioritizedReplayBuffer(
            capacity=self.config['buffer_size'],
            batch_size=self.config['batch_size'],
            alpha=self.config['alpha'],
            beta=self.config['beta'],
            beta_increment=self.config['beta_increment'],
            device=self.device
        )
        
        # 训练统计
        self.training_step = 0
        self.episode_rewards = []
        self.losses = {
            'actor_loss': [],
            'critic_loss': [],
            'encoder_loss': []
        }
        
        # 检查点管理
        self.checkpoints = []
        
        print(f"🎯 TD7 Agent initialized on {self.device}")
        print(f"   State space: {state_space.shape}")
        print(f"   Action space: {action_space.shape}")
        print(f"   Embedding dim: {self.config['embedding_dim']}")
        print(f"   Use checkpoints: {self.config['use_checkpoints']}")
        print(f"   Policy delay: {self.config['policy_delay']}")
    
    def act(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """选择动作"""
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        
        # 转换为tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 获取状态嵌入
            state_embedding = self.networks.state_encoder(state_tensor)
            
            # 获取动作
            action = self.networks.actor(state_embedding).cpu().numpy()[0]
            
            # 添加探索噪声
            if training:
                noise = np.random.normal(
                    0, self.config['exploration_noise'], size=action.shape
                )
                action += noise
        
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
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def train(self) -> Optional[Dict]:
        """训练一步"""
        if not self.replay_buffer.is_ready:
            return None
        
        if self.replay_buffer.tree.n_entries < self.config['learning_starts']:
            return None
        
        if self.training_step % self.config['train_freq'] != 0:
            self.training_step += 1
            return None
        
        # 采样批次
        batch = self.replay_buffer.sample()
        if batch is None:
            self.training_step += 1
            return None
        
        # 更新网络
        losses = self._update_networks(batch)
        
        # 检查点管理
        if (self.config['use_checkpoints'] and 
            self.training_step % self.config['checkpoint_freq'] == 0):
            self._save_checkpoint()
        
        # 软更新目标网络
        self.networks.soft_update_target_networks(self.config['tau'])
        
        self.training_step += 1
        
        # 存储损失
        for key, value in losses.items():
            self.losses[key].append(value)
        
        # 返回训练信息
        return {
            'actor_loss': losses['actor_loss'],
            'critic_loss': losses['critic_loss'],
            'encoder_loss': losses['encoder_loss'],
            'buffer_size': len(self.replay_buffer),
            'training_step': self.training_step,
            'max_priority': self.replay_buffer.max_priority
        }
    
    def _update_networks(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """更新网络参数"""
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        is_weights = batch['is_weights']
        indices = batch['indices']
        
        # 获取状态嵌入
        state_embeddings = self.networks.state_encoder(states)
        next_state_embeddings = self.networks.target_state_encoder(next_states)
        
        # 更新Critic网络
        critic_loss, td_errors = self._update_critic(
            state_embeddings, actions, rewards, next_state_embeddings, dones, is_weights
        )
        
        # 更新优先级
        priorities = self.replay_buffer.compute_lap_priority(td_errors, actions)
        self.replay_buffer.update_priorities(indices, priorities)
        
        # 延迟策略更新
        actor_loss = 0.0
        encoder_loss = 0.0
        
        if self.training_step % self.config['policy_delay'] == 0:
            # 更新Actor网络
            actor_loss = self._update_actor(state_embeddings.detach())
            
            # 更新状态编码器
            if self.training_step % self.config['embedding_update_freq'] == 0:
                encoder_loss = self._update_encoder(states, actions)
        
        return {
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'encoder_loss': encoder_loss
        }
    
    def _update_critic(self, state_embeddings, actions, rewards, next_state_embeddings, dones, is_weights):
        """更新Critic网络"""
        with torch.no_grad():
            # 计算目标动作（带噪声）
            target_actions = self.networks.target_actor(next_state_embeddings)
            
            # 添加目标平滑噪声
            noise = torch.clamp(
                torch.randn_like(target_actions) * self.config['target_noise'],
                -self.config['noise_clip'], self.config['noise_clip']
            )
            target_actions = torch.clamp(
                target_actions + noise,
                -self.networks.max_action, self.networks.max_action
            )
            
            # 计算目标Q值（取最小值减少过估计）
            target_q1, target_q2 = self.networks.target_critic(next_state_embeddings, target_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + self.config['gamma'] * (1 - dones) * target_q
        
        # 当前Q值
        current_q1, current_q2 = self.networks.critic(state_embeddings, actions)
        
        # TD误差
        td_error1 = target_q - current_q1
        td_error2 = target_q - current_q2
        
        # Huber损失（TD7特色）
        critic_loss1 = F.smooth_l1_loss(current_q1, target_q, reduction='none')
        critic_loss2 = F.smooth_l1_loss(current_q2, target_q, reduction='none')
        
        # 重要性采样加权
        critic_loss1 = (critic_loss1 * is_weights).mean()
        critic_loss2 = (critic_loss2 * is_weights).mean()
        
        critic_loss = critic_loss1 + critic_loss2
        
        # 更新Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 🔧 更新critic学习率调度
        if self.critic_scheduler is not None:
            self.critic_scheduler.step()

        # 返回TD误差（用于优先级更新）
        td_errors = torch.max(torch.abs(td_error1), torch.abs(td_error2)).detach()

        return critic_loss.item(), td_errors
    
    def _update_actor(self, state_embeddings):
        """更新Actor网络"""
        # 计算策略损失
        actions = self.networks.actor(state_embeddings)
        actor_loss = -self.networks.critic.q1(state_embeddings, actions).mean()
        
        # 更新Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 🔧 更新actor学习率调度
        if self.actor_scheduler is not None:
            self.actor_scheduler.step()

        return actor_loss.item()
    
    def _update_encoder(self, states, actions):
        """更新状态编码器（SALE机制）"""
        # 重新计算嵌入
        state_embeddings = self.networks.state_encoder(states)
        
        # SALE损失：最大化状态-动作相互作用
        # 这里使用简化的SALE损失
        pred_actions = self.networks.actor(state_embeddings)
        embedding_loss = F.mse_loss(pred_actions, actions)
        
        # 嵌入正则化
        embedding_reg = torch.mean(torch.norm(state_embeddings, dim=-1))
        
        total_encoder_loss = (
            self.config['embedding_loss_weight'] * embedding_loss + 
            0.01 * embedding_reg
        )
        
        # 更新编码器
        self.encoder_optimizer.zero_grad()
        total_encoder_loss.backward()
        self.encoder_optimizer.step()

        # 🔧 更新encoder学习率调度
        if self.encoder_scheduler is not None:
            self.encoder_scheduler.step()

        return total_encoder_loss.item()
    
    def _save_checkpoint(self):
        """保存检查点"""
        checkpoint = {
            'state_encoder': self.networks.state_encoder.state_dict(),
            'actor': self.networks.actor.state_dict(),
            'critic': self.networks.critic.state_dict(),
            'training_step': self.training_step,
            'losses': copy.deepcopy(self.losses)
        }
        
        self.checkpoints.append(checkpoint)
        
        # 保持最大检查点数量
        if len(self.checkpoints) > self.config['max_checkpoints']:
            self.checkpoints.pop(0)
        
        print(f"💾 Checkpoint saved at step {self.training_step} (total: {len(self.checkpoints)})")
    
    def load_best_checkpoint(self):
        """加载最佳检查点"""
        if not self.checkpoints:
            print("⚠️ No checkpoints available")
            return
        
        # 简单策略：加载最新的检查点
        best_checkpoint = self.checkpoints[-1]
        
        self.networks.state_encoder.load_state_dict(best_checkpoint['state_encoder'])
        self.networks.actor.load_state_dict(best_checkpoint['actor'])
        self.networks.critic.load_state_dict(best_checkpoint['critic'])
        
        print(f"✅ Loaded checkpoint from step {best_checkpoint['training_step']}")
    
    def save(self, filepath: str):
        """保存模型"""
        save_dict = {
            'state_encoder': self.networks.state_encoder.state_dict(),
            'actor': self.networks.actor.state_dict(),
            'critic': self.networks.critic.state_dict(),
            'target_state_encoder': self.networks.target_state_encoder.state_dict(),
            'target_actor': self.networks.target_actor.state_dict(),
            'target_critic': self.networks.target_critic.state_dict(),
            'encoder_optimizer': self.encoder_optimizer.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'config': self.config,
            'training_step': self.training_step,
            'checkpoints': self.checkpoints
        }
        
        torch.save(save_dict, filepath)
    
    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        self.networks.state_encoder.load_state_dict(checkpoint['state_encoder'])
        self.networks.actor.load_state_dict(checkpoint['actor'])
        self.networks.critic.load_state_dict(checkpoint['critic'])
        self.networks.target_state_encoder.load_state_dict(checkpoint['target_state_encoder'])
        self.networks.target_actor.load_state_dict(checkpoint['target_actor'])
        self.networks.target_critic.load_state_dict(checkpoint['target_critic'])
        
        self.encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        
        self.training_step = checkpoint['training_step']
        
        if 'checkpoints' in checkpoint:
            self.checkpoints = checkpoint['checkpoints']
        
        print(f"✅ TD7 model loaded from {filepath}")
    
    def get_stats(self) -> Dict:
        """获取训练统计"""
        buffer_stats = self.replay_buffer.get_stats()
        
        stats = {
            'training_step': self.training_step,
            'episodes_trained': len(self.episode_rewards),
            'num_checkpoints': len(self.checkpoints),
            **buffer_stats
        }
        
        # 添加损失统计
        for key, losses in self.losses.items():
            if losses:
                stats[f'avg_{key}'] = np.mean(losses[-100:])
        
        return stats