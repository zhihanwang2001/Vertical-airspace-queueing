"""
IMPALA Optimized for Vertical Stratified Queue System
专门针对垂直分层队列环境优化的IMPALA实现

核心优化:
1. 支持混合动作空间（连续+离散）
2. 队列系统专用的网络架构
3. 保守的V-trace参数设置
4. 针对环境特点的状态特征提取
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import time
from typing import Dict, Any, Optional, Tuple

from env.drl_optimized_env_fixed import DRLOptimizedQueueEnvFixed
from baselines.space_utils import SB3DictWrapper


class QueueSpecificNetwork(nn.Module):
    """专门为队列系统设计的网络架构"""

    def __init__(self, state_dim: int, config: Dict = None):
        super().__init__()

        # 默认配置
        self.config = config or {}
        self.hidden_dim = self.config.get('hidden_dim', 512)
        self.num_layers = self.config.get('num_layers', 3)

        # 队列特征维度（环境固定为5层）
        self.n_layers = 5

        # 分层特征提取
        # 1. 队列状态特征提取器（5层队列的专用处理）
        self.queue_feature_extractor = nn.Sequential(
            nn.Linear(self.n_layers * 7, 256),  # 7个特征per layer: lengths, util, changes, load, service, capacity, weights
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # 2. 系统级特征提取器
        self.system_feature_extractor = nn.Sequential(
            nn.Linear(4, 64),  # system_metrics (3) + prev_reward (1)
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # 3. 融合层
        fusion_input_dim = 128 + 32  # queue features + system features
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 4. 输出头
        # Actor: 混合动作空间
        # 连续动作: service_intensities (5) + arrival_multiplier (1) = 6
        self.continuous_actor_mean = nn.Linear(self.hidden_dim, 6)
        self.continuous_actor_logstd = nn.Linear(self.hidden_dim, 6)

        # 离散动作: emergency_transfers (5个二进制选择)
        self.discrete_actor = nn.Linear(self.hidden_dim, self.n_layers)

        # Critic: 价值函数
        self.critic = nn.Linear(self.hidden_dim, 1)

        # 初始化权重
        self._init_weights()

        print(f"🏗️  Queue-Specific Network initialized:")
        print(f"   - Queue layers: {self.n_layers}")
        print(f"   - Hidden dim: {self.hidden_dim}")
        print(f"   - Continuous actions: 6 (service_intensities + arrival_multiplier)")
        print(f"   - Discrete actions: {self.n_layers} (emergency_transfers)")

    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

        # Actor输出层使用小的初始化值
        nn.init.orthogonal_(self.continuous_actor_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.continuous_actor_logstd.weight, gain=0.01)
        nn.init.orthogonal_(self.discrete_actor.weight, gain=0.01)

    def extract_queue_features(self, state: torch.Tensor) -> torch.Tensor:
        """提取队列相关特征"""
        # 假设state是flatten后的35维向量
        # 重构为有意义的队列特征

        batch_size = state.shape[0]

        # 按照环境的观测空间提取特征
        # queue_lengths (5) + utilization_rates (5) + queue_changes (5) +
        # load_rates (5) + service_rates (5) + prev_reward (1) + system_metrics (3) = 29
        # 剩余维度为扩展特征

        queue_lengths = state[:, :5]
        utilization_rates = state[:, 5:10]
        queue_changes = state[:, 10:15]
        load_rates = state[:, 15:20]
        service_rates = state[:, 20:25]
        # prev_reward = state[:, 25:26]  # 后面单独处理
        # system_metrics = state[:, 26:29]  # 后面单独处理

        # 添加固定的队列特征（容量和权重）
        device = state.device
        capacities = torch.tensor([8, 6, 4, 3, 2], dtype=torch.float32, device=device).unsqueeze(0).expand(batch_size, -1)
        arrival_weights = torch.tensor([0.3, 0.25, 0.2, 0.15, 0.1], dtype=torch.float32, device=device).unsqueeze(0).expand(batch_size, -1)

        # 合并队列特征 [batch, 5*7]
        queue_features = torch.cat([
            queue_lengths, utilization_rates, queue_changes,
            load_rates, service_rates, capacities, arrival_weights
        ], dim=1)

        return self.queue_feature_extractor(queue_features)

    def extract_system_features(self, state: torch.Tensor) -> torch.Tensor:
        """提取系统级特征"""
        batch_size = state.shape[0]

        # 提取系统级特征
        if state.shape[1] >= 29:
            prev_reward = state[:, 25:26]
            system_metrics = state[:, 26:29]
        else:
            # 如果维度不够，用零填充
            prev_reward = torch.zeros(batch_size, 1, device=state.device)
            system_metrics = torch.zeros(batch_size, 3, device=state.device)

        system_features = torch.cat([system_metrics, prev_reward], dim=1)
        return self.system_feature_extractor(system_features)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播

        Returns:
            continuous_mean: 连续动作均值 [batch, 6]
            continuous_logstd: 连续动作log标准差 [batch, 6]
            discrete_logits: 离散动作logits [batch, 5]
            value: 状态价值 [batch, 1]
        """
        # 特征提取
        queue_features = self.extract_queue_features(state)
        system_features = self.extract_system_features(state)

        # 特征融合
        combined_features = torch.cat([queue_features, system_features], dim=1)
        fused_features = self.fusion_layers(combined_features)

        # 输出计算
        continuous_mean = self.continuous_actor_mean(fused_features)
        continuous_logstd = torch.clamp(self.continuous_actor_logstd(fused_features), -10, 2)
        discrete_logits = self.discrete_actor(fused_features)
        value = self.critic(fused_features)

        return continuous_mean, continuous_logstd, discrete_logits, value

    def get_action_and_value(self, state: torch.Tensor, deterministic: bool = False):
        """获取动作和价值"""
        continuous_mean, continuous_logstd, discrete_logits, value = self.forward(state)

        if deterministic:
            # 确定性策略
            continuous_action = continuous_mean
            discrete_action = torch.sigmoid(discrete_logits) > 0.5

            # 计算log_prob (用于一致性)
            continuous_log_prob = torch.zeros_like(continuous_mean).sum(dim=-1, keepdim=True)
            discrete_log_prob = torch.zeros_like(discrete_logits).sum(dim=-1, keepdim=True)
        else:
            # 随机策略
            # 连续动作采样
            continuous_std = torch.exp(continuous_logstd)
            continuous_dist = torch.distributions.Normal(continuous_mean, continuous_std)
            continuous_action = continuous_dist.sample()
            continuous_log_prob = continuous_dist.log_prob(continuous_action).sum(dim=-1, keepdim=True)

            # 离散动作采样
            discrete_dist = torch.distributions.Bernoulli(logits=discrete_logits)
            discrete_action = discrete_dist.sample()
            discrete_log_prob = discrete_dist.log_prob(discrete_action).sum(dim=-1, keepdim=True)

        # 合并log_prob
        total_log_prob = continuous_log_prob + discrete_log_prob

        # 组合动作
        action = torch.cat([continuous_action, discrete_action], dim=-1)

        return action, total_log_prob, value

    def evaluate_action(self, state: torch.Tensor, action: torch.Tensor):
        """评估给定状态和动作"""
        continuous_mean, continuous_logstd, discrete_logits, value = self.forward(state)

        # 分离连续和离散动作
        continuous_action = action[:, :6]
        discrete_action = action[:, 6:]

        # 计算连续动作的log_prob和熵
        continuous_std = torch.exp(continuous_logstd)
        continuous_dist = torch.distributions.Normal(continuous_mean, continuous_std)
        continuous_log_prob = continuous_dist.log_prob(continuous_action).sum(dim=-1, keepdim=True)
        continuous_entropy = continuous_dist.entropy().sum(dim=-1, keepdim=True)

        # 计算离散动作的log_prob和熵
        discrete_dist = torch.distributions.Bernoulli(logits=discrete_logits)
        discrete_log_prob = discrete_dist.log_prob(discrete_action).sum(dim=-1, keepdim=True)
        discrete_entropy = discrete_dist.entropy().sum(dim=-1, keepdim=True)

        # 合并
        total_log_prob = continuous_log_prob + discrete_log_prob
        total_entropy = continuous_entropy + discrete_entropy

        return total_log_prob, value, total_entropy


class OptimizedIMPALAAgent:
    """优化的IMPALA智能体"""

    def __init__(self, state_space, action_space, config: Dict = None):
        # 保守的优化配置
        default_config = {
            # 网络配置 - 增加网络容量
            'hidden_dim': 512,
            'num_layers': 3,

            # 学习参数 - 更保守的设置
            'learning_rate': 5e-5,  # 降低学习率
            'gamma': 0.99,
            'entropy_coeff': 0.02,  # 增加探索
            'value_loss_coeff': 0.5,
            'gradient_clip': 10.0,  # 更严格的梯度裁剪

            # V-trace参数 - 保守设置避免训练崩溃
            'rho_bar': 0.8,  # 降低重要性权重截断
            'c_bar': 0.8,    # 降低TD权重截断

            # 回放缓冲区 - 增加容量和序列长度
            'buffer_size': 50000,  # 增加缓冲区
            'sequence_length': 32,  # 增加序列长度捕获长期依赖
            'batch_size': 32,       # 增加批次大小

            # 训练参数 - 更频繁的更新
            'learning_starts': 2000,
            'train_freq': 2,  # 更频繁训练
            'update_freq': 50,

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
            torch.manual_seed(self.config['seed'])
            np.random.seed(self.config['seed'])

        self.state_space = state_space
        self.action_space = action_space

        # 获取状态维度
        if hasattr(state_space, 'shape'):
            self.state_dim = state_space.shape[0]
        else:
            # 处理Dict状态空间
            self.state_dim = sum([space.shape[0] for space in state_space.spaces.values()])

        # 创建专用网络
        self.network = QueueSpecificNetwork(
            state_dim=self.state_dim,
            config=self.config
        ).to(self.device)

        # 优化器
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.config['learning_rate'],
            eps=1e-8  # 增加数值稳定性
        )

        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100000, eta_min=1e-6
        )

        # 简单的经验存储
        self.memory = []
        self.max_memory_size = self.config['buffer_size']

        # 训练统计
        self.training_step = 0
        self.episode_count = 0

        print(f"🚀 Optimized IMPALA Agent initialized on {self.device}")
        print(f"   - Conservative V-trace: rho_bar={self.config['rho_bar']}, c_bar={self.config['c_bar']}")
        print(f"   - Lower learning rate: {self.config['learning_rate']}")
        print(f"   - Larger buffer: {self.config['buffer_size']}")
        print(f"   - Longer sequences: {self.config['sequence_length']}")

    def act(self, state, training: bool = True):
        """选择动作"""
        if isinstance(state, dict):
            # 将Dict状态转换为扁平向量
            state_vector = []
            for key in ['queue_lengths', 'utilization_rates', 'queue_changes',
                       'load_rates', 'service_rates', 'prev_reward', 'system_metrics']:
                if key in state:
                    value = state[key]
                    if isinstance(value, np.ndarray):
                        state_vector.extend(value.flatten())
                    elif hasattr(value, 'flatten'):
                        state_vector.extend(value.flatten())
                    elif isinstance(value, (list, tuple)):
                        state_vector.extend(value)
                    else:
                        state_vector.append(float(value))
            state = np.array(state_vector, dtype=np.float32)

        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)

        # 转换为tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob, value = self.network.get_action_and_value(
                state_tensor, deterministic=not training
            )

            action = action.cpu().numpy()[0]
            log_prob = log_prob.cpu().numpy()[0]
            value = value.cpu().numpy()[0]

        # 存储用于训练的原始动作和转换后的动作
        self._last_raw_action = action
        self._last_log_prob = log_prob[0]
        self._last_value = value[0]

        # 返回原始动作向量（让SB3DictWrapper进行转换）
        return action

    def store_transition(self, state, action, reward, next_state, done):
        """存储经验"""
        if hasattr(self, '_last_raw_action'):
            self.memory.append({
                'state': state,
                'action': self._last_raw_action,
                'reward': reward,
                'next_state': next_state,
                'done': done,
                'log_prob': self._last_log_prob,
                'value': self._last_value
            })

            # 限制内存大小
            if len(self.memory) > self.max_memory_size:
                self.memory.pop(0)

    def train(self):
        """训练一步"""
        if len(self.memory) < self.config['sequence_length'] * self.config['batch_size']:
            return None

        if self.training_step % self.config['train_freq'] != 0:
            self.training_step += 1
            return None

        # 简化的V-trace训练
        batch_size = min(self.config['batch_size'], len(self.memory) // self.config['sequence_length'])

        total_loss = 0.0
        pg_loss_sum = 0.0
        value_loss_sum = 0.0
        entropy_loss_sum = 0.0

        for _ in range(batch_size):
            # 随机采样序列
            start_idx = np.random.randint(0, len(self.memory) - self.config['sequence_length'])
            sequence = self.memory[start_idx:start_idx + self.config['sequence_length']]

            # 构建batch数据
            states = torch.FloatTensor([self._process_state(exp['state']) for exp in sequence]).to(self.device)
            actions = torch.FloatTensor([exp['action'] for exp in sequence]).to(self.device)
            rewards = torch.FloatTensor([exp['reward'] for exp in sequence]).to(self.device)
            dones = torch.FloatTensor([exp['done'] for exp in sequence]).to(self.device)
            old_log_probs = torch.FloatTensor([exp['log_prob'] for exp in sequence]).to(self.device)

            # 计算当前策略的输出
            new_log_probs, values, entropies = self.network.evaluate_action(states, actions)
            values = values.squeeze(-1)
            new_log_probs = new_log_probs.squeeze(-1)
            entropies = entropies.squeeze(-1)

            # 简化的V-trace计算
            with torch.no_grad():
                # 计算重要性权重
                importance_weights = torch.exp(new_log_probs - old_log_probs)
                clipped_importance_weights = torch.clamp(importance_weights, max=self.config['rho_bar'])

                # 计算V-trace targets
                next_values = torch.cat([values[1:], torch.zeros(1, device=self.device)])
                td_targets = rewards + self.config['gamma'] * next_values * (1 - dones)
                advantages = clipped_importance_weights * (td_targets - values)

            # 计算损失
            pg_loss = -(new_log_probs * advantages.detach()).mean()
            value_loss = F.mse_loss(values, td_targets.detach())
            entropy_loss = -entropies.mean()

            # 组合损失
            loss = pg_loss + self.config['value_loss_coeff'] * value_loss + self.config['entropy_coeff'] * entropy_loss

            total_loss += loss
            pg_loss_sum += pg_loss.item()
            value_loss_sum += value_loss.item()
            entropy_loss_sum += entropy_loss.item()

        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config['gradient_clip'])
        self.optimizer.step()
        self.scheduler.step()

        self.training_step += 1

        return {
            'total_loss': total_loss.item() / batch_size,
            'pg_loss': pg_loss_sum / batch_size,
            'value_loss': value_loss_sum / batch_size,
            'entropy_loss': entropy_loss_sum / batch_size,
            'mean_advantage': advantages.mean().item(),
            'buffer_size': len(self.memory),
            'learning_rate': self.scheduler.get_last_lr()[0]
        }

    def _process_state(self, state):
        """处理状态为向量格式"""
        if isinstance(state, dict):
            state_vector = []
            for key in ['queue_lengths', 'utilization_rates', 'queue_changes',
                       'load_rates', 'service_rates', 'prev_reward', 'system_metrics']:
                if key in state:
                    value = state[key]
                    if isinstance(value, np.ndarray):
                        state_vector.extend(value.flatten())
                    elif hasattr(value, 'flatten'):
                        state_vector.extend(value.flatten())
                    elif isinstance(value, (list, tuple)):
                        state_vector.extend(value)
                    else:
                        state_vector.append(float(value))
            return np.array(state_vector, dtype=np.float32)
        return np.array(state, dtype=np.float32)

    def save(self, filepath: str):
        """保存模型"""
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'config': self.config,
            'training_step': self.training_step
        }, filepath)

    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.training_step = checkpoint['training_step']


class OptimizedIMPALABaseline:
    """优化的IMPALA基线算法"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.agent = None
        self.env = None
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'avg_rewards': [],
            'loss_values': [],
            'training_steps': []
        }

        print("🎯 Optimized IMPALA Baseline initialized with queue-specific optimizations")

    def setup_env(self):
        """设置环境"""
        base_env = DRLOptimizedQueueEnvFixed()
        self.env = SB3DictWrapper(base_env)

        print(f"✅ Environment setup completed")
        print(f"   Observation space: {self.env.observation_space}")
        print(f"   Action space: {self.env.action_space}")

        return self.env

    def create_agent(self):
        """创建优化的IMPALA智能体"""
        if self.env is None:
            self.setup_env()

        self.agent = OptimizedIMPALAAgent(
            state_space=self.env.observation_space,
            action_space=self.env.action_space,
            config=self.config
        )

        print("✅ Optimized IMPALA Agent created successfully")
        return self.agent

    def train(self, total_timesteps: int, eval_freq: int = 10000, save_freq: int = 50000):
        """训练优化的IMPALA模型"""
        if self.agent is None:
            self.create_agent()

        # 创建TensorBoard writer
        tb_log_name = f"IMPALA_Optimized_{int(time.time())}"
        writer = SummaryWriter(log_dir=f"./tensorboard_logs/{tb_log_name}")

        print(f"🚀 Starting Optimized IMPALA training for {total_timesteps:,} timesteps...")
        print(f"   TensorBoard log: {tb_log_name}")
        print(f"   Key optimizations:")
        print(f"   - Mixed action space support")
        print(f"   - Queue-specific network architecture")
        print(f"   - Conservative V-trace parameters")
        print(f"   - Lower learning rate with scheduling")

        # 训练循环
        episode = 0
        timestep = 0
        episode_reward = 0.0
        episode_length = 0

        state, _ = self.env.reset()
        start_time = time.time()

        while timestep < total_timesteps:
            # 选择动作
            action = self.agent.act(state, training=True)

            # 执行动作
            try:
                step_result = self.env.step(action)
                if len(step_result) == 5:
                    next_state, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    next_state, reward, done, info = step_result
            except Exception as e:
                print(f"❌ Environment step error: {e}")
                break

            # 存储经验
            self.agent.store_transition(state, action, reward, next_state, done)

            # 更新统计
            episode_reward += reward
            episode_length += 1
            timestep += 1

            # 训练智能体
            if timestep >= self.config.get('learning_starts', 2000):
                train_info = self.agent.train()

                if train_info and timestep % 1000 == 0:
                    # 记录训练信息
                    writer.add_scalar('train/total_loss', train_info['total_loss'], timestep)
                    writer.add_scalar('train/pg_loss', train_info['pg_loss'], timestep)
                    writer.add_scalar('train/value_loss', train_info['value_loss'], timestep)
                    writer.add_scalar('train/entropy_loss', train_info['entropy_loss'], timestep)
                    writer.add_scalar('train/mean_advantage', train_info['mean_advantage'], timestep)
                    writer.add_scalar('train/buffer_size', train_info['buffer_size'], timestep)
                    writer.add_scalar('train/learning_rate', train_info['learning_rate'], timestep)

            # Episode结束处理
            if done:
                # 记录episode信息
                self.training_history['episode_rewards'].append(episode_reward)
                self.training_history['episode_lengths'].append(episode_length)

                # TensorBoard记录
                writer.add_scalar('train/episode_reward', episode_reward, episode)
                writer.add_scalar('train/episode_length', episode_length, episode)

                # 计算滑动平均
                if len(self.training_history['episode_rewards']) >= 100:
                    avg_reward = np.mean(self.training_history['episode_rewards'][-100:])
                    self.training_history['avg_rewards'].append(avg_reward)
                    writer.add_scalar('train/avg_reward_100', avg_reward, episode)

                # 打印进度
                if episode % 100 == 0:
                    elapsed_time = time.time() - start_time
                    recent_rewards = self.training_history['episode_rewards'][-100:] if len(self.training_history['episode_rewards']) >= 100 else self.training_history['episode_rewards']
                    avg_recent = np.mean(recent_rewards) if recent_rewards else 0

                    print(f"Episode {episode:5d} | "
                          f"Timestep {timestep:8d} | "
                          f"Reward: {episode_reward:8.2f} | "
                          f"Avg(100): {avg_recent:8.2f} | "
                          f"Length: {episode_length:4d} | "
                          f"Time: {elapsed_time:.1f}s")

                # 重置episode
                episode += 1
                episode_reward = 0.0
                episode_length = 0
                state, _ = self.env.reset()
            else:
                state = next_state

            # 评估
            if eval_freq > 0 and timestep % eval_freq == 0 and timestep > 0:
                eval_results = self.evaluate(n_episodes=5, deterministic=True, verbose=False)
                writer.add_scalar('eval/mean_reward', eval_results['mean_reward'], timestep)
                writer.add_scalar('eval/std_reward', eval_results['std_reward'], timestep)

                print(f"📊 Evaluation at step {timestep}: "
                      f"Mean reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")

            # 保存模型
            if save_freq > 0 and timestep % save_freq == 0 and timestep > 0:
                save_path = f"../../../../Models/impala_optimized_step_{timestep}.pt"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                self.agent.save(save_path)
                print(f"💾 Model saved at step {timestep}: {save_path}")

        # 训练完成
        total_time = time.time() - start_time
        writer.close()

        print(f"✅ Optimized IMPALA training completed!")
        print(f"   Total episodes: {episode}")
        print(f"   Total time: {total_time:.2f}s")
        final_avg = np.mean(self.training_history['episode_rewards'][-100:]) if len(self.training_history['episode_rewards']) >= 100 else np.mean(self.training_history['episode_rewards']) if self.training_history['episode_rewards'] else 0
        print(f"   Average reward (last 100): {final_avg:.2f}")

        # 保存最终模型
        final_save_path = "../../../../Models/impala_optimized_final.pt"
        os.makedirs(os.path.dirname(final_save_path), exist_ok=True)
        self.agent.save(final_save_path)

        return {
            'episodes': episode,
            'total_timesteps': timestep,
            'final_reward': final_avg,
            'training_time': total_time
        }

    def evaluate(self, n_episodes: int = 10, deterministic: bool = True, verbose: bool = True):
        """评估模型性能"""
        if self.agent is None:
            raise ValueError("Agent not initialized. Please train first.")

        episode_rewards = []
        episode_lengths = []

        for episode in range(n_episodes):
            state, _ = self.env.reset()
            episode_reward = 0.0
            episode_length = 0
            done = False

            while not done:
                action = self.agent.act(state, training=False)

                try:
                    step_result = self.env.step(action)
                    if len(step_result) == 5:
                        next_state, reward, terminated, truncated, info = step_result
                        done = terminated or truncated
                    else:
                        next_state, reward, done, info = step_result
                except Exception as e:
                    print(f"❌ Evaluation error: {e}")
                    break

                episode_reward += reward
                episode_length += 1
                state = next_state

                if episode_length >= 1000:
                    break

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            if verbose:
                print(f"  Episode {episode+1}/{n_episodes}: Reward = {episode_reward:.2f}, Length = {episode_length}")

        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths
        }

        if verbose:
            print(f"📈 Optimized IMPALA Evaluation Results:")
            print(f"   Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
            print(f"   Mean length: {results['mean_length']:.1f}")

        return results

    def save_results(self, path_prefix: str):
        """保存训练结果"""
        os.makedirs(os.path.dirname(path_prefix) if os.path.dirname(path_prefix) else ".", exist_ok=True)

        import json
        with open(f"{path_prefix}_history.json", 'w') as f:
            serializable_history = {}
            for key, value in self.training_history.items():
                if isinstance(value, list):
                    serializable_history[key] = [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in value]
                else:
                    serializable_history[key] = value
            json.dump(serializable_history, f, indent=2)

        print(f"💾 Optimized IMPALA results saved to: {path_prefix}")

    def save(self, path: str):
        """保存模型"""
        if self.agent is None:
            raise ValueError("Agent not trained yet!")

        self.agent.save(path)
        print(f"💾 Optimized IMPALA model saved to: {path}")

    def load(self, path: str):
        """加载模型"""
        if self.env is None:
            self.setup_env()

        if self.agent is None:
            self.create_agent()

        self.agent.load(path)
        print(f"📂 Optimized IMPALA model loaded from: {path}")

        return self.agent


def test_optimized_impala():
    """测试优化的IMPALA"""
    print("🧪 Testing Optimized IMPALA...")

    baseline = OptimizedIMPALABaseline()

    # 快速训练测试
    results = baseline.train(total_timesteps=5000)
    print(f"Training results: {results}")

    # 评估测试
    eval_results = baseline.evaluate(n_episodes=3)
    print(f"Evaluation results: {eval_results}")

    baseline.save("../../../../Models/impala_optimized_test.pt")
    print("✅ Optimized IMPALA test completed!")


if __name__ == "__main__":
    test_optimized_impala()