"""
R2D2 Baseline for Vertical Stratified Queue System
整合到现有基线算法框架中的R2D2实现
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import time
from typing import Dict, Any, Optional

from env.drl_optimized_env_fixed import DRLOptimizedQueueEnvFixed
from baselines.space_utils import SB3DictWrapper
from .r2d2_agent import R2D2Agent


class R2D2Baseline:
    """R2D2基线算法，兼容现有框架"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化R2D2基线
        
        Args:
            config: 配置参数字典
        """
        # 默认配置
        default_config = {
            # 网络配置 - 极简架构快速学习
            'hidden_dim': 128,  # 更小网络，防过拟合，快收敛
            'recurrent_dim': 64, # 小RNN，适合短序列
            'num_layers': 1,
            'recurrent_type': 'LSTM',
            'dueling': True,
            
            # 学习参数 - 超激进学习
            'learning_rate': 1e-3,  # 很高的学习率，快速学习
            'gamma': 0.95,          # 降低折扣，更关注即时奖励
            'target_update_freq': 500,   # 非常频繁的目标网络更新
            'gradient_clip': 5.0,   # 更严格梯度裁剪防止爆炸
            
            # DQN参数 - 超快探索到利用转换
            'epsilon_start': 0.9,   # 降低初始探索
            'epsilon_end': 0.1,     # 保持一定探索
            'epsilon_decay_steps': 10000,  # 更快衰减
            'double_dqn': True,
            
            # 序列回放配置 - 极度优化短episode
            'buffer_size': 500,    # 更小缓冲区，快速周转
            'sequence_length': 3,   # 超短序列，匹配实际episode长度
            'burn_in_length': 1,    # 最小burn-in
            'overlap_length': 1,
            'batch_size': 32,       # 更大batch补偿短序列
            
            # 训练参数 - 立即开始学习
            'learning_starts': 500,   # 极短等待，快速开始学习  
            'train_freq': 2,          # 非常频繁训练，每2步一次
            
            # 动作离散化 - 关键优化！
            'action_bins': 2,  # 减少到2，降低动作空间复杂度
            
            # TensorBoard日志
            'tensorboard_log': "./tensorboard_logs/",
            'verbose': 1,
            'seed': 42
        }
        
        if config:
            default_config.update(config)
        
        self.config = default_config
        self.agent = None
        self.env = None
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'avg_rewards': [],
            'loss_values': [],
            'training_steps': []
        }
        
        print("🔄 R2D2 Baseline initialized")
    
    def setup_env(self):
        """设置环境"""
        base_env = DRLOptimizedQueueEnvFixed()
        self.env = SB3DictWrapper(base_env)
        
        print(f"✅ Environment setup completed")
        print(f"   Observation space: {self.env.observation_space}")
        print(f"   Action space: {self.env.action_space}")
        
        return self.env
    
    def create_agent(self):
        """创建R2D2智能体"""
        if self.env is None:
            self.setup_env()
        
        self.agent = R2D2Agent(
            state_space=self.env.observation_space,
            action_space=self.env.action_space,
            config=self.config
        )
        
        print("✅ R2D2 Agent created successfully")
        return self.agent
    
    def train(self, total_timesteps: int, eval_freq: int = 10000, save_freq: int = 50000):
        """
        训练R2D2模型
        
        Args:
            total_timesteps: 总训练步数
            eval_freq: 评估频率
            save_freq: 保存频率
            
        Returns:
            训练历史字典
        """
        if self.agent is None:
            self.create_agent()
        
        # 创建TensorBoard writer
        tb_log_name = f"R2D2_{int(time.time())}"
        writer = SummaryWriter(
            log_dir=os.path.join(self.config['tensorboard_log'], tb_log_name)
        )
        
        print(f"🚀 Starting R2D2 training for {total_timesteps:,} timesteps...")
        print(f"   TensorBoard log: {tb_log_name}")
        
        # 训练变量
        episode = 0
        timestep = 0
        episode_reward = 0.0
        episode_length = 0
        
        # 重置环境
        state, _ = self.env.reset()
        
        start_time = time.time()
        
        while timestep < total_timesteps:
            # 选择动作
            action = self.agent.act(state, training=True)
            
            # 执行动作
            try:
                step_result = self.env.step(action)
                if len(step_result) == 5:  # Gymnasium格式
                    next_state, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:  # Gym格式
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
            if timestep >= self.config['learning_starts']:
                train_info = self.agent.train()
                
                if train_info and timestep % 1000 == 0:
                    # 记录训练信息
                    writer.add_scalar('train/loss', train_info['loss'], timestep)
                    writer.add_scalar('train/epsilon', train_info['epsilon'], timestep)
                    writer.add_scalar('train/buffer_size', train_info['buffer_size'], timestep)
                    writer.add_scalar('train/avg_q_value', train_info['avg_q_value'], timestep)
                    writer.add_scalar('train/valid_steps', train_info['valid_steps'], timestep)
            
            # 检查是否episode结束
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
                if self.config['verbose'] and episode % 100 == 0:
                    elapsed_time = time.time() - start_time
                    print(f"Episode {episode:5d} | "
                          f"Timestep {timestep:8d} | "
                          f"Reward: {episode_reward:8.2f} | "
                          f"Length: {episode_length:4d} | "
                          f"Epsilon: {self.agent.get_epsilon():.3f} | "
                          f"Time: {elapsed_time:.1f}s")
                
                # 重置episode
                episode += 1
                episode_reward = 0.0
                episode_length = 0
                state, _ = self.env.reset()
                # R2D2会在done=True时自动重置隐藏状态
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
                save_path = f"../../../../Models/r2d2_step_{timestep}.pt"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                self.agent.save(save_path)
                print(f"💾 Model saved at step {timestep}: {save_path}")
        
        # 训练完成
        total_time = time.time() - start_time
        writer.close()
        
        print(f"✅ R2D2 training completed!")
        print(f"   Total episodes: {episode}")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Average reward (last 100): {np.mean(self.training_history['episode_rewards'][-100:]):.2f}")
        
        # 保存最终模型
        final_save_path = "../../../../Models/r2d2_final.pt"
        os.makedirs(os.path.dirname(final_save_path), exist_ok=True)
        self.agent.save(final_save_path)
        
        return {
            'episodes': episode,
            'total_timesteps': timestep,
            'final_reward': np.mean(self.training_history['episode_rewards'][-10:]) if self.training_history['episode_rewards'] else 0,
            'training_time': total_time
        }
    
    def evaluate(self, n_episodes: int = 10, deterministic: bool = True, verbose: bool = True):
        """
        评估模型性能
        
        Args:
            n_episodes: 评估episode数量
            deterministic: 是否使用确定性策略
            verbose: 是否打印详细信息
            
        Returns:
            评估结果字典
        """
        if self.agent is None:
            raise ValueError("Agent not initialized. Please train first.")
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(n_episodes):
            state, _ = self.env.reset()
            self.agent.reset_hidden_state()  # 重置RNN状态
            
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
                
                # 防止无限循环
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
            'episode_lengths': episode_lengths,
            'system_metrics': []  # R2D2特定指标可以在这里添加
        }
        
        if verbose:
            print(f"📈 R2D2 Evaluation Results:")
            print(f"   Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
            print(f"   Mean length: {results['mean_length']:.1f}")
        
        return results
    
    def save_results(self, path_prefix: str):
        """保存训练结果"""
        os.makedirs(os.path.dirname(path_prefix) if os.path.dirname(path_prefix) else ".", exist_ok=True)
        
        # 保存训练历史
        import json
        with open(f"{path_prefix}_history.json", 'w') as f:
            # 转换numpy类型为Python原生类型
            serializable_history = {}
            for key, value in self.training_history.items():
                if isinstance(value, list):
                    serializable_history[key] = [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in value]
                else:
                    serializable_history[key] = value
            
            json.dump(serializable_history, f, indent=2)
        
        print(f"💾 R2D2 results saved to: {path_prefix}")
    
    def save(self, path: str):
        """保存模型"""
        if self.agent is None:
            raise ValueError("Agent not trained yet!")
        
        self.agent.save(path)
        print(f"💾 R2D2 model saved to: {path}")
    
    def load(self, path: str):
        """加载模型"""
        if self.env is None:
            self.setup_env()
        
        if self.agent is None:
            self.create_agent()
        
        self.agent.load(path)
        print(f"📂 R2D2 model loaded from: {path}")
        
        return self.agent


def test_r2d2():
    """测试R2D2"""
    print("🧪 Testing R2D2...")
    
    # 创建基线
    baseline = R2D2Baseline()
    
    # 快速训练测试
    results = baseline.train(total_timesteps=5000)
    print(f"Training results: {results}")
    
    # 评估测试
    eval_results = baseline.evaluate(n_episodes=3)
    print(f"Evaluation results: {eval_results}")
    
    # 保存测试
    baseline.save("../../../../Models/r2d2_test.pt")
    
    print("✅ R2D2 test completed!")


if __name__ == "__main__":
    test_r2d2()