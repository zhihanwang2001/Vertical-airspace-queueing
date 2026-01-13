"""
TD7 Baseline Implementation
TD7算法的基线包装器，集成现有框架和TensorBoard监控
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import time
from typing import Dict, Any, Optional
from pathlib import Path

from env.drl_optimized_env_fixed import DRLOptimizedQueueEnvFixed
from baselines.space_utils import SB3DictWrapper
from .td7_agent import TD7_Agent


class TD7Baseline:
    """TD7算法基线实现"""
    
    def __init__(self, config: Dict = None):
        """
        初始化TD7基线
        
        Args:
            config: 配置参数
        """
        
        # 默认配置
        default_config = {
            # 环境配置
            'env_id': 'VerticalQueue-v0',
            'max_episode_steps': 1000,
            'render': False,
            
            # TD7特定参数  
            'embedding_dim': 256,
            'hidden_dim': 256,
            'max_action': 1.0,
            
            # 学习参数
            'actor_lr': 3e-4,
            'critic_lr': 3e-4, 
            'encoder_lr': 3e-4,
            'gamma': 0.99,
            'tau': 0.005,
            
            # TD3特定参数
            'policy_delay': 2,
            'target_noise': 0.2,
            'noise_clip': 0.5,
            'exploration_noise': 0.1,
            
            # 优先级回放
            'buffer_size': 1000000,
            'batch_size': 256,
            'alpha': 0.6,
            'beta': 0.4,
            'beta_increment': 0.001,
            
            # SALE参数
            'embedding_loss_weight': 1.0,
            'embedding_update_freq': 1,
            
            # 检查点机制
            'use_checkpoints': True,
            'checkpoint_freq': 10000,
            'max_checkpoints': 5,
            
            # 训练参数
            'learning_starts': 25000,
            'train_freq': 1,
            'eval_freq': 5000,
            'save_freq': 20000,
            
            # 日志和保存
            'log_dir': './logs/td7',
            'save_dir': '../../../../Models/td7',
            'tensorboard_log': './logs/td7',
            'experiment_name': 'TD7_experiment',
            
            # 其他
            'seed': 42,
            'device': 'auto',
            'verbose': True
        }
        
        if config:
            default_config.update(config)
        
        self.config = default_config
        self.algorithm_name = "TD7"
        self.agent = None
        self.env = None
        self.eval_env = None
        
        # 统计信息
        self.total_timesteps = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_logs = []
        self.eval_rewards = []
        
        print(f"🎯 TD7 Baseline initialized")
        print(f"   Config: {len(default_config)} parameters")
        print(f"   Log dir: {self.config['log_dir']}")
    
    def setup_env(self):
        """设置环境"""
        base_env = DRLOptimizedQueueEnvFixed()
        self.env = SB3DictWrapper(base_env)
        
        print(f"✅ Environment setup completed")
        print(f"   Observation space: {self.env.observation_space}")
        print(f"   Action space: {self.env.action_space}")
        
        return self.env
    
    def create_agent(self):
        """创建TD7智能体"""
        if self.env is None:
            self.setup_env()
        
        self.agent = TD7_Agent(
            state_space=self.env.observation_space,
            action_space=self.env.action_space,
            config=self.config
        )
        
        print("✅ TD7 Agent created successfully")
        return self.agent
    
    def train(self, total_timesteps: int, eval_freq: int = 10000, save_freq: int = 50000):
        """
        训练TD7模型
        
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
        tb_log_name = f"TD7_{int(time.time())}"
        writer = SummaryWriter(
            log_dir=os.path.join(self.config['tensorboard_log'], tb_log_name)
        )
        
        print(f"🚀 Starting TD7 training for {total_timesteps:,} timesteps...")
        print(f"   TensorBoard log: {tb_log_name}")
        
        # 训练变量
        state, info = self.env.reset()
        episode_reward = 0
        episode_length = 0
        episode_count = 0
        
        start_time = time.time()
        
        for timestep in range(1, total_timesteps + 1):
            self.total_timesteps = timestep
            train_info = None
            
            # 选择动作
            if timestep < self.config['learning_starts']:
                action = self.env.action_space.sample()
            else:
                action = self.agent.act(state, training=True)
            
            # 执行动作
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            # 存储经验
            self.agent.store_transition(state, action, reward, next_state, done)
            
            state = next_state
            
            # 训练
            if timestep >= self.config['learning_starts']:
                train_info = self.agent.train()
                if train_info and timestep % 1000 == 0:
                    self.training_logs.append({
                        'timestep': timestep,
                        **train_info,
                        'episode_reward': episode_reward if done else None
                    })
            
            # 回合结束
            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                episode_count += 1

                # 🔧 优化: 记录每个episode的奖励到TensorBoard (用于绘制训练曲线)
                writer.add_scalar('episode/reward', episode_reward, timestep)
                writer.add_scalar('episode/length', episode_length, timestep)
                if len(self.episode_rewards) >= 10:
                    avg_reward_10 = np.mean(self.episode_rewards[-10:])
                    writer.add_scalar('episode/reward_avg_10', avg_reward_10, timestep)
                if len(self.episode_rewards) >= 100:
                    avg_reward_100 = np.mean(self.episode_rewards[-100:])
                    writer.add_scalar('episode/reward_avg_100', avg_reward_100, timestep)

                if self.config['verbose'] and episode_count % 10 == 0:
                    avg_reward = np.mean(self.episode_rewards[-10:])
                    print(f"Episode {episode_count}, Step {timestep:,}, "
                          f"Reward: {episode_reward:.2f}, Avg: {avg_reward:.2f}")

                # 重置环境
                state, info = self.env.reset()
                episode_reward = 0
                episode_length = 0
            
            # 记录到TensorBoard  
            if train_info and timestep % 100 == 0:
                for key, value in train_info.items():
                    if isinstance(value, (int, float)):
                        writer.add_scalar(f'train/{key}', value, timestep)
            
            # 评估
            if timestep % eval_freq == 0:
                eval_reward = self._evaluate(num_episodes=5)
                self.eval_rewards.append({
                    'timestep': timestep,
                    'mean_reward': eval_reward
                })
                writer.add_scalar('eval/mean_reward', eval_reward, timestep)
                print(f"📊 Evaluation at step {timestep:,}: {eval_reward:.2f}")
            
            # 保存模型
            if timestep % save_freq == 0:
                self._save_model(timestep)
        
        training_time = time.time() - start_time

        # 最终评估
        final_eval_reward = self._evaluate(num_episodes=10)

        # 🔧 优化: 导出训练曲线到CSV文件 (用于论文绘图)
        self._export_training_curve_to_csv(tb_log_name)

        # 关闭writer
        writer.close()

        results = {
            'algorithm': self.algorithm_name,
            'total_timesteps': total_timesteps,
            'final_eval_reward': final_eval_reward,
            'training_time': training_time,
            'episodes': len(self.episode_rewards),
            'avg_episode_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
        }

        if self.agent:
            results.update(self.agent.get_stats())

        print(f"🎉 TD7 training completed!")
        print(f"   Final evaluation reward: {final_eval_reward:.2f}")
        print(f"   Training time: {training_time:.1f}s")
        print(f"   Episodes completed: {len(self.episode_rewards)}")

        return results

    def _export_training_curve_to_csv(self, tb_log_name: str):
        """导出训练曲线到CSV文件"""
        import csv

        csv_dir = Path("result_excel")
        csv_dir.mkdir(exist_ok=True)
        csv_path = csv_dir / "TD7.csv"

        # 计算移动平均奖励 (每1000步)
        data_points = []
        window_size = 10  # 10个episode的移动平均

        for i in range(len(self.episode_rewards)):
            if i >= window_size - 1:
                avg_reward = np.mean(self.episode_rewards[i - window_size + 1:i + 1])
                # 估算对应的timestep (假设平均episode长度)
                avg_length = np.mean(self.episode_lengths[:i+1])
                timestep = int((i + 1) * avg_length)

                data_points.append({
                    'Step': timestep,
                    'Value': avg_reward
                })

        # 写入CSV
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['Wall time', 'Step', 'Value'])
            writer.writeheader()

            for point in data_points:
                writer.writerow({
                    'Wall time': time.time(),
                    'Step': point['Step'],
                    'Value': point['Value']
                })

        print(f"✅ Training curve exported to: {csv_path}")
        return csv_path
    
    def _evaluate(self, num_episodes: int = 10) -> float:
        """评估智能体性能"""
        if self.agent is None:
            return 0.0
        
        # 使用训练环境进行评估
        episode_rewards = []
        
        for _ in range(num_episodes):
            state, info = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = self.agent.act(state, training=False)
                state, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            episode_rewards.append(episode_reward)
        
        return np.mean(episode_rewards)
    
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
            state, info = self.env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                action = self.agent.act(state, training=False)
                state, reward, terminated, truncated, info = self.env.step(action)
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated
                
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
            'system_metrics': []  # TD7特定指标可以在这里添加
        }
        
        if verbose:
            print(f"📊 TD7 Evaluation Results:")
            print(f"   Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
            print(f"   Mean length: {results['mean_length']:.1f}")
        
        return results
    
    def _save_model(self, timestep: int):
        """保存模型"""
        if self.agent is None:
            return
        
        # 创建保存目录
        save_dir = Path(self.config['save_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = save_dir / f"td7_model_{timestep}.pt"
        self.agent.save(str(model_path))
        print(f"💾 Model saved: {model_path}")
    
    def load_model(self, model_path: str):
        """加载模型"""
        if self.agent is None:
            self.create_agent()
        
        self.agent.load(model_path)
        print(f"✅ TD7 model loaded from {model_path}")
    
    def save(self, path: str):
        """保存模型"""
        if self.agent is None:
            raise ValueError("Agent not trained yet!")
        
        # 创建保存目录
        save_dir = Path(path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        self.agent.save(path)
        print(f"💾 TD7 model saved to: {path}")
    
    def load(self, path: str):
        """加载模型"""
        if self.env is None:
            self.setup_env()
        
        if self.agent is None:
            self.create_agent()
        
        self.agent.load(path)
        print(f"📂 TD7 model loaded from: {path}")
        
        return self.agent
    
    def predict(self, state: np.ndarray, training: bool = False) -> np.ndarray:
        """预测动作"""
        if self.agent is None:
            raise ValueError("Agent not initialized. Call setup() first.")
        
        return self.agent.act(state, training=training)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            'algorithm': self.algorithm_name,
            'total_timesteps': self.total_timesteps,
            'episodes_completed': len(self.episode_rewards),
            'avg_episode_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'avg_episode_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
        }
        
        if self.agent:
            stats.update(self.agent.get_stats())
        
        if self.eval_rewards:
            stats['best_eval_reward'] = max([r['mean_reward'] for r in self.eval_rewards])
            stats['latest_eval_reward'] = self.eval_rewards[-1]['mean_reward']
        
        return stats
    
    def cleanup(self):
        """清理资源"""
        if self.env:
            self.env.close()
        if self.eval_env:
            self.eval_env.close()
        print("🧹 TD7 baseline cleanup completed")