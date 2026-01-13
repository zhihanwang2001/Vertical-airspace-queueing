"""
SB3 SAC算法基线
SB3 SAC Baseline Algorithm
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
from env.drl_optimized_env_fixed import DRLOptimizedQueueEnvFixed
from .space_utils import SB3DictWrapper


class SB3SACBaseline:
    """SB3 SAC基线算法"""
    
    def __init__(self, config=None):
        default_config = {
            'learning_rate': 3e-4,
            'buffer_size': 1000000,
            'learning_starts': 100,
            'batch_size': 256,
            'tau': 0.005,
            'gamma': 0.99,
            'train_freq': 1,
            'gradient_steps': 1,
            'ent_coef': 'auto',  # Automatic entropy tuning
            'target_update_interval': 1,
            'target_entropy': 'auto',
            'tensorboard_log': "./tensorboard_logs/",
            'verbose': 1,
            'seed': 42
        }
        
        if config:
            default_config.update(config)
        
        self.config = default_config
        self.model = None
        self.env = None
        
    def setup_env(self):
        """设置环境"""
        base_env = DRLOptimizedQueueEnvFixed()
        wrapped_env = SB3DictWrapper(base_env)
        self.env = Monitor(wrapped_env, filename=None)
        
        # 创建向量化环境
        self.vec_env = DummyVecEnv([lambda: self.env])
        
        return self.env
    
    def create_model(self):
        """创建SAC模型"""
        if self.env is None:
            self.setup_env()
        
        # 创建SAC模型
        self.model = SAC(
            "MlpPolicy",
            self.vec_env,
            learning_rate=self.config['learning_rate'],
            buffer_size=self.config['buffer_size'],
            learning_starts=self.config['learning_starts'],
            batch_size=self.config['batch_size'],
            tau=self.config['tau'],
            gamma=self.config['gamma'],
            train_freq=self.config['train_freq'],
            gradient_steps=self.config['gradient_steps'],
            ent_coef=self.config['ent_coef'],
            target_update_interval=self.config['target_update_interval'],
            target_entropy=self.config['target_entropy'],
            tensorboard_log=self.config['tensorboard_log'],
            verbose=self.config['verbose'],
            seed=self.config['seed'],
            device='auto'
        )
        
        print(f"SB3 SAC model created with device: {self.model.device}")
        return self.model
    
    def train(self, total_timesteps, eval_freq=10000, save_freq=50000):
        """训练模型"""
        if self.model is None:
            self.create_model()
        
        # 创建必要的目录
        os.makedirs('./logs/', exist_ok=True)
        os.makedirs('../../../Models/sb3_sac_best/', exist_ok=True)
        os.makedirs('../../../Models/sb3_sac_checkpoints/', exist_ok=True)
        
        # 创建评估环境
        eval_env = DummyVecEnv([lambda: Monitor(
            SB3DictWrapper(DRLOptimizedQueueEnvFixed()), 
            filename=None
        )])
        
        # 创建回调
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path='../../../Models/sb3_sac_best/',
            log_path='./logs/',
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
            n_eval_episodes=10,
            verbose=1
        )
        
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path='../../../Models/sb3_sac_checkpoints/',
            name_prefix='sb3_sac'
        )
        
        # 开始训练
        print(f"Starting SB3 SAC training for {total_timesteps:,} timesteps...")
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=None,  # 移除有问题的callbacks
            log_interval=10,
            tb_log_name="SB3_SAC"
        )
        
        print("SB3 SAC training completed!")
        
        # 返回训练结果字典以兼容比较框架
        return {
            'episodes': 0,  # SB3没有直接的episode计数
            'total_timesteps': total_timesteps,
            'final_reward': 0  # 将在评估中获得
        }
    
    def evaluate(self, n_episodes=10, deterministic=True, verbose=True):
        """评估模型"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # 创建评估环境
        eval_env = SB3DictWrapper(DRLOptimizedQueueEnvFixed())
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(n_episodes):
            obs, _ = eval_env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                
                if episode_length >= 200:  # 防止无限循环
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            if verbose:
                print(f"  Episode {episode+1}/{n_episodes}: Reward = {episode_reward:.2f}")
        
        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'system_metrics': []  # SB3算法没有系统指标
        }
        
        return results
    
    def save_results(self, path_prefix):
        """保存训练历史和结果"""
        # 创建目录
        os.makedirs(os.path.dirname(path_prefix) if os.path.dirname(path_prefix) else ".", exist_ok=True)
        
        # SB3算法没有训练历史，创建空的历史记录
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'avg_rewards': [],
            'loss_values': []
        }
        
        # 保存为JSON文件（如果需要的话）
        import json
        with open(f"{path_prefix}_history.json", 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        print(f"SB3 SAC results saved to: {path_prefix}")
    
    def save(self, path):
        """保存模型"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        self.model.save(path)
        print(f"SB3 SAC model saved to: {path}")
    
    def load(self, path):
        """加载模型"""
        if self.env is None:
            self.setup_env()
        
        self.model = SAC.load(path, env=self.vec_env)
        print(f"SB3 SAC model loaded from: {path}")
        return self.model


def test_sb3_sac():
    """测试SB3 SAC"""
    print("Testing SB3 SAC...")
    
    # 创建基线
    baseline = SB3SACBaseline()
    
    # 训练
    baseline.train(total_timesteps=50000)
    
    # 评估
    results = baseline.evaluate(n_episodes=10)
    print(f"SB3 SAC Results: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    
    # 保存
    baseline.save("../../../Models/sb3_sac_test.zip")


if __name__ == "__main__":
    test_sb3_sac()