"""
SB3 PPO算法基线
SB3 PPO Baseline Algorithm
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import math
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
from env.drl_optimized_env_fixed import DRLOptimizedQueueEnvFixed
from .space_utils import SB3DictWrapper


class LearningRateLogger(BaseCallback):
    """
    增强的学习率记录器
    Enhanced Learning Rate Logger for TensorBoard
    """
    
    def __init__(self, initial_lr: float = 3e-4, min_lr: float = 1e-6, verbose: int = 1):
        super().__init__(verbose)
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        
    def _on_step(self) -> bool:
        """记录当前学习率到TensorBoard"""
        # 获取当前学习率
        current_lr = self.model.policy.optimizer.param_groups[0]['lr']
        progress_remaining = getattr(self.model, '_current_progress_remaining', 1.0)
        progress = 1.0 - progress_remaining
        
        # 不保存历史记录以避免pickle错误
        
        # 记录到TensorBoard
        self.logger.record("train/learning_rate", current_lr)
        self.logger.record("train/lr_progress", progress)
        self.logger.record("train/lr_decay_ratio", current_lr / self.initial_lr)
        
        # 计算理论学习率（用于验证）
        cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
        theoretical_lr = self.min_lr + (self.initial_lr - self.min_lr) * cosine_factor
        self.logger.record("train/theoretical_lr", theoretical_lr)
        self.logger.record("train/lr_error", abs(current_lr - theoretical_lr))
        
        # 定期打印学习率
        if self.num_timesteps % 10000 == 0 and self.verbose > 0:
            print(f"Step {self.num_timesteps:6,}: LR={current_lr:.6f} (Progress: {progress:.1%}, Theoretical: {theoretical_lr:.6f})")
        
        return True


class SB3PPOBaseline:
    """SB3 PPO基线算法"""
    
    def __init__(self, config=None):
        default_config = {
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'clip_range_vf': None,
            'normalize_advantage': True,
            'ent_coef': 0.0,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'target_kl': None,
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
        base_env = DRLOptimizedQueueEnvFixed(max_episode_steps=10000)
        wrapped_env = SB3DictWrapper(base_env)
        self.env = Monitor(wrapped_env, filename=None)
        
        # 创建向量化环境
        self.vec_env = DummyVecEnv([lambda: self.env])
        
        return self.env
    
    def create_model(self):
        """创建PPO模型"""
        if self.env is None:
            self.setup_env()
        
        # 创建学习率调度函数
        def cosine_annealing_schedule(progress_remaining):
            """
            余弦退火学习率调度
            progress_remaining: 1.0 -> 0.0 (从开始到结束)
            """
            initial_lr = self.config['learning_rate']
            min_lr = self.config.get('min_lr', 1e-6)
            
            # 转换progress_remaining到progress (0.0 -> 1.0)
            progress = 1.0 - progress_remaining
            
            # 余弦退火公式
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            current_lr = min_lr + (initial_lr - min_lr) * cosine_factor
            
            return current_lr
        
        # 创建PPO模型
        self.model = PPO(
            "MlpPolicy",
            self.vec_env,
            learning_rate=cosine_annealing_schedule,  # 使用调度函数
            n_steps=self.config['n_steps'],
            batch_size=self.config['batch_size'],
            n_epochs=self.config['n_epochs'],
            gamma=self.config['gamma'],
            gae_lambda=self.config['gae_lambda'],
            clip_range=self.config['clip_range'],
            clip_range_vf=self.config['clip_range_vf'],
            normalize_advantage=self.config['normalize_advantage'],
            ent_coef=self.config['ent_coef'],
            vf_coef=self.config['vf_coef'],
            max_grad_norm=self.config['max_grad_norm'],
            target_kl=self.config['target_kl'],
            tensorboard_log=self.config['tensorboard_log'],
            verbose=self.config['verbose'],
            seed=self.config['seed'],
            device='auto'
        )
        
        print(f"SB3 PPO model created with device: {self.model.device}")
        return self.model
    
    def train(self, total_timesteps, eval_freq=10000, save_freq=50000):
        """训练模型"""
        if self.model is None:
            self.create_model()
        
        # 创建必要的目录
        os.makedirs('./logs/', exist_ok=True)
        os.makedirs('../../../Models/sb3_ppo_best/', exist_ok=True)
        os.makedirs('../../../Models/sb3_ppo_checkpoints/', exist_ok=True)
        
        # 创建评估环境
        eval_env = DummyVecEnv([lambda: Monitor(
            SB3DictWrapper(DRLOptimizedQueueEnvFixed()), 
            filename=None
        )])
        
        # 创建学习率记录器
        lr_logger = LearningRateLogger(
            initial_lr=self.config['learning_rate'],
            min_lr=self.config.get('min_lr', 1e-6),
            verbose=1
        )
        
        # 创建评估回调
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path='../../../Models/sb3_ppo_best/',
            log_path='./logs/',
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
            n_eval_episodes=10,
            verbose=1
        )
        
        # 创建检查点回调
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path='../../../Models/sb3_ppo_checkpoints/',
            name_prefix='sb3_ppo'
        )
        
        # 开始训练
        print(f"Starting SB3 PPO training for {total_timesteps:,} timesteps...")
        print(f"Using Cosine Annealing LR: {self.config['learning_rate']} -> {self.config.get('min_lr', 1e-6)}")
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=lr_logger,  # 只使用学习率记录器
            log_interval=10,
            tb_log_name="SB3_PPO_CosineAnneal"
        )
        
        print("SB3 PPO training completed!")
        
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
        eval_env = SB3DictWrapper(DRLOptimizedQueueEnvFixed(max_episode_steps=10000))

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

                if episode_length >= 10000:  # 防止无限循环
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
        
        print(f"SB3 PPO results saved to: {path_prefix}")
    
    def save(self, path):
        """保存模型"""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        try:
            # 尝试使用 exclude 参数来避免 pickle 错误
            # 排除环境对象，只保存模型参数
            self.model.save(path, exclude=['env', 'logger', 'ep_info_buffer', 'ep_success_buffer'])
            print(f"SB3 PPO model saved to: {path}")
        except Exception as e:
            # 如果仍然失败，保存为PyTorch state dict
            print(f"Warning: Standard save failed ({e}), using state_dict fallback...")
            import torch
            state_dict = {
                'policy_state_dict': self.model.policy.state_dict(),
                'observation_space': self.model.observation_space,
                'action_space': self.model.action_space,
            }
            torch.save(state_dict, path + '.pth')
            print(f"SB3 PPO model saved as state_dict to: {path}.pth")
    
    def load(self, path):
        """加载模型"""
        import os
        import torch

        if self.env is None:
            self.setup_env()

        # 检查是否是.pth文件（fallback格式）
        if path.endswith('.pth') or (not path.endswith('.zip') and os.path.exists(path + '.pth')):
            pth_path = path if path.endswith('.pth') else path + '.pth'
            print(f"Loading from state_dict format: {pth_path}")

            # 加载state dict
            state_dict = torch.load(pth_path, weights_only=False)

            # 创建新模型
            self.create_model()

            # 加载参数
            self.model.policy.load_state_dict(state_dict['policy_state_dict'])
            print(f"✅ SB3 PPO model loaded from state_dict: {pth_path}")
        else:
            # 标准SB3格式
            self.model = PPO.load(path, env=self.vec_env)
            print(f"SB3 PPO model loaded from: {path}")

        return self.model


def test_sb3_ppo():
    """测试SB3 PPO"""
    print("Testing SB3 PPO...")
    
    # 创建基线
    baseline = SB3PPOBaseline()
    
    # 训练
    baseline.train(total_timesteps=50000)
    
    # 评估
    results = baseline.evaluate(n_episodes=10)
    print(f"SB3 PPO Results: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    
    # 保存
    baseline.save("../../../Models/sb3_ppo_test.zip")


if __name__ == "__main__":
    test_sb3_ppo()