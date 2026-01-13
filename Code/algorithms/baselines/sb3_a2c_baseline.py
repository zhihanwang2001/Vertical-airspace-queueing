"""
SB3 A2C算法基线
SB3 A2C Baseline Algorithm
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
import math
from env.drl_optimized_env_fixed import DRLOptimizedQueueEnvFixed
from .space_utils import SB3DictWrapper


class SB3A2CBaseline:
    """SB3 A2C基线算法"""
    
    def __init__(self, config=None):
        # 🔧 优化配置v2 - 添加延迟余弦退火学习率调度解决训练不稳定问题
        default_config = {
            # 学习参数优化
            'initial_lr': 7e-4,             # 🔧 初始学习率（前300k步保持）
            'min_lr': 1e-5,                 # 🔧 最终学习率（500k步降至）
            'warmup_steps': 300000,         # 🔧 前300k步保持固定lr充分探索
            'total_steps': 500000,          # 🔧 总训练步数
            'n_steps': 32,                  # 🔧 优化: 5 → 32 (增加rollout长度改进优势估计)
            'gamma': 0.99,                  # ✅ 保持不变
            'gae_lambda': 0.95,             # 🔧 优化: 1.0 → 0.95 (bias-variance权衡)

            # 探索与价值函数
            'ent_coef': 0.01,               # 🔧 优化: 0.0 → 0.01 (添加熵正则化促进探索)
            'vf_coef': 0.5,                 # ✅ 保持不变
            'max_grad_norm': 0.5,           # ✅ 保持不变

            # 优化器配置
            'rms_prop_eps': 1e-5,           # ✅ 保持不变
            'use_rms_prop': True,           # ✅ 保持不变
            'use_sde': False,               # ✅ 保持不变
            'normalize_advantage': True,    # 🔧 优化: False → True (归一化优势减少方差)

            # 其他配置
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
        """创建A2C模型"""
        if self.env is None:
            self.setup_env()

        # 🔧 创建延迟余弦退火学习率调度函数（前300k步保持固定lr，之后余弦下降）
        def delayed_cosine_annealing_schedule(progress_remaining):
            """
            延迟余弦退火学习率调度
            前300k步: 保持固定学习率7e-4 (充分探索)
            300k步后: 余弦退火下降到1e-5 (稳定收敛)

            progress_remaining: 1.0 -> 0.0 (从开始到结束)
            """
            initial_lr = self.config.get('initial_lr', 7e-4)
            min_lr = self.config.get('min_lr', 1e-5)
            warmup_steps = self.config.get('warmup_steps', 300000)  # 前300k步保持固定lr
            total_steps = self.config.get('total_steps', 500000)     # 总训练步数

            # 计算当前步数
            current_step = int((1.0 - progress_remaining) * total_steps)

            # 前300k步: 保持固定学习率
            if current_step < warmup_steps:
                return initial_lr

            # 300k步后: 余弦退火 (将剩余200k步映射到0→1)
            annealing_progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * annealing_progress))
            current_lr = min_lr + (initial_lr - min_lr) * cosine_factor

            return current_lr

        # 🔧 优化: 增加网络容量提高表达能力
        policy_kwargs = dict(
            net_arch=dict(
                pi=[512, 512, 256],  # 🔧 Policy网络: 增加深度和宽度
                vf=[512, 512, 256]   # 🔧 Value网络: 独立的大容量网络
            )
        )

        # 创建A2C模型
        self.model = A2C(
            "MlpPolicy",
            self.vec_env,
            learning_rate=delayed_cosine_annealing_schedule,  # 🔧 v2: 使用延迟余弦退火调度
            n_steps=self.config['n_steps'],
            gamma=self.config['gamma'],
            gae_lambda=self.config['gae_lambda'],
            ent_coef=self.config['ent_coef'],
            vf_coef=self.config['vf_coef'],
            max_grad_norm=self.config['max_grad_norm'],
            rms_prop_eps=self.config['rms_prop_eps'],
            use_rms_prop=self.config['use_rms_prop'],
            use_sde=self.config['use_sde'],
            normalize_advantage=self.config['normalize_advantage'],
            policy_kwargs=policy_kwargs,           # 🔧 新增: 网络架构配置
            tensorboard_log=self.config['tensorboard_log'],
            verbose=self.config['verbose'],
            seed=self.config['seed'],
            device='auto'
        )
        
        print(f"SB3 A2C model created with device: {self.model.device}")
        return self.model
    
    def train(self, total_timesteps, eval_freq=10000, save_freq=50000):
        """训练模型"""
        if self.model is None:
            self.create_model()
        
        # 创建必要的目录
        os.makedirs('./logs/', exist_ok=True)
        os.makedirs('../../../Models/sb3_a2c_best/', exist_ok=True)
        os.makedirs('../../../Models/sb3_a2c_checkpoints/', exist_ok=True)
        
        # 创建评估环境
        eval_env = DummyVecEnv([lambda: Monitor(
            SB3DictWrapper(DRLOptimizedQueueEnvFixed()), 
            filename=None
        )])
        
        # 创建回调
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path='../../../Models/sb3_a2c_best/',
            log_path='./logs/',
            eval_freq=eval_freq,
            deterministic=True,
            render=False,
            n_eval_episodes=10,
            verbose=1
        )
        
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path='../../../Models/sb3_a2c_checkpoints/',
            name_prefix='sb3_a2c'
        )
        
        # 开始训练
        print(f"Starting SB3 A2C training for {total_timesteps:,} timesteps...")
        
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=None,  # 移除有问题的callbacks
            log_interval=10,
            tb_log_name="SB3_A2C"
        )
        
        print("SB3 A2C training completed!")
        
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
        
        print(f"SB3 A2C results saved to: {path_prefix}")
    
    def save(self, path):
        """保存模型"""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        try:
            # 尝试使用 exclude 参数来避免 pickle 错误
            # 排除环境对象，只保存模型参数
            self.model.save(path, exclude=['env', 'logger', 'ep_info_buffer', 'ep_success_buffer'])
            print(f"SB3 A2C model saved to: {path}")
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
            print(f"SB3 A2C model saved as state_dict to: {path}.pth")
    
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
            print(f"✅ SB3 A2C model loaded from state_dict: {pth_path}")
        else:
            # 标准SB3格式
            self.model = A2C.load(path, env=self.vec_env)
            print(f"SB3 A2C model loaded from: {path}")

        return self.model


def test_sb3_a2c():
    """测试SB3 A2C"""
    print("Testing SB3 A2C...")
    
    # 创建基线
    baseline = SB3A2CBaseline()
    
    # 训练
    baseline.train(total_timesteps=50000)
    
    # 评估
    results = baseline.evaluate(n_episodes=10)
    print(f"SB3 A2C Results: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    
    # 保存
    baseline.save("../../../Models/sb3_a2c_test.zip")


if __name__ == "__main__":
    test_sb3_a2c()