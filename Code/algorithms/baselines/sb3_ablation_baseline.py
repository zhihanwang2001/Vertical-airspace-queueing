"""
消融实验基线算法
Ablation Study Baseline Algorithm

为消融实验创建的特殊PPO基线，支持：
1. 动态配置修改（高层优先、容量结构、转移机制等）
2. 单目标vs多目标奖励函数切换  
3. 组件级别的开关控制
4. 与完整系统公平对比的实验设置

基于sb3_ppo_baseline.py，添加消融实验特定功能
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
from ablation_configs import AblationConfigs, AblationEnvironmentFactory


class AblationLearningRateLogger(BaseCallback):
    """消融实验学习率记录器"""
    
    def __init__(self, initial_lr: float = 3e-4, min_lr: float = 1e-6, 
                 ablation_type: str = "full_system", verbose: int = 1):
        super().__init__(verbose)
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.ablation_type = ablation_type
        
    def _on_step(self) -> bool:
        """记录当前学习率和消融实验信息到TensorBoard"""
        current_lr = self.model.policy.optimizer.param_groups[0]['lr']
        progress_remaining = getattr(self.model, '_current_progress_remaining', 1.0)
        progress = 1.0 - progress_remaining
        
        # 记录基础学习率信息
        self.logger.record("train/learning_rate", current_lr)
        self.logger.record("train/lr_progress", progress)
        self.logger.record("train/lr_decay_ratio", current_lr / self.initial_lr)
        
        # 记录消融实验类型
        self.logger.record("ablation/experiment_type", self.ablation_type)
        
        # 定期打印
        if self.num_timesteps % 10000 == 0 and self.verbose > 0:
            print(f"[{self.ablation_type}] Step {self.num_timesteps:6,}: LR={current_lr:.6f}")
        
        return True


def apply_ablation_config_to_env(env, ablation_config):
    """直接应用消融配置到环境，不使用包装器"""
    
    # 1. 修改到达权重（无高层优先实验）
    env.arrival_weights = np.array(ablation_config.arrival_weights, dtype=np.float32)
    
    # 2. 修改容量配置（传统金字塔实验）
    env.capacities = np.array(ablation_config.layer_capacities, dtype=np.int32)
    
    # 3. 修改服务率
    env.base_service_rates = np.array(ablation_config.layer_service_rates, dtype=np.float32)
    
    # 4. 处理奖励函数修改（单目标实验）
    if hasattr(ablation_config, '_reward_type') and ablation_config._reward_type == 'throughput_only':
        env._single_objective_mode = True
        
    # 5. 处理转移机制（无转移实验）
    if hasattr(ablation_config, '_transfer_enabled') and not ablation_config._transfer_enabled:
        env._transfer_disabled = True
        
    ablation_type = getattr(ablation_config, '_ablation_type', 'unknown')
    print(f"✅ 应用消融修改: {ablation_type}")
    if hasattr(ablation_config, '_removed_component'):
        print(f"   移除组件: {ablation_config._removed_component}")
    
    return env


class AblationEnvironmentWrapper:
    """消融实验环境包装器（已弃用，保留兼容性）"""
    
    def __init__(self, base_env, ablation_config):
        self.base_env = base_env
        self.ablation_config = ablation_config
        self.ablation_type = getattr(ablation_config, '_ablation_type', 'full_system')
        
        # 应用消融修改
        self._apply_ablation_modifications()
    
    def _apply_ablation_modifications(self):
        """应用消融实验的修改"""
        
        # 1. 修改到达权重（无高层优先实验）
        self.base_env.arrival_weights = np.array(self.ablation_config.arrival_weights, dtype=np.float32)
        
        # 2. 修改容量配置（传统金字塔实验）
        self.base_env.capacities = np.array(self.ablation_config.layer_capacities, dtype=np.int32)
        
        # 3. 修改服务率
        self.base_env.base_service_rates = np.array(self.ablation_config.layer_service_rates, dtype=np.float32)
        
        # 4. 处理奖励函数修改（单目标实验）
        if hasattr(self.ablation_config, '_reward_type') and self.ablation_config._reward_type == 'throughput_only':
            self.base_env._single_objective_mode = True
            
        # 5. 处理转移机制（无转移实验）
        if hasattr(self.ablation_config, '_transfer_enabled') and not self.ablation_config._transfer_enabled:
            self.base_env._transfer_disabled = True
            
        print(f"✅ 应用消融修改: {self.ablation_type}")
        if hasattr(self.ablation_config, '_removed_component'):
            print(f"   移除组件: {self.ablation_config._removed_component}")
    
    def __getattr__(self, name):
        """代理到基础环境"""
        return getattr(self.base_env, name)


class SB3AblationBaseline:
    """消融实验PPO基线算法"""
    
    def __init__(self, ablation_type: str = "full_system", config=None):
        """
        初始化消融实验基线
        
        Args:
            ablation_type: 消融实验类型
                - "full_system": 完整系统（对照组）
                - "no_high_priority": 无高层优先
                - "single_objective": 单目标优化
                - "traditional_pyramid": 传统金字塔
                - "no_transfer": 无转移机制
            config: 额外的PPO配置参数
        """
        
        self.ablation_type = ablation_type
        
        # 获取消融配置
        all_configs = AblationConfigs.get_all_ablation_configs()
        if ablation_type not in all_configs:
            raise ValueError(f"未知的消融类型: {ablation_type}")
        
        self.ablation_config = all_configs[ablation_type]
        
        # PPO配置
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
        
        print(f"🧪 初始化消融实验: {ablation_type}")
        if hasattr(self.ablation_config, '_removed_component'):
            print(f"   移除组件: {self.ablation_config._removed_component}")
        
    def setup_env(self):
        """设置消融实验环境"""
        base_env = DRLOptimizedQueueEnvFixed()
        
        # 直接应用消融配置
        apply_ablation_config_to_env(base_env, self.ablation_config)
        
        # 包装环境
        wrapped_env = SB3DictWrapper(base_env)
        self.env = Monitor(wrapped_env, filename=None)
        
        # 创建向量化环境
        self.vec_env = DummyVecEnv([lambda: self.env])
        
        print(f"✅ 消融环境设置完成: {self.ablation_type}")
        return self.env
    
    def create_model(self):
        """创建消融实验PPO模型"""
        if self.env is None:
            self.setup_env()
        
        # 余弦退火学习率调度（与完整系统保持一致）
        def cosine_annealing_schedule(progress_remaining):
            initial_lr = self.config['learning_rate']
            min_lr = self.config.get('min_lr', 1e-6)
            progress = 1.0 - progress_remaining
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            current_lr = min_lr + (initial_lr - min_lr) * cosine_factor
            return current_lr
        
        # 创建PPO模型
        self.model = PPO(
            "MlpPolicy",
            self.vec_env,
            learning_rate=cosine_annealing_schedule,
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
        
        print(f"✅ 消融PPO模型创建完成: {self.ablation_type}")
        return self.model
    
    def train(self, total_timesteps, eval_freq=10000, save_freq=50000):
        """训练消融实验模型 - 简化版本避免pickle错误"""
        if self.model is None:
            self.create_model()
        
        # 开始训练
        print(f"🚀 开始消融实验训练: {self.ablation_type}")
        print(f"   训练步数: {total_timesteps:,}")
        if hasattr(self.ablation_config, '_removed_component'):
            print(f"   移除组件: {self.ablation_config._removed_component}")
        
        # 使用简化的训练，不使用复杂的callback
        self.model.learn(
            total_timesteps=total_timesteps,
            log_interval=10
        )
        
        print(f"✅ 消融实验训练完成: {self.ablation_type}")
        
        return self.model
    
    def evaluate(self, n_episodes=10):
        """评估消融实验模型性能 - 简化版本"""
        if self.model is None:
            raise ValueError("模型未训练，请先调用train()")
        
        print(f"📊 评估消融实验: {self.ablation_type}")
        
        episode_rewards = []
        
        for episode in range(n_episodes):
            obs = self.vec_env.reset()
            total_reward = 0
            done = False
            step_count = 0
            
            while not done and step_count < 1000:  # 限制最大步数
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.vec_env.step(action)
                total_reward += reward[0]
                step_count += 1
            
            episode_rewards.append(total_reward)
            
            if episode % 5 == 0:
                print(f"   Episode {episode+1}/{n_episodes}: Reward={total_reward:.2f}")
        
        # 计算统计量
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        
        results = {
            'ablation_type': self.ablation_type,
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'n_episodes': n_episodes,
            'removed_component': getattr(self.ablation_config, '_removed_component', 'None')
        }
        
        print(f"📈 {self.ablation_type} 评估结果:")
        print(f"   平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")
        
        return results


# 消融实验管理器
class AblationExperimentManager:
    """消融实验管理器"""
    
    def __init__(self, total_timesteps=100000):
        self.total_timesteps = total_timesteps
        self.results = {}
        
    def run_all_ablation_experiments(self):
        """运行所有消融实验"""
        ablation_types = [
            'full_system',
            'no_high_priority', 
            'single_objective',
            'traditional_pyramid',
            'no_transfer'
        ]
        
        print(f"🧪 开始完整消融实验研究")
        print(f"   实验数量: {len(ablation_types)}")
        print(f"   每个实验训练步数: {self.total_timesteps:,}")
        print("=" * 60)
        
        for i, ablation_type in enumerate(ablation_types, 1):
            print(f"\n🎯 执行实验 {i}/{len(ablation_types)}: {ablation_type}")
            print("-" * 40)
            
            try:
                # 创建并训练模型
                baseline = SB3AblationBaseline(ablation_type)
                baseline.train(self.total_timesteps)
                
                # 评估性能
                results = baseline.evaluate(n_episodes=20)
                self.results[ablation_type] = results
                
                print(f"✅ {ablation_type} 实验完成")
                
            except Exception as e:
                print(f"❌ {ablation_type} 实验失败: {str(e)}")
                self.results[ablation_type] = {'error': str(e)}
        
        print(f"\n🎉 消融实验研究完成!")
        self._print_comparison_results()
        
        return self.results
    
    def _print_comparison_results(self):
        """打印对比结果"""
        print(f"\n📊 消融实验对比结果:")
        print("=" * 80)
        print(f"{'实验类型':<20} {'平均奖励':<15} {'标准差':<10} {'性能下降':<10} {'移除组件'}")
        print("-" * 80)
        
        full_system_reward = self.results.get('full_system', {}).get('mean_reward', 0)
        
        for ablation_type, result in self.results.items():
            if 'error' in result:
                print(f"{ablation_type:<20} {'ERROR':<15} {'-':<10} {'-':<10} {'-'}")
                continue
                
            mean_reward = result.get('mean_reward', 0)
            std_reward = result.get('std_reward', 0)
            removed_component = result.get('removed_component', 'None')
            
            if ablation_type == 'full_system':
                performance_drop = '0.0%'
            else:
                if full_system_reward > 0:
                    drop_percent = (full_system_reward - mean_reward) / full_system_reward * 100
                    performance_drop = f"{drop_percent:.1f}%"
                else:
                    performance_drop = 'N/A'
            
            print(f"{ablation_type:<20} {mean_reward:<15.2f} {std_reward:<10.2f} "
                  f"{performance_drop:<10} {removed_component}")
        
        print("-" * 80)
    
    def save_results(self, filepath="ablation_results.json"):
        """保存实验结果"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"💾 实验结果保存至: {filepath}")


# 测试和示例用法
if __name__ == "__main__":
    print("🧪 消融实验基线测试")
    
    # 测试单个消融实验
    print("\n1. 测试单个消融实验...")
    baseline = SB3AblationBaseline("no_high_priority")
    
    # 快速测试训练
    print("   快速训练测试...")
    baseline.train(total_timesteps=1000)  # 快速测试
    
    # 评估
    print("   评估测试...")
    results = baseline.evaluate(n_episodes=3)
    
    print(f"✅ 单个消融实验测试完成!")
    print(f"   结果: {results}")