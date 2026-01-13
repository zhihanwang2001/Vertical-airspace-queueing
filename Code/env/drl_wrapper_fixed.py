"""
DRL包装器 - 修复版本
DRL Wrapper for Fixed Environment

包装修复后的环境，使其兼容PPO训练
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Union, Any

class DictToBoxActionWrapperFixed(gym.ActionWrapper):
    """修复版动作空间包装器"""
    
    def __init__(self, env):
        super().__init__(env)
        
        # 动作维度
        self.service_dim = 5
        self.arrival_dim = 1
        self.transfer_dim = 5
        self.total_dim = self.service_dim + self.arrival_dim + self.transfer_dim
        
        # Box动作空间
        low = np.concatenate([
            np.full(self.service_dim, 0.1),
            np.full(self.arrival_dim, 0.5),
            np.full(self.transfer_dim, 0.0)
        ])
        
        high = np.concatenate([
            np.full(self.service_dim, 2.0),
            np.full(self.arrival_dim, 5.0),
            np.full(self.transfer_dim, 1.0)
        ])
        
        self.action_space = spaces.Box(
            low=low, 
            high=high, 
            shape=(self.total_dim,), 
            dtype=np.float32
        )
        
        print(f"✅ 修复版动作空间转换: Dict -> Box({self.total_dim}维)")
    
    def action(self, action: np.ndarray) -> Dict[str, Union[np.ndarray, float]]:
        """转换Box动作为Dict动作"""
        
        service_intensities = action[:self.service_dim].astype(np.float32)
        arrival_multiplier = action[self.service_dim:self.service_dim+self.arrival_dim].astype(np.float32)
        emergency_transfers_raw = action[self.service_dim+self.arrival_dim:]
        
        # 二进制转换
        emergency_transfers = (emergency_transfers_raw > 0.5).astype(np.int8)
        
        return {
            'service_intensities': service_intensities,
            'arrival_multiplier': arrival_multiplier,
            'emergency_transfers': emergency_transfers
        }

class ObservationWrapperFixed(gym.ObservationWrapper):
    """修复版观测空间包装器"""
    
    def __init__(self, env):
        super().__init__(env)
        
        # 计算总观测维度
        self.obs_dims = {}
        total_dim = 0
        
        # 按固定顺序定义观测组件
        obs_components = [
            'queue_lengths',      # 5维
            'utilization_rates',  # 5维  
            'queue_changes',      # 5维
            'load_rates',         # 5维
            'service_rates',      # 5维
            'prev_reward',        # 1维
            'system_metrics'      # 3维
        ]
        
        # 计算维度
        component_dims = {
            'queue_lengths': 5,
            'utilization_rates': 5,
            'queue_changes': 5,
            'load_rates': 5,
            'service_rates': 5,
            'prev_reward': 1,
            'system_metrics': 3
        }
        
        for component in obs_components:
            dim = component_dims[component]
            self.obs_dims[component] = (total_dim, total_dim + dim)
            total_dim += dim
        
        # 创建扁平化观测空间
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_dim,),
            dtype=np.float32
        )
        
        print(f"✅ 修复版观测空间转换: Dict -> Box({total_dim}维)")
        for component, (start, end) in self.obs_dims.items():
            print(f"   - {component}: 维度 {start}-{end-1}")
    
    def observation(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """转换Dict观测为扁平化数组"""
        flat_obs = []
        
        # 按固定顺序提取观测
        obs_order = ['queue_lengths', 'utilization_rates', 'queue_changes', 
                    'load_rates', 'service_rates', 'prev_reward', 'system_metrics']
        
        for key in obs_order:
            if key in obs:
                value = obs[key]
                if isinstance(value, np.ndarray):
                    flat_obs.extend(value.flatten())
                else:
                    flat_obs.append(float(value))
        
        return np.array(flat_obs, dtype=np.float32)

def create_wrapped_fixed_environment():
    """创建修复版包装环境"""
    try:
        from .drl_optimized_env_fixed import create_fixed_drl_environment
    except ImportError:
        from drl_optimized_env_fixed import create_fixed_drl_environment
    
    # 创建修复版环境
    base_env = create_fixed_drl_environment()
    
    # 添加包装器
    wrapped_env = DictToBoxActionWrapperFixed(base_env)
    wrapped_env = ObservationWrapperFixed(wrapped_env)
    
    return wrapped_env

def test_fixed_wrapper():
    """测试修复版包装器"""
    print("🧪 测试修复版DRL包装器")
    print("=" * 60)
    
    env = create_wrapped_fixed_environment()
    
    print(f"动作空间: {env.action_space}")
    print(f"观测空间: {env.observation_space}")
    
    # 测试运行
    obs, info = env.reset()
    print(f"初始观测维度: {obs.shape}")
    print(f"初始info keys: {list(info.keys())}")
    
    for step in range(3):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        
        print(f"\nStep {step+1}:")
        print(f"动作: {action[:6]}")
        print(f"奖励: {reward:.3f}")
        print(f"稳定性: {info.get('stability_score', 0):.3f}")
        print(f"吞吐率: {info.get('throughput', 0):.3f}")
        
        if term or trunc:
            break
    
    print("\n✅ 修复版包装器测试完成")

if __name__ == "__main__":
    test_fixed_wrapper()