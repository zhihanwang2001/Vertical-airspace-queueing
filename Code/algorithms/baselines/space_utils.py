"""
空间处理工具函数
Utility functions for handling Dict observation and action spaces
"""

import numpy as np
from typing import Dict, Any, Union
try:
    import gymnasium as gym
except ImportError:
    import gym


def get_space_dimensions(space):
    """获取空间的维度"""
    if hasattr(space, 'spaces'):
        # Dict space
        total_dim = 0
        for subspace in space.spaces.values():
            if hasattr(subspace, 'shape'):
                total_dim += np.prod(subspace.shape)
            else:
                total_dim += 1
        return total_dim
    else:
        # Box space
        return space.shape[0] if hasattr(space, 'shape') else 1


def flatten_dict_observation(obs: Union[Dict, np.ndarray]) -> np.ndarray:
    """将字典观测扁平化为向量"""
    if isinstance(obs, dict):
        flattened = []
        for key in sorted(obs.keys()):
            value = obs[key]
            if isinstance(value, (list, np.ndarray)):
                flattened.extend(np.array(value).flatten())
            else:
                flattened.append(float(value))
        return np.array(flattened, dtype=np.float32)
    elif isinstance(obs, (list, np.ndarray)):
        return np.array(obs, dtype=np.float32).flatten()
    else:
        return np.array([obs], dtype=np.float32)


def dict_action_to_vector(action: Dict) -> np.ndarray:
    """将字典动作转换为向量形式（用于存储到replay buffer）"""
    vector = []
    
    # service_intensities: 映射回[-1,1]范围
    service_intensities = action.get('service_intensities', np.array([1.0]*5))
    service_raw = (service_intensities - 0.1) / (2.0 - 0.1) * 2 - 1
    vector.extend(service_raw)
    
    # arrival_multiplier: 映射回[-1,1]范围
    arrival_multiplier = action.get('arrival_multiplier', np.array([1.0]))
    if isinstance(arrival_multiplier, np.ndarray):
        arrival_val = arrival_multiplier[0]
    else:
        arrival_val = float(arrival_multiplier)
    arrival_raw = (arrival_val - 0.5) / (5.0 - 0.5) * 2 - 1
    vector.append(arrival_raw)
    
    # emergency_transfers: 映射到[-1,1]
    emergency_transfers = action.get('emergency_transfers', np.array([0]*5))
    emergency_raw = emergency_transfers * 2 - 1  # 0->-1, 1->1
    vector.extend(emergency_raw)
    
    return np.array(vector, dtype=np.float32)


def vector_to_dict_action(action_vector: np.ndarray) -> Dict:
    """将向量动作转换为字典格式"""
    action_vector = action_vector.flatten()
    idx = 0
    
    # service_intensities: 从[-1,1]映射到[0.1,2.0]
    service_intensities = action_vector[idx:idx+5]
    service_intensities = 0.1 + (service_intensities + 1) / 2 * (2.0 - 0.1)
    service_intensities = np.clip(service_intensities, 0.1, 2.0).astype(np.float32)
    idx += 5
    
    # arrival_multiplier: 从[-1,1]映射到[0.5,5.0]
    arrival_multiplier = action_vector[idx]
    arrival_multiplier = 0.5 + (arrival_multiplier + 1) / 2 * (5.0 - 0.5)
    arrival_multiplier = np.clip(arrival_multiplier, 0.5, 5.0)
    idx += 1
    
    # emergency_transfers: 从[-1,1]映射到{0,1}
    emergency_transfers = (action_vector[idx:idx+5] > 0).astype(np.int8)
    
    return {
        'service_intensities': service_intensities,
        'arrival_multiplier': np.array([arrival_multiplier], dtype=np.float32),
        'emergency_transfers': emergency_transfers
    }


def get_random_dict_action() -> Dict:
    """生成随机字典动作"""
    return {
        'service_intensities': np.random.uniform(0.1, 2.0, 5).astype(np.float32),
        'arrival_multiplier': np.random.uniform(0.5, 5.0, 1).astype(np.float32),
        'emergency_transfers': np.random.randint(0, 2, 5).astype(np.int8)
    }


def validate_dict_action(action: Dict) -> Dict:
    """验证并修正字典动作格式"""
    validated_action = {}
    
    # service_intensities
    service_intensities = action.get('service_intensities', np.ones(5))
    if not isinstance(service_intensities, np.ndarray):
        service_intensities = np.array(service_intensities)
    validated_action['service_intensities'] = np.clip(service_intensities, 0.1, 2.0).astype(np.float32)
    
    # arrival_multiplier
    arrival_multiplier = action.get('arrival_multiplier', 1.0)
    if isinstance(arrival_multiplier, (list, np.ndarray)):
        arrival_multiplier = arrival_multiplier[0] if len(arrival_multiplier) > 0 else 1.0
    arrival_multiplier = np.clip(float(arrival_multiplier), 0.5, 5.0)
    validated_action['arrival_multiplier'] = np.array([arrival_multiplier], dtype=np.float32)
    
    # emergency_transfers
    emergency_transfers = action.get('emergency_transfers', np.zeros(5))
    if not isinstance(emergency_transfers, np.ndarray):
        emergency_transfers = np.array(emergency_transfers)
    validated_action['emergency_transfers'] = np.clip(emergency_transfers, 0, 1).astype(np.int8)
    
    return validated_action


class SB3DictWrapper(gym.Wrapper):
    """
    SB3兼容的字典空间包装器
    将Dict观测和动作空间转换为Box空间以兼容SB3算法
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        # 计算观测空间维度
        obs_dim = get_space_dimensions(env.observation_space)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # 计算动作空间维度 (5+1+5=11)
        action_dim = get_space_dimensions(env.action_space)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )
        
        print(f"SB3DictWrapper: obs_dim={obs_dim}, action_dim={action_dim}")
    
    def reset(self, **kwargs):
        """重置环境并转换观测"""
        obs, info = self.env.reset(**kwargs)
        flat_obs = flatten_dict_observation(obs)
        return flat_obs, info
    
    def step(self, action):
        """执行步骤并转换观测"""
        # 将向量动作转换为字典格式
        dict_action = vector_to_dict_action(action)
        
        # 执行环境步骤
        obs, reward, terminated, truncated, info = self.env.step(dict_action)
        
        # 转换观测
        flat_obs = flatten_dict_observation(obs)
        
        return flat_obs, reward, terminated, truncated, info