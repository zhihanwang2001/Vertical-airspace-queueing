"""
可配置环境包装器
Configurable Environment Wrapper

用于跨区域泛化测试：支持动态修改环境参数以适配不同的heterogeneous configs
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces

import numpy as np
from env.drl_optimized_env_fixed import DRLOptimizedQueueEnvFixed
from env.config import VerticalQueueConfig


class ConfigurableEnvWrapper(gym.Wrapper):
    """
    可配置环境包装器

    接受VerticalQueueConfig并动态修改DRLOptimizedQueueEnvFixed的内部参数
    用于跨区域泛化性测试
    """

    def __init__(self, config: VerticalQueueConfig = None):
        """
        初始化可配置环境

        Args:
            config: VerticalQueueConfig实例（如果为None，使用默认配置）
        """
        # 创建基础环境
        base_env = DRLOptimizedQueueEnvFixed()
        super().__init__(base_env)

        # 如果提供了config，应用配置
        if config is not None:
            self._apply_config(config)
            self.config = config
        else:
            # 使用默认配置
            self.config = VerticalQueueConfig()

        print(f"✅ ConfigurableEnvWrapper initialized")
        print(f"   Arrival rate: {self.env.base_arrival_rate:.3f}")
        print(f"   Capacities: {self.env.capacities}")
        print(f"   Service rates: {self.env.base_service_rates}")

    def _apply_config(self, config: VerticalQueueConfig):
        """
        将VerticalQueueConfig的参数应用到环境

        这是核心方法：修改self.env的内部参数以匹配config
        """
        # 修改到达率
        self.env.base_arrival_rate = config.base_arrival_rate
        self.env.arrival_weights = np.array(config.arrival_weights, dtype=np.float32)

        # 修改容量
        self.env.capacities = np.array(config.layer_capacities, dtype=np.int32)

        # 修改服务率
        self.env.base_service_rates = np.array(config.layer_service_rates, dtype=np.float32)

        # 修改随机种子
        if hasattr(config, 'random_seed'):
            np.random.seed(config.random_seed)

        # 更新观测空间（因为容量可能改变了）
        self.env.observation_space = spaces.Dict({
            'queue_lengths': spaces.Box(
                low=0, high=max(self.env.capacities), shape=(self.env.n_layers,), dtype=np.float32
            ),
            'utilization_rates': spaces.Box(
                low=0.0, high=1.0, shape=(self.env.n_layers,), dtype=np.float32
            ),
            'queue_changes': spaces.Box(
                low=-1.0, high=1.0, shape=(self.env.n_layers,), dtype=np.float32
            ),
            'load_rates': spaces.Box(
                low=0.0, high=5.0, shape=(self.env.n_layers,), dtype=np.float32
            ),
            'service_rates': spaces.Box(
                low=0.0, high=10.0, shape=(self.env.n_layers,), dtype=np.float32
            ),
            'prev_reward': spaces.Box(
                low=-100.0, high=100.0, shape=(1,), dtype=np.float32
            ),
            'system_metrics': spaces.Box(
                low=0.0, high=10.0, shape=(3,), dtype=np.float32
            )
        })

        print(f"🔧 Config applied:")
        print(f"   Base arrival rate: {self.env.base_arrival_rate:.3f}")
        print(f"   Layer capacities: {self.env.capacities}")
        print(f"   Service rates: {self.env.base_service_rates}")

    def reset(self, seed=None, options=None):
        """重置环境"""
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        """执行一步"""
        return self.env.step(action)

    def get_config_summary(self) -> dict:
        """
        获取当前配置摘要

        Returns:
            dict: 配置摘要字典
        """
        return {
            'base_arrival_rate': float(self.env.base_arrival_rate),
            'total_capacity': int(np.sum(self.env.capacities)),
            'layer_capacities': self.env.capacities.tolist(),
            'service_rates': self.env.base_service_rates.tolist(),
            'arrival_weights': self.env.arrival_weights.tolist()
        }


if __name__ == "__main__":
    """测试可配置环境包装器"""
    print("\n" + "="*80)
    print("测试ConfigurableEnvWrapper")
    print("="*80 + "\n")

    # 测试1: 使用默认配置
    print("1. 测试默认配置:")
    env_default = ConfigurableEnvWrapper()
    obs, info = env_default.reset()
    print(f"   ✅ 默认环境创建成功")
    print(f"   观测空间: {obs.keys()}")

    # 测试2: 使用自定义配置
    print("\n2. 测试自定义配置 (Region B - Weather):")
    from env.config import VerticalQueueConfig
    custom_config = VerticalQueueConfig()
    # 模拟Region B: 服务率降低20%
    custom_config.layer_service_rates = [rate * 0.8 for rate in custom_config.layer_service_rates]

    env_custom = ConfigurableEnvWrapper(custom_config)
    obs, info = env_custom.reset()
    print(f"   ✅ 自定义环境创建成功")
    print(f"   配置摘要: {env_custom.get_config_summary()}")

    # 测试3: 运行几步
    print("\n3. 测试环境运行:")
    for step in range(5):
        action = env_custom.action_space.sample()
        obs, reward, terminated, truncated, info = env_custom.step(action)
        print(f"   Step {step+1}: Reward = {reward:.2f}")

    print("\n" + "="*80)
    print("✅ ConfigurableEnvWrapper测试完成！")
    print("="*80 + "\n")
