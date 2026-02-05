"""
Configurable Environment Wrapper

For cross-region generalization testing: supports dynamic modification of environment parameters to adapt to different heterogeneous configs
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
    Configurable Environment Wrapper

    Accepts VerticalQueueConfig and dynamically modifies internal parameters of DRLOptimizedQueueEnvFixed
    Used for cross-region generalization testing
    """

    def __init__(self, config: VerticalQueueConfig = None):
        """
        Initialize configurable environment

        Args:
            config: VerticalQueueConfig instance (if None, use default configuration)
        """
        # Create base environment
        base_env = DRLOptimizedQueueEnvFixed()
        super().__init__(base_env)

        # If config is provided, apply configuration
        if config is not None:
            self._apply_config(config)
            self.config = config
        else:
            # Use default configuration
            self.config = VerticalQueueConfig()

        print(f"✅ ConfigurableEnvWrapper initialized")
        print(f"   Arrival rate: {self.env.base_arrival_rate:.3f}")
        print(f"   Capacities: {self.env.capacities}")
        print(f"   Service rates: {self.env.base_service_rates}")

    def _apply_config(self, config: VerticalQueueConfig):
        """
        Apply VerticalQueueConfig parameters to environment

        This is the core method: modifies self.env's internal parameters to match config
        """
        # Modify arrival rate
        self.env.base_arrival_rate = config.base_arrival_rate
        self.env.arrival_weights = np.array(config.arrival_weights, dtype=np.float32)

        # Modify capacity
        self.env.capacities = np.array(config.layer_capacities, dtype=np.int32)

        # Modify service rate
        self.env.base_service_rates = np.array(config.layer_service_rates, dtype=np.float32)

        # Modify random seed
        if hasattr(config, 'random_seed'):
            np.random.seed(config.random_seed)

        # Update observation space (because capacity may have changed)
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
        """Reset environment"""
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        """Execute one step"""
        return self.env.step(action)

    def get_config_summary(self) -> dict:
        """
        Get current configuration summary

        Returns:
            dict: Configuration summary dictionary
        """
        return {
            'base_arrival_rate': float(self.env.base_arrival_rate),
            'total_capacity': int(np.sum(self.env.capacities)),
            'layer_capacities': self.env.capacities.tolist(),
            'service_rates': self.env.base_service_rates.tolist(),
            'arrival_weights': self.env.arrival_weights.tolist()
        }


if __name__ == "__main__":
    """Test configurable environment wrapper"""
    print("\n" + "="*80)
    print("Testing ConfigurableEnvWrapper")
    print("="*80 + "\n")

    # Test 1: Use default configuration
    print("1. Testing default configuration:")
    env_default = ConfigurableEnvWrapper()
    obs, info = env_default.reset()
    print(f"   ✅ Default environment created successfully")
    print(f"   Observation space: {obs.keys()}")

    # Test 2: Use custom configuration
    print("\n2. Testing custom configuration (Region B - Weather):")
    from env.config import VerticalQueueConfig
    custom_config = VerticalQueueConfig()
    # Simulate Region B: Service rate reduced by 20%
    custom_config.layer_service_rates = [rate * 0.8 for rate in custom_config.layer_service_rates]

    env_custom = ConfigurableEnvWrapper(custom_config)
    obs, info = env_custom.reset()
    print(f"   ✅ Custom environment created successfully")
    print(f"   Configuration summary: {env_custom.get_config_summary()}")

    # Test 3: Run a few steps
    print("\n3. Testing environment execution:")
    for step in range(5):
        action = env_custom.action_space.sample()
        obs, reward, terminated, truncated, info = env_custom.step(action)
        print(f"   Step {step+1}: Reward = {reward:.2f}")

    print("\n" + "="*80)
    print("✅ ConfigurableEnvWrapper testing complete!")
    print("="*80 + "\n")
