"""
DRL Wrapper - Fixed Version

Wraps the fixed environment to make it compatible with PPO training
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Union, Any

class DictToBoxActionWrapperFixed(gym.ActionWrapper):
    """Fixed version action space wrapper"""

    def __init__(self, env):
        super().__init__(env)

        # Action dimensions
        self.service_dim = 5
        self.arrival_dim = 1
        self.transfer_dim = 5
        self.total_dim = self.service_dim + self.arrival_dim + self.transfer_dim

        # Box action space
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

        print(f"✅ Fixed version action space conversion: Dict -> Box({self.total_dim} dimensions)")

    def action(self, action: np.ndarray) -> Dict[str, Union[np.ndarray, float]]:
        """Convert Box action to Dict action"""

        service_intensities = action[:self.service_dim].astype(np.float32)
        arrival_multiplier = action[self.service_dim:self.service_dim+self.arrival_dim].astype(np.float32)
        emergency_transfers_raw = action[self.service_dim+self.arrival_dim:]

        # Binary conversion
        emergency_transfers = (emergency_transfers_raw > 0.5).astype(np.int8)

        return {
            'service_intensities': service_intensities,
            'arrival_multiplier': arrival_multiplier,
            'emergency_transfers': emergency_transfers
        }

class ObservationWrapperFixed(gym.ObservationWrapper):
    """Fixed version observation space wrapper"""

    def __init__(self, env):
        super().__init__(env)

        # Calculate total observation dimensions
        self.obs_dims = {}
        total_dim = 0

        # Define observation components in fixed order
        obs_components = [
            'queue_lengths',      # 5 dimensions
            'utilization_rates',  # 5 dimensions
            'queue_changes',      # 5 dimensions
            'load_rates',         # 5 dimensions
            'service_rates',      # 5 dimensions
            'prev_reward',        # 1 dimension
            'system_metrics'      # 3 dimensions
        ]

        # Calculate dimensions
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

        # Create flattened observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_dim,),
            dtype=np.float32
        )

        print(f"✅ Fixed version observation space conversion: Dict -> Box({total_dim} dimensions)")
        for component, (start, end) in self.obs_dims.items():
            print(f"   - {component}: dimensions {start}-{end-1}")

    def observation(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Convert Dict observation to flattened array"""
        flat_obs = []

        # Extract observations in fixed order
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
    """Create fixed version wrapped environment"""
    try:
        from .drl_optimized_env_fixed import create_fixed_drl_environment
    except ImportError:
        from drl_optimized_env_fixed import create_fixed_drl_environment

    # Create fixed version environment
    base_env = create_fixed_drl_environment()

    # Add wrappers
    wrapped_env = DictToBoxActionWrapperFixed(base_env)
    wrapped_env = ObservationWrapperFixed(wrapped_env)

    return wrapped_env

def test_fixed_wrapper():
    """Test fixed version wrapper"""
    print("🧪 Testing fixed version DRL wrapper")
    print("=" * 60)

    env = create_wrapped_fixed_environment()

    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    # Test run
    obs, info = env.reset()
    print(f"Initial observation dimensions: {obs.shape}")
    print(f"Initial info keys: {list(info.keys())}")

    for step in range(3):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)

        print(f"\nStep {step+1}:")
        print(f"Action: {action[:6]}")
        print(f"Reward: {reward:.3f}")
        print(f"Stability: {info.get('stability_score', 0):.3f}")
        print(f"Throughput: {info.get('throughput', 0):.3f}")

        if term or trunc:
            break

    print("\n✅ Fixed version wrapper testing complete")

if __name__ == "__main__":
    test_fixed_wrapper()