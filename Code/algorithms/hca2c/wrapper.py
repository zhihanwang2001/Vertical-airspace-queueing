"""
Hierarchical Environment Wrapper for HCA2C

Transforms the flat observation space into a hierarchical structure:
- Global state: System-level metrics
- Layer states: Per-layer information with neighbor context
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces


class HierarchicalEnvWrapper(gym.Wrapper):
    """
    Hierarchical Environment Wrapper

    Transforms observations from the base environment into a hierarchical structure
    suitable for the HCA2C algorithm.

    Global State (6 dimensions):
        - system_metrics (3): total_load, avg_utilization, stability_indicator
        - total_queue (1): sum of all queue lengths
        - avg_utilization (1): mean utilization across layers
        - prev_reward (1): previous step reward

    Layer State (8 dimensions per layer):
        - queue_length (1)
        - capacity (1)
        - utilization_rate (1)
        - service_rate (1)
        - load_rate (1)
        - neighbor_pressure_up (1): utilization of layer above
        - neighbor_pressure_down (1): utilization of layer below
        - queue_change (1): rate of change in queue length
    """

    def __init__(self, env, capacities: np.ndarray = None):
        super().__init__(env)

        self.n_layers = 5

        # Default capacities (inverted pyramid)
        if capacities is None:
            self.capacities = np.array([8, 6, 4, 3, 2], dtype=np.float32)
        else:
            self.capacities = np.array(capacities, dtype=np.float32)

        # Define hierarchical observation space
        self.global_state_dim = 6
        self.layer_state_dim = 8

        self.observation_space = spaces.Dict({
            'global': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.global_state_dim,),
                dtype=np.float32
            ),
            'layers': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.n_layers, self.layer_state_dim),
                dtype=np.float32
            )
        })

        # Keep original action space but we'll transform actions internally
        # Action space: 11 dimensions
        # - service_intensities: 5 (one per layer)
        # - arrival_multiplier: 1
        # - emergency_transfers: 5 (one per layer)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(11,),
            dtype=np.float32
        )

        # Store previous observation for computing changes
        self._prev_obs = None

    def reset(self, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Reset environment and return hierarchical observation"""
        obs, info = self.env.reset(**kwargs)
        self._prev_obs = obs

        hier_obs = self._transform_observation(obs)
        return hier_obs, info

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict]:
        """
        Execute action and return hierarchical observation

        Args:
            action: Flat action vector [11] from HCA2C policy

        Returns:
            hier_obs: Hierarchical observation
            reward: Scalar reward
            terminated: Episode termination flag
            truncated: Episode truncation flag
            info: Additional information
        """
        # Transform action to environment format
        env_action = self._transform_action(action)

        # Step environment
        obs, reward, terminated, truncated, info = self.env.step(env_action)

        # Transform observation
        hier_obs = self._transform_observation(obs, self._prev_obs)
        self._prev_obs = obs

        # Add hierarchical info
        info['hierarchical'] = {
            'global_state': hier_obs['global'],
            'layer_states': hier_obs['layers']
        }

        return hier_obs, reward, terminated, truncated, info

    def _transform_observation(self, obs: Dict, prev_obs: Dict = None) -> Dict[str, np.ndarray]:
        """
        Transform flat observation to hierarchical structure

        Args:
            obs: Original observation dictionary
            prev_obs: Previous observation (for computing changes)

        Returns:
            Hierarchical observation dictionary
        """
        # Extract global state
        global_state = self._extract_global_state(obs)

        # Extract layer states
        layer_states = np.zeros((self.n_layers, self.layer_state_dim), dtype=np.float32)

        for i in range(self.n_layers):
            layer_states[i] = self._extract_layer_state(obs, i, prev_obs)

        return {
            'global': global_state,
            'layers': layer_states
        }

    def _extract_global_state(self, obs: Dict) -> np.ndarray:
        """Extract global state from observation"""
        # Handle both dict and flat observations
        if isinstance(obs, dict):
            system_metrics = obs.get('system_metrics', np.zeros(3))
            queue_lengths = obs.get('queue_lengths', np.zeros(self.n_layers))
            utilization_rates = obs.get('utilization_rates', np.zeros(self.n_layers))
            prev_reward = obs.get('prev_reward', np.array([0.0]))
        else:
            # Flat observation - parse based on known structure
            system_metrics = np.zeros(3)
            queue_lengths = np.zeros(self.n_layers)
            utilization_rates = np.zeros(self.n_layers)
            prev_reward = np.array([0.0])

        # Compute global metrics
        total_queue = np.sum(queue_lengths) / np.sum(self.capacities)  # Normalized
        avg_utilization = np.mean(utilization_rates)

        global_state = np.concatenate([
            system_metrics.flatten()[:3],  # 3 dims
            [total_queue],                  # 1 dim
            [avg_utilization],              # 1 dim
            prev_reward.flatten()[:1]       # 1 dim
        ]).astype(np.float32)

        return global_state

    def _extract_layer_state(self, obs: Dict, layer_idx: int,
                            prev_obs: Dict = None) -> np.ndarray:
        """Extract state for a specific layer"""
        i = layer_idx

        if isinstance(obs, dict):
            queue_length = obs.get('queue_lengths', np.zeros(self.n_layers))[i]
            utilization = obs.get('utilization_rates', np.zeros(self.n_layers))[i]
            service_rate = obs.get('service_rates', np.ones(self.n_layers))[i]
            load_rate = obs.get('load_rates', np.zeros(self.n_layers))[i]
            queue_change = obs.get('queue_changes', np.zeros(self.n_layers))[i]
            utilization_rates = obs.get('utilization_rates', np.zeros(self.n_layers))
        else:
            # Default values for flat observation
            queue_length = 0.0
            utilization = 0.0
            service_rate = 1.0
            load_rate = 0.0
            queue_change = 0.0
            utilization_rates = np.zeros(self.n_layers)

        # Neighbor pressures
        neighbor_up = utilization_rates[i - 1] if i > 0 else 0.0
        neighbor_down = utilization_rates[i + 1] if i < self.n_layers - 1 else 0.0

        # Normalize queue length by capacity
        normalized_queue = queue_length / self.capacities[i]

        layer_state = np.array([
            normalized_queue,           # Normalized queue length
            self.capacities[i] / 8.0,   # Normalized capacity (max=8)
            utilization,                # Utilization rate
            service_rate / 2.0,         # Normalized service rate
            load_rate / 5.0,            # Normalized load rate
            neighbor_up,                # Upper neighbor pressure
            neighbor_down,              # Lower neighbor pressure
            queue_change                # Queue change rate
        ], dtype=np.float32)

        return layer_state

    def _transform_action(self, action: np.ndarray) -> Dict:
        """
        Transform flat action vector to environment action dictionary

        Args:
            action: Flat action vector [11] with values in [-1, 1]
                - [0:5]: service intensities
                - [5]: arrival multiplier
                - [6:11]: emergency transfers

        Returns:
            Environment action dictionary
        """
        action = np.clip(action, -1.0, 1.0)

        # Service intensities: [-1, 1] -> [0.5, 2.0] (higher minimum for stability)
        # Changed from [0.1, 2.0] to [0.5, 2.0] to prevent immediate crashes
        service_intensities = (action[:5] + 1) / 2 * 1.5 + 0.5

        # Arrival multiplier: [-1, 1] -> [0.5, 3.0] (reduced max for stability)
        # Changed from [0.5, 5.0] to [0.5, 3.0] to prevent overload
        arrival_multiplier = (action[5] + 1) / 2 * 2.5 + 0.5

        # Emergency transfers: [-1, 1] -> {0, 1}
        emergency_transfers = (action[6:11] > 0).astype(np.int8)

        return {
            'service_intensities': service_intensities.astype(np.float32),
            'arrival_multiplier': np.array([arrival_multiplier], dtype=np.float32),
            'emergency_transfers': emergency_transfers
        }

    def get_flat_observation(self, hier_obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Convert hierarchical observation back to flat vector"""
        return np.concatenate([
            hier_obs['global'],
            hier_obs['layers'].flatten()
        ])

    @property
    def flat_obs_dim(self) -> int:
        """Total dimension of flattened observation"""
        return self.global_state_dim + self.n_layers * self.layer_state_dim


class FlatHierarchicalWrapper(gym.Wrapper):
    """
    Wrapper that provides hierarchical structure but with flat observation space
    for compatibility with standard RL algorithms.

    This is useful for comparing HCA2C with standard A2C/PPO using the same
    observation preprocessing.
    """

    def __init__(self, env, capacities: np.ndarray = None):
        # First wrap with hierarchical wrapper
        self.hier_wrapper = HierarchicalEnvWrapper(env, capacities)
        super().__init__(self.hier_wrapper)

        # Override observation space to be flat
        flat_dim = self.hier_wrapper.flat_obs_dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(flat_dim,),
            dtype=np.float32
        )

    def reset(self, **kwargs):
        hier_obs, info = self.hier_wrapper.reset(**kwargs)
        flat_obs = self.hier_wrapper.get_flat_observation(hier_obs)
        return flat_obs, info

    def step(self, action):
        hier_obs, reward, terminated, truncated, info = self.hier_wrapper.step(action)
        flat_obs = self.hier_wrapper.get_flat_observation(hier_obs)
        return flat_obs, reward, terminated, truncated, info


if __name__ == "__main__":
    # Test the wrapper
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

    from env.drl_optimized_env_fixed import DRLOptimizedQueueEnvFixed

    print("Testing HierarchicalEnvWrapper...")

    # Create base environment
    base_env = DRLOptimizedQueueEnvFixed()

    # Wrap with hierarchical wrapper
    hier_env = HierarchicalEnvWrapper(base_env)

    print(f"✅ Global state dim: {hier_env.global_state_dim}")
    print(f"✅ Layer state dim: {hier_env.layer_state_dim}")
    print(f"✅ Total flat dim: {hier_env.flat_obs_dim}")
    print(f"✅ Action space: {hier_env.action_space}")

    # Test reset
    obs, info = hier_env.reset()
    print(f"\n✅ Global state shape: {obs['global'].shape}")
    print(f"✅ Layer states shape: {obs['layers'].shape}")
    print(f"✅ Global state: {obs['global']}")
    print(f"✅ Layer 0 state: {obs['layers'][0]}")

    # Test step
    action = hier_env.action_space.sample()
    obs, reward, terminated, truncated, info = hier_env.step(action)
    print(f"\n✅ After step:")
    print(f"   Reward: {reward:.4f}")
    print(f"   Terminated: {terminated}")
    print(f"   Global state: {obs['global']}")

    # Test flat wrapper
    print("\n\nTesting FlatHierarchicalWrapper...")
    flat_env = FlatHierarchicalWrapper(DRLOptimizedQueueEnvFixed())
    obs, info = flat_env.reset()
    print(f"✅ Flat observation shape: {obs.shape}")

    print("\n✅ All wrapper tests passed!")
