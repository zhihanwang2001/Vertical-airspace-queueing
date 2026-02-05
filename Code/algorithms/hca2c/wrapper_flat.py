"""
Flat Observation Space Wrapper for Ablation Study

This wrapper removes neighbor-aware features to test their contribution.
Uses a reduced observation space without explicit neighbor information.
"""

import numpy as np
from typing import Dict
from .wrapper import HierarchicalEnvWrapper


class FlatObservationWrapper(HierarchicalEnvWrapper):
    """
    Flat observation wrapper - removes neighbor information

    Changes from HierarchicalEnvWrapper:
    - Layer state: 8 dims -> 6 dims (removes neighbor_up, neighbor_down)
    - Total observation: 46 dims -> 36 dims

    This tests the contribution of neighbor-aware features to HCA2C performance.
    """

    def __init__(self, env, capacities=None):
        super().__init__(env, capacities)

        # Update layer state dimension
        self.layer_state_dim = 6  # Override from 8 to 6

        # Update observation space dimensions
        # Global: 6 dims (unchanged)
        # Layer: 6 dims × 5 layers = 30 dims (was 8 × 5 = 40)
        # Total: 36 dims (was 46)
        import gymnasium as gym
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(36,),
            dtype=np.float32
        )

    def _transform_observation(self, obs: Dict, prev_obs: Dict = None) -> np.ndarray:
        """
        Transform observation to flat array WITHOUT neighbor information

        Args:
            obs: Original observation dictionary
            prev_obs: Previous observation (for computing changes)

        Returns:
            Flat observation array (36 dims)
        """
        # Extract global state (6 dims)
        global_state = self._extract_global_state(obs)

        # Extract layer states (5 × 6 = 30 dims)
        layer_states = []
        for i in range(self.n_layers):
            layer_state = self._extract_layer_state(obs, i, prev_obs)
            layer_states.append(layer_state)

        # Flatten to 1D array
        flat_obs = np.concatenate([global_state] + layer_states)

        return flat_obs

    def _extract_layer_state(self, obs: Dict, layer_idx: int,
                            prev_obs: Dict = None) -> np.ndarray:
        """
        Extract layer state WITHOUT neighbor information

        Args:
            obs: Current observation dictionary
            layer_idx: Layer index (0-4)
            prev_obs: Previous observation (unused)

        Returns:
            6-dimensional layer state (no neighbor features)
        """
        i = layer_idx

        if isinstance(obs, dict):
            queue_length = obs.get('queue_lengths', np.zeros(self.n_layers))[i]
            utilization = obs.get('utilization_rates', np.zeros(self.n_layers))[i]
            service_rate = obs.get('service_rates', np.ones(self.n_layers))[i]
            load_rate = obs.get('load_rates', np.zeros(self.n_layers))[i]
            queue_change = obs.get('queue_changes', np.zeros(self.n_layers))[i]
        else:
            queue_length = 0.0
            utilization = 0.0
            service_rate = 1.0
            load_rate = 0.0
            queue_change = 0.0

        # Normalize
        normalized_queue = queue_length / self.capacities[i]

        # Return 6-dim state (no neighbor_up, neighbor_down)
        layer_state = np.array([
            normalized_queue,           # Normalized queue length
            self.capacities[i] / 8.0,   # Normalized capacity
            utilization,                # Utilization rate
            service_rate / 2.0,         # Normalized service rate
            load_rate / 5.0,            # Normalized load rate
            queue_change                # Queue change rate
        ], dtype=np.float32)

        return layer_state


if __name__ == '__main__':
    # Test the wrapper
    from env.drl_optimized_env_fixed import DRLOptimizedQueueEnvFixed

    print("Testing FlatObservationWrapper...")

    # Create base environment
    base_env = DRLOptimizedQueueEnvFixed(max_episode_steps=100)

    # Wrap with flat observation wrapper
    wrapped_env = FlatObservationWrapper(base_env)

    print(f"Observation space: {wrapped_env.observation_space}")
    print(f"Expected shape: (36,)")

    # Test reset
    obs, info = wrapped_env.reset()
    print(f"Reset observation shape: {obs.shape}")
    assert obs.shape == (36,), f"Expected (36,), got {obs.shape}"

    # Test step
    action = wrapped_env.action_space.sample()
    obs, reward, terminated, truncated, info = wrapped_env.step(action)
    print(f"Step observation shape: {obs.shape}")
    assert obs.shape == (36,), f"Expected (36,), got {obs.shape}"

    print("✓ FlatObservationWrapper test passed!")
