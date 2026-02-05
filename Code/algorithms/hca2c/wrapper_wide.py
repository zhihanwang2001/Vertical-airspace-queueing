"""
Wide Action Space Wrapper for Ablation Study

This wrapper uses the same action space ranges as A2C/PPO baselines
to test the contribution of capacity-aware action clipping.
"""

import numpy as np
from typing import Dict
from .wrapper import HierarchicalEnvWrapper


class WideActionWrapper(HierarchicalEnvWrapper):
    """
    Wide action space wrapper - moderately wider than HCA2C-Full

    Changes from HierarchicalEnvWrapper:
    - Service intensities: [0.5, 1.5] -> [0.4, 1.6]
    - Arrival multiplier: [1.0, 3.0] -> [0.8, 3.5]

    This tests the contribution of capacity-aware (conservative) action clipping
    to HCA2C performance.
    """

    def _transform_action(self, action: np.ndarray) -> Dict:
        """
        Transform action with moderately WIDE ranges

        Args:
            action: Flat action vector [11] with values in [-1, 1]
                - [0:5]: service intensities
                - [5]: arrival multiplier
                - [6:11]: emergency transfers

        Returns:
            Environment action dictionary with WIDE action ranges
        """
        action = np.clip(action, -1.0, 1.0)

        # Service intensities: [-1, 1] -> [0.4, 1.6] (moderately wider)
        # Wider than HCA2C-Full [0.5, 1.5] by 20%
        service_intensities = (action[:5] + 1) / 2 * 1.2 + 0.4

        # Arrival multiplier: [-1, 1] -> [0.8, 3.5] (moderately wider)
        # Wider than HCA2C-Full [1.0, 3.0] by ~17%
        arrival_multiplier = (action[5] + 1) / 2 * 2.7 + 0.8

        # Emergency transfers: [-1, 1] -> {0, 1} (unchanged)
        emergency_transfers = (action[6:11] > 0).astype(np.int8)

        return {
            'service_intensities': service_intensities.astype(np.float32),
            'arrival_multiplier': np.array([arrival_multiplier], dtype=np.float32),
            'emergency_transfers': emergency_transfers
        }


if __name__ == '__main__':
    # Test the wrapper
    from env.drl_optimized_env_fixed import DRLOptimizedQueueEnvFixed

    print("Testing WideActionWrapper...")

    # Create base environment
    base_env = DRLOptimizedQueueEnvFixed(max_episode_steps=100)

    # Wrap with wide action wrapper
    wrapped_env = WideActionWrapper(base_env)

    print(f"Action space: {wrapped_env.action_space}")

    # Test reset
    obs, info = wrapped_env.reset()
    print(f"Reset observation shape: {obs.shape}")

    # Test action transformation
    test_action = np.array([-1.0] * 11)  # Minimum action
    transformed = wrapped_env._transform_action(test_action)
    print(f"\nMinimum action transformation:")
    print(f"  Service intensities: {transformed['service_intensities']}")
    print(f"  Expected: [0.4, 0.4, 0.4, 0.4, 0.4]")
    print(f"  Arrival multiplier: {transformed['arrival_multiplier']}")
    print(f"  Expected: [0.8]")

    test_action = np.array([1.0] * 11)  # Maximum action
    transformed = wrapped_env._transform_action(test_action)
    print(f"\nMaximum action transformation:")
    print(f"  Service intensities: {transformed['service_intensities']}")
    print(f"  Expected: [1.6, 1.6, 1.6, 1.6, 1.6]")
    print(f"  Arrival multiplier: {transformed['arrival_multiplier']}")
    print(f"  Expected: [3.5]")

    # Test step
    action = wrapped_env.action_space.sample()
    obs, reward, terminated, truncated, info = wrapped_env.step(action)
    print(f"\nStep observation shape: {obs.shape}")

    print("✓ WideActionWrapper test passed!")
