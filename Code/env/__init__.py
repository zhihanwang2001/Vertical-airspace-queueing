"""
Vertical Stratified Queue Environment Package

RP1 Research Project - Phase 2 Environment Implementation
Vertical layered queue system based on theoretical framework

Module imports and package configuration
"""

from .vertical_queue_env import VerticalQueueEnv
from .state_manager import StateManager
from .queue_dynamics import QueueDynamics
from .delivery_cabinet import DeliveryCabinet, GridCell, TemperatureZone
from .config import VerticalQueueConfig
from .utils import MathUtils, PerformanceAnalyzer, VisualizationUtils
from .reward_function import RewardFunction, RewardWeights, RewardPresets
# Removed unused wrappers: drl_reward_wrapper, stochastic_wrapper, compact_state_wrapper

# Version information
__version__ = "1.0.0"
__author__ = "RP1 Research Team"
__description__ = "Vertical Stratified Queue Environment for UAV Food Delivery Systems"

# Package level configuration
__all__ = [
    # Core environment
    "VerticalQueueEnv",

    # System components
    "StateManager",
    "QueueDynamics",
    "DeliveryCabinet",
    "GridCell",
    "TemperatureZone",

    # Configuration and utilities
    "VerticalQueueConfig",
    "MathUtils",
    "PerformanceAnalyzer",
    "VisualizationUtils",

    # Reward system
    "RewardFunction",
    "RewardWeights",
    "RewardPresets",

    # DRL adaptation (removed unused wrappers)
]

# Quick creation function
def create_environment(config_name: str = "default", **kwargs):
    """
    Quickly create vertical queue environment

    Args:
        config_name: Configuration name ("default", "high_traffic", "fair_priority", "stable_focus")
        **kwargs: Additional configuration parameters

    Returns:
        VerticalQueueEnv: Configured environment instance
    """
    config = VerticalQueueConfig()

    if config_name == "high_traffic":
        config.base_arrival_rate = 0.4
        config.layer_capacities = [10, 8, 6, 4, 3]
    elif config_name == "fair_priority":
        reward_weights = RewardPresets.get_fairness_focused_weights()
        kwargs.setdefault('reward_weights', reward_weights)
    elif config_name == "stable_focus":
        reward_weights = RewardPresets.get_stability_focused_weights()
        kwargs.setdefault('reward_weights', reward_weights)

    # Apply additional configuration
    for key, value in kwargs.items():
        if hasattr(config, key) and key != 'reward_weights':
            setattr(config, key, value)

    return VerticalQueueEnv(config=config, **kwargs)

def get_default_config():
    """
    Get default configuration

    Returns:
        VerticalQueueConfig: Default configuration instance
    """
    return VerticalQueueConfig()

def validate_environment(env):
    """
    Validate environment configuration

    Args:
        env: VerticalQueueEnv instance

    Returns:
        Dict: Validation results
    """
    validation_results = {
        'valid': True,
        'warnings': [],
        'errors': []
    }

    # Check configuration validity
    config = env.config

    # 1. Layer configuration check
    if len(config.layer_capacities) != len(config.layer_heights):
        validation_results['errors'].append("Layer capacity and height count mismatch")
        validation_results['valid'] = False

    # 2. Stability check
    for i, capacity in enumerate(config.layer_capacities):
        if capacity <= 0:
            validation_results['errors'].append(f"Layer {i} capacity must be greater than 0")
            validation_results['valid'] = False

    # 3. Load factor check
    if config.base_arrival_rate <= 0:
        validation_results['errors'].append("Base arrival rate must be greater than 0")
        validation_results['valid'] = False

    # 4. Theoretical consistency check
    total_capacity = sum(config.layer_capacities)
    theoretical_max = config.theoretical_max_throughput

    if theoretical_max > total_capacity:
        validation_results['warnings'].append(
            f"Theoretical max throughput ({theoretical_max}) may exceed system capacity ({total_capacity})"
        )

    return validation_results

# Module initialization check
def _check_dependencies():
    """Check dependencies"""
    try:
        import numpy as np
        import gymnasium as gym
        return True
    except ImportError as e:
        print(f"Warning: Missing required dependency {e}")
        return False

# Execute initialization check
_check_dependencies()

# Package information (avoid import-time prints in library use)
# Use logging in executables/tests if runtime messages are needed.
