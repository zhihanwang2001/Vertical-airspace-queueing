"""
Hierarchical Capacity-Aware A2C (HCA2C) Algorithm

A novel DRL algorithm designed specifically for vertical layered queueing systems.

Key innovations:
1. Hierarchical Policy Decomposition - Global policy + 5 layer-specific policies
2. Capacity-Aware Action Clipping - Dynamic action space based on current state
3. Inter-Layer Coordination - Message passing between adjacent layers
"""

from .networks import (
    GlobalPolicyNetwork,
    LayerPolicyNetwork,
    CoordinationModule,
    HierarchicalActorCritic
)
from .wrapper import HierarchicalEnvWrapper
from .clipper import CapacityAwareClipper
from .hca2c_agent import HCA2C

__all__ = [
    'GlobalPolicyNetwork',
    'LayerPolicyNetwork',
    'CoordinationModule',
    'HierarchicalActorCritic',
    'HierarchicalEnvWrapper',
    'CapacityAwareClipper',
    'HCA2C'
]

__version__ = '1.0.0'
