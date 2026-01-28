"""
Advanced DRL Algorithms Package
Advanced deep reinforcement learning algorithm collection for CCF-B journal paper

Contains state-of-the-art DRL algorithm implementations:
1. Rainbow DQN - Integrates 6 DQN improvements
2. IMPALA - Distributed Importance Weighted Actor-Learner Architecture
3. R2D2 - Recurrent Replay Distributed DQN
4. SAC v2 - Improved Soft Actor-Critic algorithm
5. TD7 - Further improvements on TD3 algorithm

All algorithms feature:
- Integrated TensorBoard monitoring
- Compatible with existing baseline framework
- Support for vertical layered queue environment
- Complete training and evaluation functionality
"""

# Rainbow DQN
from .rainbow_dqn import RainbowDQNAgent, RainbowDQNBaseline

# IMPALA
from .impala import IMPALAAgent, IMPALABaseline
from .impala.impala_optimized import OptimizedIMPALABaseline

# R2D2
from .r2d2 import R2D2Agent, R2D2Baseline

# SAC v2
from .sac_v2 import SAC_v2_Agent, SAC_v2_Baseline

# TD7
from .td7 import TD7_Agent, TD7Baseline

__version__ = "1.0.0"
__all__ = [
    # Rainbow DQN
    "RainbowDQNAgent",
    "RainbowDQNBaseline",

    # IMPALA
    "IMPALAAgent",
    "IMPALABaseline",
    "OptimizedIMPALABaseline",

    # R2D2
    "R2D2Agent", "R2D2Baseline",

    # SAC v2
    "SAC_v2_Agent", "SAC_v2_Baseline",

    # TD7
    "TD7_Agent", "TD7Baseline",
]

# Algorithm registry
AVAILABLE_ALGORITHMS = {
    "rainbow_dqn": {
        "name": "Rainbow DQN",
        "type": "value_based",
        "baseline_class": "RainbowDQNBaseline", 
        "description": "Deep Q-Network with 6 improvements: Double DQN, Prioritized Replay, Dueling Networks, Multi-step Learning, Distributional RL, Noisy Networks",
        "paper": "Rainbow: Combining Improvements in Deep Reinforcement Learning (Hessel et al., 2018)",
        "status": "implemented"
    },
    "impala": {
        "name": "IMPALA",
        "type": "actor_critic",
        "baseline_class": "IMPALABaseline",
        "description": "Importance Weighted Actor-Learner Architecture for distributed RL",
        "paper": "IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures (Espeholt et al., 2018)",
        "status": "implemented"
    },
    "impala_optimized": {
        "name": "IMPALA Optimized",
        "type": "actor_critic",
        "baseline_class": "OptimizedIMPALABaseline",
        "description": "Queue-specific optimized IMPALA with mixed action space support and conservative V-trace parameters",
        "paper": "IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures (Espeholt et al., 2018) + Queue-specific optimizations",
        "status": "implemented"
    },
    "r2d2": {
        "name": "R2D2",
        "type": "value_based",
        "baseline_class": "R2D2Baseline", 
        "description": "Recurrent Replay Distributed DQN for partially observable environments",
        "paper": "Recurrent Experience Replay in Distributed Reinforcement Learning (Kapturowski et al., 2019)",
        "status": "implemented"
    },
    "sac_v2": {
        "name": "SAC v2", 
        "type": "actor_critic",
        "baseline_class": "SAC_v2_Baseline",
        "description": "Soft Actor-Critic with automatic entropy tuning and other improvements",
        "paper": "Soft Actor-Critic Algorithms and Applications (Haarnoja et al., 2019)",
        "status": "implemented"
    },
    "td7": {
        "name": "TD7",
        "type": "actor_critic",
        "baseline_class": "TD7Baseline", 
        "description": "TD3 with additional improvements for better stability and performance",
        "paper": "TD7: A Regularized Actor-Critic Method for Robotic Reinforcement Learning (Fujimoto & Gu, 2021)",
        "status": "implemented"
    }
}


def get_available_algorithms():
    """Get list of available algorithms"""
    return AVAILABLE_ALGORITHMS


def get_algorithm_info(algorithm_name: str):
    """Get detailed algorithm information"""
    return AVAILABLE_ALGORITHMS.get(algorithm_name, None)


def create_algorithm_baseline(algorithm_name: str, config=None):
    """Factory function: Create algorithm baseline"""
    if algorithm_name == "rainbow_dqn":
        return RainbowDQNBaseline(config)
    elif algorithm_name == "impala":
        return IMPALABaseline(config)
    elif algorithm_name == "impala_optimized":
        return OptimizedIMPALABaseline(config)
    elif algorithm_name == "r2d2":
        return R2D2Baseline(config)
    elif algorithm_name == "sac_v2":
        return SAC_v2_Baseline(config)
    elif algorithm_name == "td7":
        return TD7Baseline(config)
    else:
        available = ", ".join(AVAILABLE_ALGORITHMS.keys())
        raise ValueError(f"Unknown algorithm: {algorithm_name}. Available: {available}")


def print_algorithms_status():
    """Print implementation status of all algorithms"""
    print("🤖 Advanced DRL Algorithms Status:")
    print("=" * 60)
    
    for name, info in AVAILABLE_ALGORITHMS.items():
        status_emoji = {
            "implemented": "✅",
            "in_progress": "🚧", 
            "planned": "📋"
        }
        
        print(f"{status_emoji.get(info['status'], '❓')} {info['name']}")
        print(f"   Type: {info['type']}")
        print(f"   Status: {info['status']}")
        print(f"   Paper: {info['paper']}")
        print()


if __name__ == "__main__":
    print_algorithms_status()