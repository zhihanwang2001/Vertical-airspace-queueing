"""
Baseline Algorithms for Vertical Stratified Queuing System
Baseline algorithms for vertical stratified queue system

Contains SB3 algorithm implementations and baseline comparisons:
- SB3_TD3 (Stable-Baselines3 TD3)
- SB3_SAC (Stable-Baselines3 SAC)
- SB3_PPO (Stable-Baselines3 PPO with Cosine LR)
- SB3_A2C (Stable-Baselines3 A2C)
- SB3_DDPG (Stable-Baselines3 DDPG)
- SB3_Ablation (Ablation study baseline)
- Random Agent (Random baseline)
- Heuristic Agent (Heuristic baseline)
"""

from .base_baseline import BaseBaseline
from .random_baseline import RandomBaseline
from .heuristic_baseline import HeuristicBaseline

# Optional SB3-based baselines (import lazily to avoid hard dependency)
try:
    from .sb3_td3_baseline import SB3TD3Baseline
    from .sb3_sac_baseline import SB3SACBaseline
    from .sb3_ppo_baseline import SB3PPOBaseline
    from .sb3_a2c_baseline import SB3A2CBaseline
    from .sb3_ddpg_baseline import SB3DDPGBaseline
    from .sb3_ablation_baseline import SB3AblationBaseline
except Exception:
    # SB3 not installed; these will be unavailable
    SB3TD3Baseline = None
    SB3SACBaseline = None
    SB3PPOBaseline = None
    SB3A2CBaseline = None
    SB3DDPGBaseline = None
    SB3AblationBaseline = None


__all__ = [
    'BaseBaseline',
    'SB3TD3Baseline',
    'SB3SACBaseline',
    'SB3PPOBaseline',
    'SB3A2CBaseline',
    'SB3DDPGBaseline',
    'SB3AblationBaseline',
    'RandomBaseline',
    'HeuristicBaseline'
]
