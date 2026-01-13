"""
垂直分层队列环境包
Vertical Stratified Queue Environment Package

RP1研究项目 - Phase 2 环境实现
基于01理论的垂直分层队列系统

模块导入和包配置
"""

from .vertical_queue_env import VerticalQueueEnv
from .state_manager import StateManager
from .queue_dynamics import QueueDynamics
from .delivery_cabinet import DeliveryCabinet, GridCell, TemperatureZone
from .config import VerticalQueueConfig
from .utils import MathUtils, PerformanceAnalyzer, VisualizationUtils
from .reward_function import RewardFunction, RewardWeights, RewardPresets
# Removed unused wrappers: drl_reward_wrapper, stochastic_wrapper, compact_state_wrapper

# 版本信息
__version__ = "1.0.0"
__author__ = "RP1 Research Team"
__description__ = "Vertical Stratified Queue Environment for UAV Food Delivery Systems"

# 包级别配置
__all__ = [
    # 核心环境
    "VerticalQueueEnv",
    
    # 系统组件
    "StateManager", 
    "QueueDynamics",
    "DeliveryCabinet",
    "GridCell",
    "TemperatureZone",
    
    # 配置和工具
    "VerticalQueueConfig",
    "MathUtils",
    "PerformanceAnalyzer", 
    "VisualizationUtils",
    
    # 奖励系统
    "RewardFunction",
    "RewardWeights", 
    "RewardPresets",
    
    # DRL适配 (removed unused wrappers)
]

# 快速创建函数
def create_environment(config_name: str = "default", **kwargs):
    """
    快速创建垂直队列环境
    
    Args:
        config_name: 配置名称 ("default", "high_traffic", "fair_priority", "stable_focus")
        **kwargs: 额外配置参数
        
    Returns:
        VerticalQueueEnv: 配置好的环境实例
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
    
    # 应用额外配置
    for key, value in kwargs.items():
        if hasattr(config, key) and key != 'reward_weights':
            setattr(config, key, value)
    
    return VerticalQueueEnv(config=config, **kwargs)

def get_default_config():
    """
    获取默认配置
    
    Returns:
        VerticalQueueConfig: 默认配置实例
    """
    return VerticalQueueConfig()

def validate_environment(env):
    """
    验证环境配置
    
    Args:
        env: VerticalQueueEnv实例
        
    Returns:
        Dict: 验证结果
    """
    validation_results = {
        'valid': True,
        'warnings': [],
        'errors': []
    }
    
    # 检查配置有效性
    config = env.config
    
    # 1. 层级配置检查
    if len(config.layer_capacities) != len(config.layer_heights):
        validation_results['errors'].append("层级容量和高度数量不匹配")
        validation_results['valid'] = False
    
    # 2. 稳定性检查
    for i, capacity in enumerate(config.layer_capacities):
        if capacity <= 0:
            validation_results['errors'].append(f"层级{i}容量必须大于0")
            validation_results['valid'] = False
    
    # 3. 负载因子检查
    if config.base_arrival_rate <= 0:
        validation_results['errors'].append("基础到达率必须大于0")
        validation_results['valid'] = False
    
    # 4. 理论一致性检查
    total_capacity = sum(config.layer_capacities)
    theoretical_max = config.theoretical_max_throughput
    
    if theoretical_max > total_capacity:
        validation_results['warnings'].append(
            f"理论最大吞吐量({theoretical_max})可能超过系统容量({total_capacity})"
        )
    
    return validation_results

# 模块初始化检查
def _check_dependencies():
    """检查依赖项"""
    try:
        import numpy as np
        import gymnasium as gym
        return True
    except ImportError as e:
        print(f"警告: 缺少必要依赖 {e}")
        return False

# 执行初始化检查
_check_dependencies()

# 包信息
print(f"垂直分层队列环境包 v{__version__} 加载成功")
print(f"描述: {__description__}")
print("可用环境配置:")
print("  - default: 标准配置")  
print("  - high_traffic: 高流量配置")
print("  - fair_priority: 公平性优先配置")
print("  - stable_focus: 稳定性优先配置")
print("使用 create_environment() 快速创建环境")