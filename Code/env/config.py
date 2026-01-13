"""
配置文件
Configuration File

基于01理论文档的完整参数配置：
- 垂直分层队列标准参数 
- 倒金字塔容量结构
- 性能预测基准
- 环境实验设置
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np


@dataclass
class VerticalQueueConfig:
    """
    垂直分层队列配置类
    
    基于01理论文档的标准参数配置：
    - 高度层配置：L = {100m, 80m, 60m, 40m, 20m}
    - 容量配置：C = {8, 6, 4, 3, 2} (倒金字塔结构)
    - 到达参数：λ₀ = 0.25/step, α = [0.1, 0.15, 0.25, 0.3, 0.2]
    - 服务参数：μ = [0.8, 0.9, 1.0, 1.2, 1.5]
    """
    
    # ============= 基础系统参数 =============
    num_layers: int = 5
    layer_heights: List[float] = None  # [100, 80, 60, 40, 20]米 (L5→L1, 高→低)
    layer_capacities: List[int] = None  # [8, 6, 4, 3, 2] 倒金字塔 (L5→L1)

    # ============= 到达过程参数 =============
    base_arrival_rate: float = 0.25  # λ₀ = 0.25/step
    arrival_weights: List[float] = None  # α = [0.3, 0.25, 0.2, 0.15, 0.1] (L5→L1)

    # ============= 服务过程参数 =============
    layer_service_rates: List[float] = None  # μ = [1.2, 1.0, 0.8, 0.6, 0.4] (L5→L1)
    min_wait_times: List[int] = None  # τ_min = [10, 8, 6, 4, 2] steps
    
    # ============= 外卖柜参数 =============
    total_cabinet_cells: int = 24
    cells_per_zone: int = 8
    temperature_zones: List[str] = None  # ['cold', 'hot', 'normal']
    
    # ============= 环境参数 =============
    max_episode_steps: int = 1000
    max_waiting_time: int = 50  # 最大等待时间(步数)
    history_length: int = 20    # 历史记录长度
    random_seed: int = 42
    
    # ============= 性能预测基准 (来自01理论) =============
    theoretical_performance: Dict = None
    
    # ============= 实验设置参数 =============
    experiment_configs: Dict = None
    
    def __post_init__(self):
        """初始化默认值"""
        
        # 高度层配置 (基于01理论标准配置)
        if self.layer_heights is None:
            self.layer_heights = [100.0, 80.0, 60.0, 40.0, 20.0]
        
        # 倒金字塔容量结构 (01理论核心创新)
        if self.layer_capacities is None:
            self.layer_capacities = [8, 6, 4, 3, 2]
        
        # 到达权重 (高层承担主要流量)
        # 数组顺序: 索引0→4 对应 L5→L1 (100m→20m)
        if self.arrival_weights is None:
            self.arrival_weights = [0.3, 0.25, 0.2, 0.15, 0.1]  # L5→L1

        # 服务率 (高层服务更快: L5最快1.2, L1最慢0.4)
        # 数组顺序: 索引0→4 对应 L5→L1 (100m→20m)
        if self.layer_service_rates is None:
            self.layer_service_rates = [1.2, 1.0, 0.8, 0.6, 0.4]  # L5→L1
        
        # 最小等待时间 (转移触发条件)
        if self.min_wait_times is None:
            self.min_wait_times = [10, 8, 6, 4, 2]
        
        # 温区配置
        if self.temperature_zones is None:
            self.temperature_zones = ['cold', 'hot', 'normal']
        
        # 理论性能预测 (来自01理论文档)
        if self.theoretical_performance is None:
            self.theoretical_performance = {
                'avg_waiting_time_range': (15, 25),      # 平均等待时间：15-25步
                'system_throughput_range': (0.8, 1.2),  # 系统吞吐量：0.8-1.2订单/步
                'space_utilization_range': (0.75, 0.85), # 空间利用率：75-85%
                'efficiency_improvement': (0.4, 0.6),    # 垂直效率提升：40-60%
                'stability_threshold': 1.0,              # 稳定性阈值：ρᵢ < 1
                'optimal_load_factor': 0.8               # 最优负载系数
            }
        
        # 实验配置
        if self.experiment_configs is None:
            self.experiment_configs = {
                # H1验证：倒金字塔容量分配最优性
                'H1_capacity_configurations': {
                    'pyramid': [8, 6, 4, 3, 2],         # 倒金字塔 (我们的理论)
                    'uniform': [5, 5, 5, 5, 5],         # 均匀分配
                    'reverse_pyramid': [2, 3, 4, 6, 8], # 正金字塔
                    'random': [4, 7, 3, 5, 6]           # 随机分配
                },
                
                # H2验证：分层vs单层队列性能对比
                'H2_baseline_comparisons': {
                    'single_layer_capacity': 23,         # 单层总容量
                    'fifo_queue': True,                  # FIFO排队
                    'priority_queue': True,              # 优先级排队
                    'mm1_model': True                    # M/M/1理论模型
                },
                
                # H3验证：系统稳定性测试
                'H3_stability_tests': {
                    'load_test_factors': [0.3, 0.5, 0.7, 0.8, 0.9, 0.95], # 负载系数测试
                    'arrival_spike_tests': [2.0, 3.0, 5.0],                # 到达率冲击测试
                    'service_degradation': [0.8, 0.6, 0.4],                # 服务率降低测试
                },
                
                # H4验证：多目标帕累托最优
                'H4_pareto_analysis': {
                    'objectives': ['throughput', 'fairness', 'efficiency', 'stability'],
                    'weight_combinations': [
                        [0.4, 0.2, 0.2, 0.2],  # 吞吐量优先
                        [0.2, 0.4, 0.2, 0.2],  # 公平性优先  
                        [0.2, 0.2, 0.4, 0.2],  # 效率优先
                        [0.2, 0.2, 0.2, 0.4],  # 稳定性优先
                        [0.25, 0.25, 0.25, 0.25] # 均衡配置
                    ]
                }
            }
        
        # 验证配置有效性
        self._validate_config()
    
    def _validate_config(self):
        """验证配置参数的有效性"""
        
        # 检查数组长度一致性
        assert len(self.layer_heights) == self.num_layers, "高度层数量不匹配"
        assert len(self.layer_capacities) == self.num_layers, "容量配置数量不匹配"
        assert len(self.arrival_weights) == self.num_layers, "到达权重数量不匹配"
        assert len(self.layer_service_rates) == self.num_layers, "服务率数量不匹配"
        assert len(self.min_wait_times) == self.num_layers, "最小等待时间数量不匹配"
        
        # 检查倒金字塔特性 (01理论核心)
        for i in range(len(self.layer_capacities) - 1):
            assert self.layer_capacities[i] >= self.layer_capacities[i+1], \
                f"违反倒金字塔结构: C[{i}]={self.layer_capacities[i]} < C[{i+1}]={self.layer_capacities[i+1]}"
        
        # 检查高层服务更快 (01理论要求：L5最快, L1最慢)
        # 由于数组索引0→4对应L5→L1, 因此服务率应递减
        for i in range(len(self.layer_service_rates) - 1):
            assert self.layer_service_rates[i] >= self.layer_service_rates[i+1], \
                f"违反高层服务快原则: μ[{i}]={self.layer_service_rates[i]} < μ[{i+1}]={self.layer_service_rates[i+1]} (索引{i}是L{5-i},应该≥索引{i+1}的L{5-i-1})"
        
        # 检查高度递减
        for i in range(len(self.layer_heights) - 1):
            assert self.layer_heights[i] > self.layer_heights[i+1], \
                f"高度层应递减: H[{i}]={self.layer_heights[i]} <= H[{i+1}]={self.layer_heights[i+1]}"
        
        # 检查概率和为1
        weight_sum = sum(self.arrival_weights)
        assert abs(weight_sum - 1.0) < 0.01, f"到达权重和应为1.0，当前为{weight_sum}"
        
        print("✅ 配置验证通过：符合01理论要求")
    
    @property
    def theoretical_max_throughput(self) -> float:
        """
        理论最大吞吐量
        
        基于01理论：Λ_max = Σ μᵢ·cᵢ
        """
        return sum(rate * cap for rate, cap in zip(self.layer_service_rates, self.layer_capacities))
    
    @property 
    def total_system_capacity(self) -> int:
        """
        系统总容量
        """
        return sum(self.layer_capacities)
    
    def get_stability_condition_params(self) -> Dict:
        """
        获取稳定性条件参数
        
        用于计算 ρᵢ = λᵢᵉᶠᶠ/(μᵢ·cᵢ) < 1
        """
        return {
            'service_capacities': [rate * cap for rate, cap in 
                                 zip(self.layer_service_rates, self.layer_capacities)],
            'base_arrival_rate': self.base_arrival_rate,
            'arrival_weights': self.arrival_weights,
            'stability_threshold': self.theoretical_performance['stability_threshold']
        }
    
    def get_experiment_config(self, hypothesis: str) -> Dict:
        """
        获取特定假设的实验配置
        
        Args:
            hypothesis: 'H1', 'H2', 'H3', 'H4'
        """
        hypothesis_key = f'{hypothesis}_'
        configs = {}
        
        for key, value in self.experiment_configs.items():
            if key.startswith(hypothesis_key):
                configs[key] = value
        
        if not configs:
            raise ValueError(f"未找到假设 {hypothesis} 的配置")
        
        return configs
    
    def create_baseline_config(self, baseline_type: str) -> 'VerticalQueueConfig':
        """
        创建基线对比配置
        
        Args:
            baseline_type: 'uniform', 'reverse_pyramid', 'single_layer'
        """
        config = VerticalQueueConfig()
        
        if baseline_type == 'uniform':
            # 均匀容量分配
            avg_capacity = sum(self.layer_capacities) // self.num_layers
            config.layer_capacities = [avg_capacity] * self.num_layers
            
        elif baseline_type == 'reverse_pyramid':
            # 正金字塔结构 (与我们的理论相反)
            config.layer_capacities = list(reversed(self.layer_capacities))
            
        elif baseline_type == 'single_layer':
            # 单层队列
            config.num_layers = 1
            config.layer_heights = [self.layer_heights[0]]
            config.layer_capacities = [sum(self.layer_capacities)]
            config.arrival_weights = [1.0]
            config.layer_service_rates = [self.layer_service_rates[0]]
            config.min_wait_times = [self.min_wait_times[0]]
            
        return config
    
    def get_performance_bounds(self) -> Dict:
        """
        获取理论性能边界
        
        用于评估实际性能与理论预测的差异
        """
        return {
            'waiting_time': {
                'min': self.theoretical_performance['avg_waiting_time_range'][0],
                'max': self.theoretical_performance['avg_waiting_time_range'][1],
                'target': (self.theoretical_performance['avg_waiting_time_range'][0] + 
                          self.theoretical_performance['avg_waiting_time_range'][1]) / 2
            },
            'throughput': {
                'min': self.theoretical_performance['system_throughput_range'][0], 
                'max': self.theoretical_performance['system_throughput_range'][1],
                'theoretical_max': self.theoretical_max_throughput
            },
            'utilization': {
                'min': self.theoretical_performance['space_utilization_range'][0],
                'max': self.theoretical_performance['space_utilization_range'][1],
                'optimal': self.theoretical_performance['optimal_load_factor']
            },
            'efficiency_improvement': {
                'min': self.theoretical_performance['efficiency_improvement'][0],
                'max': self.theoretical_performance['efficiency_improvement'][1]
            }
        }
    
    def export_config(self) -> Dict:
        """
        导出配置为字典格式
        """
        return {
            'system_params': {
                'num_layers': self.num_layers,
                'layer_heights': self.layer_heights,
                'layer_capacities': self.layer_capacities,
                'base_arrival_rate': self.base_arrival_rate,
                'arrival_weights': self.arrival_weights,
                'layer_service_rates': self.layer_service_rates,
                'min_wait_times': self.min_wait_times
            },
            'cabinet_params': {
                'total_cells': self.total_cabinet_cells,
                'cells_per_zone': self.cells_per_zone,
                'temperature_zones': self.temperature_zones
            },
            'environment_params': {
                'max_episode_steps': self.max_episode_steps,
                'max_waiting_time': self.max_waiting_time,
                'history_length': self.history_length,
                'random_seed': self.random_seed
            },
            'theoretical_performance': self.theoretical_performance,
            'experiment_configs': self.experiment_configs
        }
    
    def __str__(self) -> str:
        """
        配置信息的字符串表示
        """
        return f"""
垂直分层队列配置 (基于01理论):
================================
系统结构:
  - 层数: {self.num_layers}
  - 高度: {self.layer_heights} (m)
  - 容量: {self.layer_capacities} (倒金字塔)
  - 总容量: {self.total_system_capacity}

动力学参数:
  - 基础到达率: {self.base_arrival_rate}/step
  - 到达权重: {self.arrival_weights}
  - 服务率: {self.layer_service_rates} (上升优先性：高层更快)
  - 最小等待: {self.min_wait_times} steps

理论预测:
  - 等待时间: {self.theoretical_performance['avg_waiting_time_range']} steps
  - 吞吐量: {self.theoretical_performance['system_throughput_range']} 订单/step
  - 利用率: {self.theoretical_performance['space_utilization_range']}
  - 效率提升: {self.theoretical_performance['efficiency_improvement']}

外卖柜配置:
  - 总格子: {self.total_cabinet_cells}
  - 每温区: {self.cells_per_zone}
  - 温区类型: {self.temperature_zones}
"""


# 预定义配置
class ConfigPresets:
    """预定义配置模板"""
    
    @staticmethod
    def get_standard_config() -> VerticalQueueConfig:
        """获取标准配置 (01理论默认参数)"""
        return VerticalQueueConfig()
    
    @staticmethod  
    def get_high_load_config() -> VerticalQueueConfig:
        """高负载测试配置"""
        config = VerticalQueueConfig()
        config.base_arrival_rate = 0.4  # 增加到达率
        config.max_episode_steps = 2000  # 延长测试时间
        return config
    
    @staticmethod
    def get_stability_test_config() -> VerticalQueueConfig:
        """稳定性测试配置"""
        config = VerticalQueueConfig()
        config.base_arrival_rate = 0.35  # 接近稳定性边界
        config.max_waiting_time = 100   # 增加等待容忍度
        return config
    
    @staticmethod
    def get_efficiency_test_config() -> VerticalQueueConfig:
        """效率测试配置"""
        config = VerticalQueueConfig()
        # 优化服务率配置
        config.layer_service_rates = [0.9, 1.0, 1.1, 1.3, 1.6]
        config.min_wait_times = [8, 6, 4, 3, 1]  # 更快的转移
        return config
    
    @staticmethod
    def get_debug_config() -> VerticalQueueConfig:
        """调试配置 (快速测试)"""
        config = VerticalQueueConfig()
        config.max_episode_steps = 100
        config.base_arrival_rate = 0.1
        config.history_length = 5
        return config


# 测试配置
if __name__ == "__main__":
    print("测试配置系统...")
    
    # 创建标准配置
    config = VerticalQueueConfig()
    print(config)
    
    # 测试稳定性参数
    stability_params = config.get_stability_condition_params()
    print(f"服务容量: {stability_params['service_capacities']}")
    print(f"理论最大吞吐量: {config.theoretical_max_throughput:.2f}")
    
    # 测试实验配置
    h1_config = config.get_experiment_config('H1')
    print(f"H1实验配置: {list(h1_config.keys())}")
    
    # 测试基线配置
    uniform_config = config.create_baseline_config('uniform')
    print(f"均匀配置容量: {uniform_config.layer_capacities}")
    
    # 测试性能边界
    bounds = config.get_performance_bounds()
    print(f"等待时间目标: {bounds['waiting_time']['target']}")
    
    print("\n✅ 配置系统测试完成!")