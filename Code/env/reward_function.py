"""
Reward Function Module

Implements multi-objective optimization reward function from theoretical framework:
J(π) = w₁·Throughput - w₂·W̄ - w₃·Var(W) + w₄·Fairness

Includes:
- Basic reward components: throughput, waiting time, fairness, stability
- Penalty mechanisms: constraint violations, system instability, performance degradation
- Adaptive weights: dynamically adjust reward weights based on system state
- Layered rewards: layer-specific reward calculation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math

from .config import VerticalQueueConfig
from .utils import MathUtils


@dataclass
class RewardWeights:
    """Reward weight configuration"""
    throughput: float = 0.3        # Throughput weight
    waiting_time: float = 0.25     # Waiting time weight
    fairness: float = 0.2          # Fairness weight
    stability: float = 0.15        # Stability weight
    efficiency: float = 0.1        # Efficiency weight

    # Penalty weights
    instability_penalty: float = 1.0      # Instability penalty
    constraint_violation: float = 0.5     # Constraint violation penalty
    performance_degradation: float = 0.3  # Performance degradation penalty


class RewardFunction:
    """
    Multi-objective reward function

    Implements objective function from theoretical framework:
    max J(π) = w₁·Throughput - w₂·W̄ - w₃·Var(W) + w₄·Fairness

    Design principles:
    1. Multi-objective balance: throughput, waiting time, fairness, stability
    2. Stability priority: ensure stability condition ρᵢ < 1
    3. Adaptive weights: dynamically adjust weights based on system state
    4. Layer awareness: consider special structure of vertical layered queue
    """

    def __init__(self, config: VerticalQueueConfig, weights: Optional[RewardWeights] = None):
        self.config = config
        self.weights = weights or RewardWeights()
        self.math_utils = MathUtils()

        # Reward calculation history (for trend analysis)
        self.reward_history = []
        self.component_history = {
            'throughput': [],
            'waiting_time': [],
            'fairness': [],
            'stability': [],
            'efficiency': []
        }

        # Baseline values (for normalization)
        self.baseline_values = {
            'max_throughput': config.theoretical_max_throughput,
            'target_waiting_time': (config.theoretical_performance['avg_waiting_time_range'][0] +
                                  config.theoretical_performance['avg_waiting_time_range'][1]) / 2,
            'target_utilization': config.theoretical_performance['optimal_load_factor']
        }

        # Adaptive parameters
        self.adaptive_weights_enabled = True
        self.weight_adaptation_rate = 0.05
        
    def calculate_reward(self, queue_info: Dict, cabinet_info: Dict,
                        performance_stats: Dict) -> float:
        """
        Calculate comprehensive reward

        Args:
            queue_info: Queue information
            cabinet_info: Delivery cabinet information
            performance_stats: Performance statistics

        Returns:
            Scalar reward value
        """
        # 1. Calculate reward components
        components = self._calculate_reward_components(queue_info, cabinet_info, performance_stats)

        # 2. Check constraints and penalties
        penalties = self._calculate_penalties(queue_info, cabinet_info, performance_stats)

        # 3. Apply adaptive weights
        if self.adaptive_weights_enabled:
            adaptive_weights = self._calculate_adaptive_weights(components, performance_stats)
        else:
            adaptive_weights = self.weights

        # 4. Calculate weighted reward
        weighted_reward = (
            adaptive_weights.throughput * components['throughput'] +
            adaptive_weights.waiting_time * components['waiting_time'] +
            adaptive_weights.fairness * components['fairness'] +
            adaptive_weights.stability * components['stability'] +
            adaptive_weights.efficiency * components['efficiency']
        )

        # 5. Apply penalties
        total_penalty = (
            adaptive_weights.instability_penalty * penalties['instability'] +
            adaptive_weights.constraint_violation * penalties['constraint_violation'] +
            adaptive_weights.performance_degradation * penalties['performance_degradation']
        )

        # 6. Final reward
        final_reward = weighted_reward - total_penalty

        # 7. Record history
        self._update_reward_history(components, final_reward)

        return float(final_reward)

    def _calculate_reward_components(self, queue_info: Dict, cabinet_info: Dict,
                                   performance_stats: Dict) -> Dict[str, float]:
        """
        Calculate reward components
        """
        components = {}

        # 1. Throughput reward
        components['throughput'] = self._calculate_throughput_reward(queue_info, performance_stats)

        # 2. Waiting time reward (negative reward, longer waiting time means lower reward)
        components['waiting_time'] = self._calculate_waiting_time_reward(queue_info)

        # 3. Fairness reward
        components['fairness'] = self._calculate_fairness_reward(queue_info)

        # 4. Stability reward
        components['stability'] = self._calculate_stability_reward(queue_info)

        # 5. Efficiency reward
        components['efficiency'] = self._calculate_efficiency_reward(queue_info, cabinet_info)

        return components

    def _calculate_throughput_reward(self, queue_info: Dict, performance_stats: Dict) -> float:
        """
        Calculate throughput reward

        Based on theoretical framework: Throughput = E[Λ_system]
        """
        current_throughput = queue_info.get('throughput', 0)
        max_throughput = self.baseline_values['max_throughput']

        if max_throughput <= 0:
            return 0.0

        # Normalize throughput [0, 1]
        normalized_throughput = min(current_throughput / max_throughput, 1.0)

        # Nonlinear reward (encourage high throughput)
        throughput_reward = normalized_throughput ** 0.8

        return throughput_reward

    def _calculate_waiting_time_reward(self, queue_info: Dict) -> float:
        """
        Calculate waiting time reward (negative value, longer waiting time means lower reward)

        Based on theoretical framework: W̄ = E[(1/n)Σᵢ Wᵢ]
        """
        waiting_times = queue_info.get('waiting_times', [])

        if not waiting_times or all(w == 0 for w in waiting_times):
            return 0.0

        # Calculate average waiting time
        avg_waiting_time = np.mean(waiting_times)
        target_waiting_time = self.baseline_values['target_waiting_time']

        # Normalize waiting time (target time as baseline)
        if target_waiting_time > 0:
            normalized_wait = avg_waiting_time / target_waiting_time
        else:
            normalized_wait = avg_waiting_time / 20.0  # Default baseline

        # Waiting time reward (shorter waiting time means higher reward)
        wait_reward = -math.log(1 + normalized_wait)  # Logarithmic penalty

        return wait_reward

    def _calculate_fairness_reward(self, queue_info: Dict) -> float:
        """
        Calculate fairness reward

        Based on theoretical framework: Fairness = 1 - (max(Wᵢ) - min(Wᵢ))/max(Wᵢ)
        """
        waiting_times = queue_info.get('waiting_times', [])

        if len(waiting_times) < 2:
            return 1.0  # Single layer or no data is considered perfectly fair

        # Filter zero values
        non_zero_waits = [w for w in waiting_times if w > 0]

        if len(non_zero_waits) < 2:
            return 1.0

        max_wait = max(non_zero_waits)
        min_wait = min(non_zero_waits)

        if max_wait <= 0:
            return 1.0

        # Theoretical framework fairness formula
        fairness = 1.0 - (max_wait - min_wait) / max_wait

        # Additional fairness metric: Gini coefficient
        gini = self.math_utils.calculate_gini_coefficient(np.array(non_zero_waits))
        gini_fairness = 1.0 - gini

        # Combined fairness score
        combined_fairness = 0.7 * fairness + 0.3 * gini_fairness

        return max(combined_fairness, 0.0)

    def _calculate_stability_reward(self, queue_info: Dict) -> float:
        """
        Calculate stability reward

        Based on theoretical framework stability condition: ρᵢ = λᵢᵉᶠᶠ/(μᵢ·cᵢ) < 1
        """
        load_factors = queue_info.get('load_factors', {})

        if not load_factors:
            return 0.0

        # Extract load factor values
        rho_values = list(load_factors.values())

        # Stability score calculation
        stability_scores = []
        for rho in rho_values:
            if rho < 0.7:
                # Very stable
                score = 1.0
            elif rho < 0.9:
                # Relatively stable
                score = 1.0 - (rho - 0.7) / 0.2 * 0.3
            elif rho < 1.0:
                # Close to unstable
                score = 0.7 - (rho - 0.9) / 0.1 * 0.6
            else:
                # Unstable
                score = 0.1 - min(rho - 1.0, 1.0) * 0.1

            stability_scores.append(max(score, 0.0))

        # System stability = worst layer stability (bucket effect)
        system_stability = min(stability_scores) if stability_scores else 0.0

        return system_stability

    def _calculate_efficiency_reward(self, queue_info: Dict, cabinet_info: Dict) -> float:
        """
        Calculate efficiency reward

        Comprehensive coordination efficiency of airspace and ground
        """
        # Airspace efficiency: space utilization
        queue_lengths = queue_info.get('queue_lengths', [])
        capacities = self.config.layer_capacities

        if len(queue_lengths) == len(capacities):
            space_utilizations = [q/c if c > 0 else 0 for q, c in zip(queue_lengths, capacities)]
            avg_space_util = np.mean(space_utilizations)
        else:
            avg_space_util = 0.0
        
        # 地面效率：外卖柜利用率
        cabinet_occupancy = cabinet_info.get('occupancy_rate', 0)
        
        # 协调效率：空域和地面的平衡程度
        coordination_efficiency = 1.0 - abs(avg_space_util - cabinet_occupancy)
        
        # 综合效率
        efficiency = (avg_space_util * 0.4 + 
                     cabinet_occupancy * 0.3 + 
                     coordination_efficiency * 0.3)
        
        return min(efficiency, 1.0)
    
    def _calculate_penalties(self, queue_info: Dict, cabinet_info: Dict, 
                           performance_stats: Dict) -> Dict[str, float]:
        """
        计算各种惩罚
        """
        penalties = {
            'instability': 0.0,
            'constraint_violation': 0.0,
            'performance_degradation': 0.0
        }
        
        # 1. 不稳定惩罚
        load_factors = queue_info.get('load_factors', {})
        for rho in load_factors.values():
            if rho >= 1.0:
                penalties['instability'] += (rho - 1.0) * 2.0  # 严重惩罚不稳定
        
        # 2. 约束违反惩罚
        # 容量约束违反
        queue_lengths = queue_info.get('queue_lengths', [])
        capacities = self.config.layer_capacities
        
        for i, (length, capacity) in enumerate(zip(queue_lengths, capacities)):
            if length > capacity:
                violations = length - capacity
                penalties['constraint_violation'] += violations * 0.1
        
        # 外卖柜约束违反
        if cabinet_info.get('system_failed', False):
            penalties['constraint_violation'] += 1.0
        
        # 3. 性能恶化惩罚
        if len(self.component_history['throughput']) > 5:
            recent_throughput = np.mean(self.component_history['throughput'][-5:])
            earlier_throughput = np.mean(self.component_history['throughput'][-10:-5])
            
            if recent_throughput < earlier_throughput * 0.8:  # 性能下降20%以上
                penalties['performance_degradation'] += 0.5
        
        return penalties
    
    def _calculate_adaptive_weights(self, components: Dict[str, float], 
                                  performance_stats: Dict) -> RewardWeights:
        """
        计算自适应权重
        
        根据系统状态动态调整奖励权重
        """
        adaptive_weights = RewardWeights()
        
        # 基础权重
        base_weights = [
            self.weights.throughput,
            self.weights.waiting_time,
            self.weights.fairness,
            self.weights.stability,
            self.weights.efficiency
        ]
        
        # 根据系统性能调整权重
        # 1. 如果稳定性较差，增加稳定性权重
        if components['stability'] < 0.7:
            base_weights[3] *= 1.5  # 增加稳定性权重
            base_weights[0] *= 0.8  # 减少吞吐量权重
        
        # 2. 如果公平性较差，增加公平性权重
        if components['fairness'] < 0.6:
            base_weights[2] *= 1.3  # 增加公平性权重
            base_weights[1] *= 0.9  # 减少等待时间权重
        
        # 3. 如果效率较低，增加效率权重
        if components['efficiency'] < 0.5:
            base_weights[4] *= 1.4  # 增加效率权重
        
        # 归一化权重
        normalized_weights = self.math_utils.normalize_vector(base_weights, method='sum')
        
        # 更新自适应权重
        adaptive_weights.throughput = normalized_weights[0]
        adaptive_weights.waiting_time = normalized_weights[1]
        adaptive_weights.fairness = normalized_weights[2]
        adaptive_weights.stability = normalized_weights[3]
        adaptive_weights.efficiency = normalized_weights[4]
        
        # 惩罚权重保持不变
        adaptive_weights.instability_penalty = self.weights.instability_penalty
        adaptive_weights.constraint_violation = self.weights.constraint_violation
        adaptive_weights.performance_degradation = self.weights.performance_degradation
        
        return adaptive_weights
    
    def _update_reward_history(self, components: Dict[str, float], total_reward: float):
        """
        更新奖励历史记录
        """
        self.reward_history.append(total_reward)
        
        for component_name, value in components.items():
            if component_name in self.component_history:
                self.component_history[component_name].append(value)
        
        # 保持历史长度
        max_history = 1000
        if len(self.reward_history) > max_history:
            self.reward_history.pop(0)
            
        for component_list in self.component_history.values():
            if len(component_list) > max_history:
                component_list.pop(0)
    
    def get_reward_analysis(self) -> Dict:
        """
        获取奖励分析结果
        """
        if not self.reward_history:
            return {}
        
        analysis = {
            'total_reward': {
                'mean': np.mean(self.reward_history),
                'std': np.std(self.reward_history),
                'trend': self._calculate_trend(self.reward_history)
            },
            'components': {}
        }
        
        for component_name, history in self.component_history.items():
            if history:
                analysis['components'][component_name] = {
                    'mean': np.mean(history),
                    'std': np.std(history),
                    'trend': self._calculate_trend(history),
                    'current': history[-1] if history else 0
                }
        
        return analysis
    
    def _calculate_trend(self, data: List[float], window: int = 20) -> str:
        """
        计算数据趋势
        """
        if len(data) < window:
            return "insufficient_data"
        
        recent_data = data[-window:]
        slope, _, _ = self.math_utils.linear_regression(
            np.arange(len(recent_data)), 
            np.array(recent_data)
        )
        
        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"
    
    def set_weights(self, **kwargs):
        """
        设置奖励权重
        """
        for key, value in kwargs.items():
            if hasattr(self.weights, key):
                setattr(self.weights, key, value)
    
    def enable_adaptive_weights(self, enabled: bool = True):
        """
        启用/禁用自适应权重
        """
        self.adaptive_weights_enabled = enabled
    
    def reset_history(self):
        """
        重置历史记录
        """
        self.reward_history.clear()
        for component_list in self.component_history.values():
            component_list.clear()
    
    def get_current_weights(self) -> RewardWeights:
        """
        获取当前权重
        """
        return self.weights


# 预定义奖励配置
class RewardPresets:
    """预定义奖励配置"""
    
    @staticmethod
    def get_balanced_weights() -> RewardWeights:
        """平衡配置"""
        return RewardWeights(
            throughput=0.25,
            waiting_time=0.25,
            fairness=0.25,
            stability=0.15,
            efficiency=0.1
        )
    
    @staticmethod
    def get_throughput_focused_weights() -> RewardWeights:
        """吞吐量优先配置"""
        return RewardWeights(
            throughput=0.5,
            waiting_time=0.2,
            fairness=0.1,
            stability=0.15,
            efficiency=0.05
        )
    
    @staticmethod
    def get_fairness_focused_weights() -> RewardWeights:
        """公平性优先配置"""
        return RewardWeights(
            throughput=0.2,
            waiting_time=0.2,
            fairness=0.4,
            stability=0.15,
            efficiency=0.05
        )
    
    @staticmethod
    def get_stability_focused_weights() -> RewardWeights:
        """稳定性优先配置"""
        return RewardWeights(
            throughput=0.15,
            waiting_time=0.2,
            fairness=0.15,
            stability=0.4,
            efficiency=0.1
        )


# 测试奖励函数
if __name__ == "__main__":
    from .config import VerticalQueueConfig
    
    config = VerticalQueueConfig()
    reward_function = RewardFunction(config)
    
    print("奖励函数创建成功!")
    print(f"基准值: {reward_function.baseline_values}")
    
    # 测试奖励计算
    dummy_queue_info = {
        'throughput': 0.8,
        'waiting_times': [12, 15, 18, 20, 25],
        'queue_lengths': [3, 4, 2, 2, 1],
        'load_factors': {'layer_0': 0.6, 'layer_1': 0.7, 'layer_2': 0.5, 'layer_3': 0.8, 'layer_4': 0.4}
    }
    
    dummy_cabinet_info = {
        'occupancy_rate': 0.6,
        'system_failed': False
    }
    
    dummy_performance_stats = {}
    
    reward = reward_function.calculate_reward(dummy_queue_info, dummy_cabinet_info, dummy_performance_stats)
    print(f"计算奖励: {reward:.4f}")
    
    # 测试奖励分析
    for _ in range(10):
        reward_function.calculate_reward(dummy_queue_info, dummy_cabinet_info, dummy_performance_stats)
    
    analysis = reward_function.get_reward_analysis()
    print(f"奖励分析: {analysis}")
    
    print("✅ 奖励函数测试完成!")