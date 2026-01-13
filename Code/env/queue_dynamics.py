"""
队列动力学模块
Queue Dynamics Module

实现01理论中的垂直分层队列动力学：
- 非齐次泊松到达过程 Ni(t) ~ NHPP(λi(t))
- 分层服务规则 Si ~ Gi(μi, σi²)
- 层间转移动力学 T(li, li-1 | Q(t))
- 稳定性条件验证 ρi = λi^eff/(μi·ci) < 1
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random
from collections import deque

from .config import VerticalQueueConfig
from .utils import MathUtils


@dataclass
class UAVOrder:
    """UAV订单对象"""
    id: int
    arrival_time: int
    priority: str  # 'high', 'medium', 'low'
    temperature_zone: str  # 'hot', 'cold', 'normal'
    size: str  # 'small', 'medium', 'large'
    urgency_level: float  # [0,1]
    wait_tolerance: int  # 最大容忍等待时间
    current_layer: int  # 当前所在层
    total_wait_time: int = 0


@dataclass
class LayerState:
    """单层队列状态"""
    layer_id: int
    height: float  # 高度(m)
    capacity: int  # 容量
    current_length: int = 0  # 当前队列长度
    waiting_orders: deque = None  # 等待队列
    service_rate: float = 1.0  # 当前服务率
    arrival_rate: float = 0.25  # 当前到达率
    total_arrivals: int = 0  # 总到达数
    total_departures: int = 0  # 总离开数
    total_transfers_in: int = 0  # 转入数
    total_transfers_out: int = 0  # 转出数
    
    def __post_init__(self):
        if self.waiting_orders is None:
            self.waiting_orders = deque()


class QueueDynamics:
    """
    队列动力学管理器

    实现01理论的核心动力学：
    1. 到达过程：λi(t) = λ0 · αi · f(urgency) · g(saturation_{i-1})
    2. 服务过程：Si ~ Gi(μi, σi²)，μi < μi+1（上升优先性，高层服务更快）
    3. 转移过程：T(li, li-1 | Q(t)) 基于等待时间和容量状态
    4. 稳定性：ρi = λi^eff/(μi·ci) < 1
    """
    
    def __init__(self, config: VerticalQueueConfig):
        self.config = config
        self.math_utils = MathUtils()
        
        # 动力学参数 (基于01理论标准配置)
        self.base_arrival_rate = config.base_arrival_rate  # λ0 = 0.25
        self.arrival_weights = np.array(config.arrival_weights)  # α = [0.1, 0.15, 0.25, 0.3, 0.2]
        self.service_rates = np.array(config.layer_service_rates)  # μ = [0.8, 0.9, 1.0, 1.2, 1.5]
        self.min_wait_times = np.array(config.min_wait_times)  # τ_min = [10, 8, 6, 4, 2]
        
        # 初始化层状态
        self.layers = self._initialize_layers()
        
        # 系统参数
        self.current_step = 0
        self.order_counter = 0
        
        # 转移参数
        self.transfer_probabilities = np.zeros(config.num_layers - 1)  # 4个转移概率
        self.transfer_enable = np.ones(config.num_layers, dtype=bool)  # 转移开关
        
        # 性能统计
        self.performance_history = {
            'throughput': [],
            'waiting_times': [],
            'queue_lengths': [],
            'load_factors': [],
            'transfer_counts': []
        }
        
        # 随机数生成器
        self.rng = np.random.RandomState(config.random_seed)
    
    def _initialize_layers(self) -> List[LayerState]:
        """
        初始化5层队列状态
        
        基于01理论的倒金字塔容量结构：C = {8, 6, 4, 3, 2}
        """
        layers = []
        for i in range(self.config.num_layers):
            layer = LayerState(
                layer_id=i,
                height=self.config.layer_heights[i],
                capacity=self.config.layer_capacities[i],
                service_rate=self.config.layer_service_rates[i],
                arrival_rate=self.base_arrival_rate * self.arrival_weights[i]
            )
            layers.append(layer)
        return layers
    
    def reset(self):
        """
        重置队列动力学系统
        """
        self.current_step = 0
        self.order_counter = 0
        
        # 清空所有队列
        for layer in self.layers:
            layer.current_length = 0
            layer.waiting_orders.clear()
            layer.total_arrivals = 0
            layer.total_departures = 0
            layer.total_transfers_in = 0
            layer.total_transfers_out = 0
        
        # 重置统计
        for key in self.performance_history:
            self.performance_history[key].clear()
    
    def step(self, action: Dict) -> Dict:
        """
        队列动力学单步更新
        
        实现01理论的完整动力学过程：
        1. 生成新到达 (NHPP过程)
        2. 处理服务离开
        3. 执行层间转移
        4. 更新等待时间
        5. 计算性能指标
        
        Args:
            action: 包含控制决策的动作字典
            
        Returns:
            队列状态信息字典
        """
        self.current_step += 1
        
        # 1. 处理新到达 (基于01理论的到达过程)
        new_arrivals = self._process_arrivals()
        
        # 2. 处理服务离开 (分层服务规则)
        service_info = self._process_service()
        
        # 3. 处理层间转移 (转移动力学)
        transfer_info = self._process_transfers()
        
        # 4. 更新等待时间
        self._update_waiting_times()
        
        # 5. 计算性能指标
        performance_metrics = self._calculate_performance_metrics()
        
        # 6. 更新历史记录
        self._update_performance_history(performance_metrics)
        
        # 7. 准备返回信息
        return {
            'queue_lengths': [layer.current_length for layer in self.layers],
            'waiting_times': self._get_average_waiting_times(),
            'throughput': performance_metrics['throughput'],
            'load_factors': self._calculate_load_factors(),
            'priority_distributions': self._get_priority_distributions(),
            'transfer_states': self._get_transfer_states(),
            'service_states': self._get_service_states(),
            'new_arrivals': new_arrivals,
            'service_completions': service_info['completions'],
            'service_requests': service_info.get('completed_orders', []),  # 为外卖柜提供服务请求
            'transfers': transfer_info['total_transfers'],
            'blocked': sum(performance_metrics.get('blocked_arrivals', []))
        }
    
    def _process_arrivals(self) -> List[int]:
        """
        处理新到达过程
        
        实现01理论的到达模型：
        λi(t) = λ0 · αi · f(urgency) · g(saturation_{i-1})
        
        Returns:
            各层新到达数量列表
        """
        new_arrivals = []
        
        for i, layer in enumerate(self.layers):
            # 计算当前层的有效到达率
            effective_rate = self._calculate_effective_arrival_rate(i)
            
            # 泊松采样
            arrivals = self.rng.poisson(effective_rate)
            
            # 容量检查和到达处理
            actual_arrivals = 0
            for _ in range(arrivals):
                if layer.current_length < layer.capacity:
                    # 创建新订单
                    order = self._create_order(i)
                    layer.waiting_orders.append(order)
                    layer.current_length += 1
                    layer.total_arrivals += 1
                    actual_arrivals += 1
                else:
                    # 容量满，订单被阻塞
                    # 可以考虑转移到上层
                    if i < len(self.layers) - 1 and self.layers[i + 1].current_length < self.layers[i + 1].capacity:
                        # 转移到上层
                        order = self._create_order(i + 1)
                        self.layers[i + 1].waiting_orders.append(order)
                        self.layers[i + 1].current_length += 1
                        self.layers[i + 1].total_arrivals += 1
                        actual_arrivals += 1
            
            new_arrivals.append(actual_arrivals)
        
        return new_arrivals
    
    def _calculate_effective_arrival_rate(self, layer_idx: int) -> float:
        """
        计算层i的有效到达率
        
        基于01理论公式：
        λi(t) = λ0 · αi · f(urgency) · g(saturation_{i-1})
        """
        # 基础到达率
        base_rate = self.base_arrival_rate * self.arrival_weights[layer_idx]
        
        # 紧急程度因子 f(urgency)
        urgency_factor = self._calculate_urgency_factor()
        
        # 下层饱和度因子 g(saturation_{i-1})
        saturation_factor = 1.0
        if layer_idx > 0:
            prev_layer = self.layers[layer_idx - 1]
            saturation = prev_layer.current_length / prev_layer.capacity if prev_layer.capacity > 0 else 0
            saturation_factor = max(0, 1 - saturation)  # 下层越满，当前层到达越少
        
        return base_rate * urgency_factor * saturation_factor
    
    def _calculate_urgency_factor(self) -> float:
        """
        计算紧急程度因子
        
        f(urgency) = Σ βu · P(urgency = u)
        """
        # 简化实现：基于当前系统负载动态调整
        total_load = sum(layer.current_length for layer in self.layers)
        max_load = sum(layer.capacity for layer in self.layers)
        
        system_pressure = total_load / max_load if max_load > 0 else 0
        
        # 系统压力越大，紧急订单比例越高，总体到达率可能增加
        urgency_factor = 1.0 + 0.5 * system_pressure
        return min(urgency_factor, 2.0)  # 限制最大增长
    
    def _create_order(self, layer_idx: int) -> UAVOrder:
        """
        创建新订单
        """
        self.order_counter += 1
        
        # 随机生成订单属性
        priorities = ['low', 'medium', 'high']
        priority_probs = [0.5, 0.3, 0.2]  # 低优先级多，高优先级少
        priority = self.rng.choice(priorities, p=priority_probs)
        
        temp_zones = ['normal', 'hot', 'cold']
        temp_probs = [0.5, 0.3, 0.2]
        temp_zone = self.rng.choice(temp_zones, p=temp_probs)
        
        sizes = ['small', 'medium', 'large']
        size_probs = [0.4, 0.4, 0.2]
        size = self.rng.choice(sizes, p=size_probs)
        
        # 紧急程度和等待容忍度
        urgency_map = {'low': 0.2, 'medium': 0.5, 'high': 0.8}
        urgency = urgency_map[priority] + self.rng.normal(0, 0.1)
        urgency = np.clip(urgency, 0, 1)
        
        wait_tolerance = int(self.rng.exponential(20)) + 5  # 5-100步的等待容忍度
        
        return UAVOrder(
            id=self.order_counter,
            arrival_time=self.current_step,
            priority=priority,
            temperature_zone=temp_zone,
            size=size,
            urgency_level=urgency,
            wait_tolerance=wait_tolerance,
            current_layer=layer_idx
        )
    
    def _process_service(self) -> Dict:
        """
        处理服务离开过程

        实现01理论的分层服务特性：
        1. 上升优先性：μi < μi+1 (高度越高服务越快，如μ1=0.8 < μ5=1.5)
        2. 容量倒金字塔：ci < ci+1 (高度越高容量越大，如C1=2 < C5=8)
        3. 紧急优先性：高紧急度订单优先服务
        """
        service_completions = []
        
        for i, layer in enumerate(self.layers):
            if layer.current_length == 0:
                service_completions.append(0)
                continue
            
            # 计算当前层的有效服务率
            effective_service_rate = layer.service_rate
            
            # 服务能力 (可以同时服务多个订单)
            service_capacity = min(layer.capacity, layer.current_length)
            
            # 为每个服务位置决定是否完成服务
            completed = 0
            orders_to_remove = []
            
            # 按优先级排序订单 (紧急优先性)
            orders_list = list(layer.waiting_orders)
            orders_list.sort(key=lambda x: (x.priority == 'high', x.priority == 'medium', -x.total_wait_time))
            
            for j, order in enumerate(orders_list[:service_capacity]):
                # 服务完成概率 (基于服务率)
                service_prob = effective_service_rate
                
                if self.rng.random() < service_prob:
                    orders_to_remove.append(order)
                    completed += 1
            
            # 移除已完成的订单
            for order in orders_to_remove:
                if order in layer.waiting_orders:
                    layer.waiting_orders.remove(order)
                    layer.current_length -= 1
                    layer.total_departures += 1
            
            service_completions.append(completed)
        
        return {
            'completions': service_completions,
            'total_completed': sum(service_completions)
        }
    
    def _process_transfers(self) -> Dict:
        """
        处理层间转移过程
        
        实现01理论的转移动力学：
        T(li, li-1 | Q(t)) = φ(wi/τi^min) · ψ((ci-1 - ni-1)/ci-1)
        """
        total_transfers = 0
        transfer_details = []
        
        # 从上层到下层逐层处理转移
        for i in range(len(self.layers) - 1, 0, -1):  # L5->L4, L4->L3, ..., L2->L1
            source_layer = self.layers[i]
            target_layer = self.layers[i - 1]
            
            if source_layer.current_length == 0 or not self.transfer_enable[i]:
                transfer_details.append(0)
                continue
            
            transfers = 0
            orders_to_transfer = []
            
            # 检查每个订单是否满足转移条件
            for order in source_layer.waiting_orders:
                # 转移条件检查
                if self._check_transfer_conditions(order, i, i - 1):
                    # 计算转移概率
                    transfer_prob = self._calculate_transfer_probability(order, i, i - 1)
                    
                    if self.rng.random() < transfer_prob:
                        orders_to_transfer.append(order)
            
            # 执行转移 (受目标层容量限制)
            for order in orders_to_transfer:
                if target_layer.current_length < target_layer.capacity:
                    # 执行转移
                    source_layer.waiting_orders.remove(order)
                    source_layer.current_length -= 1
                    source_layer.total_transfers_out += 1
                    
                    order.current_layer = i - 1
                    target_layer.waiting_orders.append(order)
                    target_layer.current_length += 1
                    target_layer.total_transfers_in += 1
                    
                    transfers += 1
                    total_transfers += 1
                else:
                    break  # 目标层满了，停止转移
            
            transfer_details.append(transfers)
        
        return {
            'transfer_details': transfer_details,
            'total_transfers': total_transfers
        }
    
    def _check_transfer_conditions(self, order: UAVOrder, from_layer: int, to_layer: int) -> bool:
        """
        检查转移条件
        
        基于01理论的转移触发条件：
        1. 时间条件：wi ≥ τi^min
        2. 空间条件：nj < cj  
        3. 优先级条件：priority ≥ threshold_j
        """
        # 1. 时间条件
        min_wait = self.min_wait_times[from_layer]
        if order.total_wait_time < min_wait:
            return False
        
        # 2. 空间条件 (在_process_transfers中检查)
        target_layer = self.layers[to_layer]
        if target_layer.current_length >= target_layer.capacity:
            return False
        
        # 3. 优先级条件 (高优先级更容易转移)
        if order.priority == 'low' and order.total_wait_time < min_wait * 2:
            return False
        
        return True
    
    def _calculate_transfer_probability(self, order: UAVOrder, from_layer: int, to_layer: int) -> float:
        """
        计算转移概率
        
        基于01理论公式：
        T(li, li-1 | Q(t)) = φ(wi/τi^min) · ψ((ci-1 - ni-1)/ci-1)
        """
        # φ(wi/τi^min) - 等待时间激活函数
        min_wait = self.min_wait_times[from_layer]
        wait_ratio = order.total_wait_time / min_wait if min_wait > 0 else 1.0
        phi = min(1.0, max(0.0, wait_ratio))  # φ(x) = min(1, max(0, x))
        
        # ψ((ci-1 - ni-1)/ci-1) - 空位可用性函数
        target_layer = self.layers[to_layer]
        available_ratio = (target_layer.capacity - target_layer.current_length) / target_layer.capacity
        available_ratio = max(0, available_ratio)
        gamma = 2.0  # 指数参数
        psi = available_ratio ** gamma  # ψ(x) = x^γ
        
        # 最终转移概率
        transfer_prob = phi * psi
        
        # 优先级调整
        priority_multiplier = {'low': 0.8, 'medium': 1.0, 'high': 1.2}
        transfer_prob *= priority_multiplier.get(order.priority, 1.0)
        
        return min(transfer_prob, 1.0)
    
    def _update_waiting_times(self):
        """
        更新所有订单的等待时间
        """
        for layer in self.layers:
            for order in layer.waiting_orders:
                order.total_wait_time += 1
    
    def _calculate_performance_metrics(self) -> Dict:
        """
        计算性能指标
        
        基于01理论的性能度量：
        1. 系统吞吐量：Λ_system = Σ λi · (1 - P_block,i)
        2. 平均等待时间：Wi = (ρi/(1-ρi)) · (1/μi) · (1+Cs²)/2
        3. 负载系数：ρi = λi^eff/(μi·ci)
        """
        # 系统吞吐量 (当前步的服务完成数)
        throughput = sum(layer.total_departures for layer in self.layers)
        if self.current_step > 0:
            throughput = throughput / self.current_step  # 平均每步吞吐量
        
        # 平均等待时间
        total_wait = 0
        total_orders = 0
        for layer in self.layers:
            for order in layer.waiting_orders:
                total_wait += order.total_wait_time
                total_orders += 1
        
        avg_waiting_time = total_wait / total_orders if total_orders > 0 else 0
        
        # 负载系数
        load_factors = self._calculate_load_factors()
        
        # 层利用率
        utilizations = []
        for layer in self.layers:
            util = layer.current_length / layer.capacity if layer.capacity > 0 else 0
            utilizations.append(util)
        
        return {
            'throughput': throughput,
            'avg_waiting_time': avg_waiting_time,
            'load_factors': load_factors,
            'utilizations': utilizations,
            'total_orders_in_system': total_orders
        }
    
    def _calculate_load_factors(self) -> Dict[str, float]:
        """
        计算各层负载系数
        
        ρi = λi^eff/(μi·ci)
        """
        load_factors = {}
        
        for i, layer in enumerate(self.layers):
            # 有效到达率 (包括从上层转移的)
            effective_arrival = self._calculate_effective_arrival_rate(i)
            
            # 添加从上层转移的到达率
            if i < len(self.layers) - 1:
                transfer_rate = self.transfer_probabilities[i] * effective_arrival
                effective_arrival += transfer_rate
            
            # 负载系数
            service_capacity = layer.service_rate * layer.capacity
            rho = effective_arrival / service_capacity if service_capacity > 0 else 0
            
            load_factors[f'layer_{i}'] = min(rho, 2.0)  # 限制最大值避免数值问题
        
        return load_factors
    
    def _get_average_waiting_times(self) -> List[float]:
        """
        获取各层平均等待时间
        """
        waiting_times = []
        for layer in self.layers:
            if len(layer.waiting_orders) > 0:
                avg_wait = sum(order.total_wait_time for order in layer.waiting_orders) / len(layer.waiting_orders)
            else:
                avg_wait = 0.0
            waiting_times.append(avg_wait)
        return waiting_times
    
    def _get_priority_distributions(self) -> List[List[float]]:
        """
        获取各层优先级分布
        """
        distributions = []
        for layer in self.layers:
            if len(layer.waiting_orders) > 0:
                priorities = [order.priority for order in layer.waiting_orders]
                high_count = priorities.count('high')
                medium_count = priorities.count('medium')
                low_count = priorities.count('low')
                total = len(priorities)
                
                dist = [high_count/total, medium_count/total, low_count/total]
            else:
                dist = [0.33, 0.33, 0.34]  # 默认均匀分布
            
            distributions.append(dist)
        return distributions
    
    def _get_transfer_states(self) -> List[float]:
        """
        获取各层转移状态
        """
        transfer_states = []
        for i, layer in enumerate(self.layers):
            if i == 0:  # 最底层不能转移
                transfer_states.append(0.0)
            else:
                # 计算当前层准备转移的订单比例
                ready_to_transfer = 0
                for order in layer.waiting_orders:
                    if self._check_transfer_conditions(order, i, i-1):
                        ready_to_transfer += 1
                
                ratio = ready_to_transfer / len(layer.waiting_orders) if len(layer.waiting_orders) > 0 else 0
                transfer_states.append(ratio)
        
        return transfer_states
    
    def _get_service_states(self) -> List[float]:
        """
        获取各层服务状态
        """
        service_states = []
        for layer in self.layers:
            # 服务活跃度 = 当前长度 / 容量 * 服务率
            if layer.capacity > 0:
                activity = (layer.current_length / layer.capacity) * layer.service_rate
            else:
                activity = 0.0
            service_states.append(min(activity, 1.0))
        
        return service_states
    
    def _update_performance_history(self, metrics: Dict):
        """
        更新性能历史记录
        """
        self.performance_history['throughput'].append(metrics['throughput'])
        self.performance_history['waiting_times'].append(metrics['avg_waiting_time'])
        self.performance_history['queue_lengths'].append([layer.current_length for layer in self.layers])
        self.performance_history['load_factors'].append(list(metrics['load_factors'].values()))
        
        # 保持历史记录长度
        max_history = 1000
        for key in self.performance_history:
            if len(self.performance_history[key]) > max_history:
                self.performance_history[key].pop(0)
    
    def update_transfer_probabilities(self, transfer_decisions: np.ndarray):
        """
        更新转移概率 (基于智能体动作)
        """
        # transfer_decisions是5维binary vector，表示每层是否允许转移
        self.transfer_enable = transfer_decisions.astype(bool)
    
    def update_service_priorities(self, service_priorities: np.ndarray):
        """
        更新服务优先级权重
        """
        # 调整各层服务率
        normalized_priorities = service_priorities / (np.sum(service_priorities) + 1e-8)
        for i, layer in enumerate(self.layers):
            if i < len(normalized_priorities):
                # 在基础服务率基础上调整
                layer.service_rate = self.service_rates[i] * (0.5 + normalized_priorities[i])
    
    def update_arrival_weights(self, arrival_weights: np.ndarray):
        """
        更新到达权重分配
        """
        # 更新到达权重
        self.arrival_weights = arrival_weights / (np.sum(arrival_weights) + 1e-8)
        
        # 更新各层到达率
        for i, layer in enumerate(self.layers):
            if i < len(self.arrival_weights):
                layer.arrival_rate = self.base_arrival_rate * self.arrival_weights[i]
    
    def get_queue_lengths(self) -> List[int]:
        """获取当前队列长度"""
        return [layer.current_length for layer in self.layers]
    
    def get_waiting_times(self) -> List[float]:
        """获取平均等待时间"""
        return self._get_average_waiting_times()
    
    def get_load_factors(self) -> Dict[str, float]:
        """获取负载系数"""
        return self._calculate_load_factors()
    
    def get_system_info(self) -> Dict:
        """
        获取系统详细信息
        """
        return {
            'current_step': self.current_step,
            'total_orders_generated': self.order_counter,
            'layer_stats': [
                {
                    'layer_id': layer.layer_id,
                    'height': layer.height,
                    'capacity': layer.capacity,
                    'current_length': layer.current_length,
                    'service_rate': layer.service_rate,
                    'arrival_rate': layer.arrival_rate,
                    'total_arrivals': layer.total_arrivals,
                    'total_departures': layer.total_departures,
                    'total_transfers_in': layer.total_transfers_in,
                    'total_transfers_out': layer.total_transfers_out
                }
                for layer in self.layers
            ],
            'performance_history': self.performance_history
        }


# 测试队列动力学
if __name__ == "__main__":
    from .config import VerticalQueueConfig
    
    config = VerticalQueueConfig()
    queue_dynamics = QueueDynamics(config)
    
    print("队列动力学模块创建成功!")
    print(f"层数: {len(queue_dynamics.layers)}")
    
    # 显示层信息
    for i, layer in enumerate(queue_dynamics.layers):
        print(f"L{i+1}({layer.height}m): 容量{layer.capacity}, 服务率{layer.service_rate:.2f}")
    
    # 测试动力学步进
    dummy_action = {
        'transfer_decisions': np.array([1, 1, 1, 1, 0]),
        'service_priorities': np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
        'arrival_weights': np.array([0.1, 0.15, 0.25, 0.3, 0.2])
    }
    
    print("\n开始动力学仿真...")
    for step in range(10):
        info = queue_dynamics.step(dummy_action)
        print(f"Step {step + 1}: 队列长度{info['queue_lengths']}, 吞吐量{info['throughput']:.3f}")
    
    print("\n队列动力学测试完成!")