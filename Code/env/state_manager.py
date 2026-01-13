"""
状态管理器
State Manager

实现01理论中的128维状态空间设计：
- 6个维度段的分层编码
- 语义分离的观测空间
- 马尔可夫性保证的状态表示
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .config import VerticalQueueConfig


@dataclass
class StateSegment:
    """状态段定义"""
    name: str
    start_idx: int
    end_idx: int
    size: int
    description: str


class StateManager:
    """
    状态管理器
    
    负责128维状态空间的构建和管理，实现：
    1. 分层编码：6个语义维度段
    2. 马尔可夫性：包含完整的决策相关信息
    3. 归一化：所有状态值归一化到[0,1]区间
    4. 语义分离：不同类型信息分开编码
    """
    
    def __init__(self, config: VerticalQueueConfig):
        self.config = config
        self.state_dim = 128
        
        # 定义6个状态维度段
        self._define_state_segments()
        
        # 初始化状态缓存
        self.current_state = np.zeros(self.state_dim, dtype=np.float32)
        self.state_history = []
        self.action_history = []
        
        # 状态统计信息（用于归一化）
        self.state_stats = self._initialize_state_stats()
    
    def _define_state_segments(self):
        """
        定义128维状态的6个维度段
        
        基于01理论的状态空间设计：
        1. 队列状态段 (40维): 5层队列的详细状态
        2. 外卖柜状态段 (32维): 24格外卖柜状态 + 8维温区信息  
        3. 系统性能段 (16维): 吞吐量、等待时间、利用率等
        4. 动力学参数段 (16维): 到达率、服务率、转移概率
        5. 历史信息段 (16维): 近期状态变化趋势
        6. 控制信息段 (8维): 当前控制策略和动作效果
        """
        segments = []
        current_idx = 0
        
        # 1. 队列状态段 (40维)
        segments.append(StateSegment(
            name="queue_states",
            start_idx=current_idx,
            end_idx=current_idx + 40,
            size=40,
            description="5层队列的详细状态：长度、等待时间、优先级分布等"
        ))
        current_idx += 40
        
        # 2. 外卖柜状态段 (32维)
        segments.append(StateSegment(
            name="cabinet_states",
            start_idx=current_idx,
            end_idx=current_idx + 32,
            size=32,
            description="24格外卖柜状态 + 8维温区管理信息"
        ))
        current_idx += 32
        
        # 3. 系统性能段 (16维)
        segments.append(StateSegment(
            name="performance_metrics",
            start_idx=current_idx,
            end_idx=current_idx + 16,
            size=16,
            description="系统性能指标：吞吐量、利用率、稳定性等"
        ))
        current_idx += 16
        
        # 4. 动力学参数段 (16维)
        segments.append(StateSegment(
            name="dynamics_params",
            start_idx=current_idx,
            end_idx=current_idx + 16,
            size=16,
            description="当前动力学参数：到达率、服务率、转移概率"
        ))
        current_idx += 16
        
        # 5. 历史信息段 (16维)
        segments.append(StateSegment(
            name="history_info",
            start_idx=current_idx,
            end_idx=current_idx + 16,
            size=16,
            description="历史状态变化趋势和统计信息"
        ))
        current_idx += 16
        
        # 6. 控制信息段 (8维)
        segments.append(StateSegment(
            name="control_info",
            start_idx=current_idx,
            end_idx=current_idx + 8,
            size=8,
            description="当前控制策略和动作效果反馈"
        ))
        current_idx += 8
        
        # 验证总维度
        assert current_idx == self.state_dim, f"状态维度不匹配: {current_idx} != {self.state_dim}"
        
        # 存储段定义
        self.segments = {seg.name: seg for seg in segments}
        self.segment_list = segments
    
    def _initialize_state_stats(self) -> Dict:
        """
        初始化状态统计信息
        
        用于状态归一化和异常检测
        """
        return {
            'min_values': np.zeros(self.state_dim),
            'max_values': np.ones(self.state_dim),
            'mean_values': np.zeros(self.state_dim),
            'std_values': np.ones(self.state_dim),
            'update_count': 0
        }
    
    def reset(self):
        """
        重置状态管理器
        """
        self.current_state = np.zeros(self.state_dim, dtype=np.float32)
        self.state_history = []
        self.action_history = []
    
    def update_state(self, queue_info: Dict, cabinet_info: Dict, 
                     performance_info: Dict, dynamics_info: Dict):
        """
        更新完整的128维状态向量
        
        Args:
            queue_info: 队列信息字典
            cabinet_info: 外卖柜信息字典  
            performance_info: 性能信息字典
            dynamics_info: 动力学信息字典
        """
        # 构建新状态向量
        new_state = np.zeros(self.state_dim, dtype=np.float32)
        
        # 1. 更新队列状态段
        self._update_queue_states(new_state, queue_info)
        
        # 2. 更新外卖柜状态段
        self._update_cabinet_states(new_state, cabinet_info)
        
        # 3. 更新系统性能段
        self._update_performance_metrics(new_state, performance_info)
        
        # 4. 更新动力学参数段
        self._update_dynamics_params(new_state, dynamics_info)
        
        # 5. 更新历史信息段
        self._update_history_info(new_state)
        
        # 6. 更新控制信息段
        self._update_control_info(new_state)
        
        # 归一化状态
        new_state = self._normalize_state(new_state)
        
        # 更新状态历史
        self.state_history.append(self.current_state.copy())
        if len(self.state_history) > self.config.history_length:
            self.state_history.pop(0)
        
        # 更新当前状态
        self.current_state = new_state
        
        # 更新状态统计
        self._update_state_stats(new_state)
    
    def _update_queue_states(self, state: np.ndarray, queue_info: Dict):
        """
        更新队列状态段 (40维)
        
        每层8维：
        - 队列长度 (1维)
        - 平均等待时间 (1维)  
        - 优先级分布 (3维: high/medium/low)
        - 负载系数ρ (1维)
        - 转移状态 (1维)
        - 服务状态 (1维)
        """
        seg = self.segments["queue_states"]
        
        queue_lengths = queue_info.get('queue_lengths', [0] * self.config.num_layers)
        waiting_times = queue_info.get('waiting_times', [0] * self.config.num_layers)
        priority_dists = queue_info.get('priority_distributions', 
                                        [[0.33, 0.33, 0.34]] * self.config.num_layers)
        load_factors = queue_info.get('load_factors', [0] * self.config.num_layers)
        transfer_states = queue_info.get('transfer_states', [0] * self.config.num_layers)
        service_states = queue_info.get('service_states', [0] * self.config.num_layers)
        
        for i in range(self.config.num_layers):
            base_idx = seg.start_idx + i * 8
            
            # 队列长度 (归一化到容量)
            capacity = self.config.layer_capacities[i]
            state[base_idx] = queue_lengths[i] / capacity if capacity > 0 else 0
            
            # 平均等待时间 (归一化到最大等待时间)
            max_wait = self.config.max_waiting_time
            state[base_idx + 1] = min(waiting_times[i] / max_wait, 1.0)
            
            # 优先级分布 (3维，已归一化)
            priority_dist = priority_dists[i] if i < len(priority_dists) else [0.33, 0.33, 0.34]
            state[base_idx + 2:base_idx + 5] = priority_dist[:3]
            
            # 负载系数 (ρi，理论上应 < 1)
            state[base_idx + 5] = min(load_factors[i] if i < len(load_factors) else 0, 1.0)
            
            # 转移状态 (当前层向下转移的活跃度)
            state[base_idx + 6] = transfer_states[i] if i < len(transfer_states) else 0
            
            # 服务状态 (当前层服务活跃度)  
            state[base_idx + 7] = service_states[i] if i < len(service_states) else 0
    
    def _update_cabinet_states(self, state: np.ndarray, cabinet_info: Dict):
        """
        更新外卖柜状态段 (32维)
        
        24维格子状态 + 8维温区管理：
        - 24个格子占用状态 (24维，0/1)
        - 3个温区温度 (3维，归一化)  
        - 3个温区负载 (3维，占用率)
        - 温控系统状态 (1维)
        - 服务队列长度 (1维)
        """
        seg = self.segments["cabinet_states"]
        
        # 24个格子状态
        grid_states = cabinet_info.get('grid_states', [0] * 24)
        state[seg.start_idx:seg.start_idx + 24] = grid_states[:24]
        
        # 温区温度 (归一化到目标温度范围)
        temperatures = cabinet_info.get('temperatures', [20, 60, 5])  # 常温、热、冷
        temp_ranges = [(15, 25), (55, 65), (0, 10)]  # 各温区正常范围
        
        for i, (temp, (min_t, max_t)) in enumerate(zip(temperatures, temp_ranges)):
            normalized_temp = (temp - min_t) / (max_t - min_t) if max_t > min_t else 0.5
            state[seg.start_idx + 24 + i] = np.clip(normalized_temp, 0, 1)
        
        # 温区负载
        zone_loads = cabinet_info.get('zone_loads', [0, 0, 0])
        state[seg.start_idx + 27:seg.start_idx + 30] = zone_loads[:3]
        
        # 温控系统状态
        thermal_status = cabinet_info.get('thermal_status', 0)
        state[seg.start_idx + 30] = thermal_status
        
        # 服务队列长度
        service_queue = cabinet_info.get('service_queue_length', 0)
        max_service_queue = 10  # 假设最大服务队列长度
        state[seg.start_idx + 31] = min(service_queue / max_service_queue, 1.0)
    
    def _update_performance_metrics(self, state: np.ndarray, performance_info: Dict):
        """
        更新系统性能段 (16维)
        
        关键性能指标：
        - 系统吞吐量 (1维)
        - 平均等待时间 (1维)
        - 等待时间方差 (1维) 
        - 公平性指标 (1维)
        - 各层利用率 (5维)
        - 稳定性指标 (3维)
        - 效率指标 (4维)
        """
        seg = self.segments["performance_metrics"]
        
        # 系统吞吐量
        throughput = performance_info.get('throughput', 0)
        max_throughput = self.config.theoretical_max_throughput
        state[seg.start_idx] = min(throughput / max_throughput, 1.0)
        
        # 平均等待时间
        avg_waiting = performance_info.get('avg_waiting_time', 0)
        max_wait = self.config.max_waiting_time
        state[seg.start_idx + 1] = min(avg_waiting / max_wait, 1.0)
        
        # 等待时间方差  
        wait_variance = performance_info.get('waiting_time_variance', 0)
        max_variance = max_wait * max_wait
        state[seg.start_idx + 2] = min(wait_variance / max_variance, 1.0)
        
        # 公平性指标
        fairness = performance_info.get('fairness', 1.0)
        state[seg.start_idx + 3] = fairness
        
        # 各层利用率 (5维)
        utilizations = performance_info.get('layer_utilizations', [0] * 5)
        state[seg.start_idx + 4:seg.start_idx + 9] = utilizations[:5]
        
        # 稳定性指标 (3维)
        stability_metrics = performance_info.get('stability_metrics', [0, 0, 0])
        state[seg.start_idx + 9:seg.start_idx + 12] = stability_metrics[:3]
        
        # 效率指标 (4维)
        efficiency_metrics = performance_info.get('efficiency_metrics', [0, 0, 0, 0])
        state[seg.start_idx + 12:seg.start_idx + 16] = efficiency_metrics[:4]
    
    def _update_dynamics_params(self, state: np.ndarray, dynamics_info: Dict):
        """
        更新动力学参数段 (16维)
        
        当前系统动力学参数：
        - 各层到达率 (5维)
        - 各层服务率 (5维)  
        - 层间转移概率 (4维)
        - 系统负载 (1维)
        - 控制参数 (1维)
        """
        seg = self.segments["dynamics_params"]
        
        # 各层到达率 (归一化到基础到达率)
        arrival_rates = dynamics_info.get('arrival_rates', [0] * 5)
        base_rate = self.config.base_arrival_rate
        for i in range(5):
            rate = arrival_rates[i] if i < len(arrival_rates) else 0
            state[seg.start_idx + i] = min(rate / (base_rate * 2), 1.0)  # 假设最大为2倍基础率
        
        # 各层服务率 (归一化到理论最大服务率)
        service_rates = dynamics_info.get('service_rates', [0] * 5)
        max_service = max(self.config.layer_service_rates) * 2  # 假设最大为理论值的2倍
        for i in range(5):
            rate = service_rates[i] if i < len(service_rates) else 0
            state[seg.start_idx + 5 + i] = min(rate / max_service, 1.0)
        
        # 层间转移概率 (4维，L5->L4, L4->L3, L3->L2, L2->L1)
        transfer_probs = dynamics_info.get('transfer_probabilities', [0] * 4)
        state[seg.start_idx + 10:seg.start_idx + 14] = transfer_probs[:4]
        
        # 系统负载
        system_load = dynamics_info.get('system_load', 0)
        state[seg.start_idx + 14] = min(system_load, 1.0)
        
        # 控制参数 (当前控制强度)
        control_intensity = dynamics_info.get('control_intensity', 0)
        state[seg.start_idx + 15] = control_intensity
    
    def _update_history_info(self, state: np.ndarray):
        """
        更新历史信息段 (16维)
        
        历史趋势和统计信息：
        - 近期性能趋势 (8维)
        - 状态变化速率 (4维)
        - 异常检测指标 (2维)
        - 学习进度指标 (2维)
        """
        seg = self.segments["history_info"]
        
        if len(self.state_history) > 0:
            # 近期性能趋势 (计算最近几步的性能变化)
            recent_states = self.state_history[-min(8, len(self.state_history)):]
            if len(recent_states) > 1:
                # 计算性能段的变化趋势
                perf_seg = self.segments["performance_metrics"]
                recent_perf = [s[perf_seg.start_idx:perf_seg.start_idx+8] for s in recent_states]
                
                # 计算趋势 (线性拟合斜率)
                for i in range(8):
                    if len(recent_perf) > 1:
                        values = [perf[i] if i < len(perf) else 0 for perf in recent_perf]
                        trend = self._calculate_trend(values)
                        state[seg.start_idx + i] = np.clip((trend + 1) / 2, 0, 1)  # 归一化到[0,1]
            
            # 状态变化速率
            if len(self.state_history) > 0:
                last_state = self.state_history[-1]
                state_diff = np.abs(self.current_state - last_state)
                # 取几个关键段的平均变化率
                key_segments = ["queue_states", "performance_metrics", "dynamics_params"]
                for i, seg_name in enumerate(key_segments[:4]):
                    if seg_name in self.segments:
                        seg_info = self.segments[seg_name]
                        avg_change = np.mean(state_diff[seg_info.start_idx:seg_info.end_idx])
                        state[seg.start_idx + 8 + i] = min(avg_change * 10, 1.0)  # 放大变化率
            
            # 异常检测指标 (简单实现)
            state[seg.start_idx + 12] = self._detect_anomaly()
            state[seg.start_idx + 13] = self._calculate_stability_score()
            
            # 学习进度指标 (可以根据需要实现更复杂的逻辑)
            state[seg.start_idx + 14] = min(len(self.state_history) / 1000, 1.0)  # 经验积累
            state[seg.start_idx + 15] = self._calculate_learning_progress()
    
    def _update_control_info(self, state: np.ndarray):
        """
        更新控制信息段 (8维)
        
        当前控制策略和效果：
        - 最近动作效果 (4维)
        - 控制策略参数 (3维)  
        - 自适应参数 (1维)
        """
        seg = self.segments["control_info"]
        
        if len(self.action_history) > 0:
            last_action = self.action_history[-1]
            
            # 最近动作效果 (简化评估)
            # 这里可以实现更复杂的动作效果评估逻辑
            action_effects = [0.5, 0.5, 0.5, 0.5]  # 占位符
            state[seg.start_idx:seg.start_idx + 4] = action_effects
            
            # 控制策略参数
            if 'service_priorities' in last_action:
                priorities = last_action['service_priorities'][:3]
                state[seg.start_idx + 4:seg.start_idx + 7] = priorities
            
            # 自适应参数 (控制强度自适应)
            adaptive_param = self._calculate_adaptive_param()
            state[seg.start_idx + 7] = adaptive_param
    
    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """
        归一化状态向量
        
        确保所有状态值在[0,1]区间内
        """
        # 简单的clip操作，更复杂的归一化可以基于状态统计
        return np.clip(state, 0.0, 1.0)
    
    def _update_state_stats(self, state: np.ndarray):
        """
        更新状态统计信息
        """
        stats = self.state_stats
        stats['update_count'] += 1
        
        # 更新最小最大值
        stats['min_values'] = np.minimum(stats['min_values'], state)
        stats['max_values'] = np.maximum(stats['max_values'], state)
        
        # 更新均值和标准差 (在线更新)
        alpha = 1.0 / min(stats['update_count'], 1000)  # 学习率衰减
        stats['mean_values'] = (1 - alpha) * stats['mean_values'] + alpha * state
        
        diff = state - stats['mean_values']
        stats['std_values'] = (1 - alpha) * stats['std_values'] + alpha * (diff * diff)
    
    def _calculate_trend(self, values: List[float]) -> float:
        """
        计算数值序列的趋势 (简单线性拟合)
        
        Returns:
            趋势值，正数表示上升，负数表示下降
        """
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x = np.arange(n)
        y = np.array(values)
        
        # 简单线性回归
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if denominator == 0:
            return 0.0
        
        slope = numerator / denominator
        return np.clip(slope, -1.0, 1.0)
    
    def _detect_anomaly(self) -> float:
        """
        简单异常检测
        
        Returns:
            异常程度 [0,1]，0表示正常，1表示异常
        """
        if len(self.state_history) < 5:
            return 0.0
        
        # 检查最近状态与历史均值的偏差
        recent_mean = np.mean(self.state_history[-5:], axis=0)
        historical_mean = np.mean(self.state_history[:-5], axis=0) if len(self.state_history) > 5 else recent_mean
        
        deviation = np.mean(np.abs(recent_mean - historical_mean))
        return min(deviation * 5, 1.0)  # 放大偏差
    
    def _calculate_stability_score(self) -> float:
        """
        计算系统稳定性得分
        
        Returns:
            稳定性得分 [0,1]，1表示最稳定
        """
        if len(self.state_history) < 3:
            return 1.0
        
        # 计算最近状态的方差
        recent_states = np.array(self.state_history[-3:])
        variance = np.mean(np.var(recent_states, axis=0))
        
        # 转换为稳定性得分
        stability = 1.0 / (1.0 + variance * 100)
        return stability
    
    def _calculate_learning_progress(self) -> float:
        """
        计算学习进度指标
        
        Returns:
            学习进度 [0,1]
        """
        # 简单实现：基于性能改善程度
        if len(self.state_history) < 10:
            return 0.0
        
        # 比较最近和早期的性能
        perf_seg = self.segments["performance_metrics"]
        recent_perf = np.mean([s[perf_seg.start_idx] for s in self.state_history[-5:]])
        early_perf = np.mean([s[perf_seg.start_idx] for s in self.state_history[:5]])
        
        improvement = recent_perf - early_perf
        return np.clip((improvement + 1) / 2, 0, 1)
    
    def _calculate_adaptive_param(self) -> float:
        """
        计算自适应控制参数
        
        Returns:
            自适应参数 [0,1]
        """
        # 基于系统性能自动调整控制强度
        if len(self.state_history) < 2:
            return 0.5
        
        # 简单实现：基于性能变化调整
        perf_seg = self.segments["performance_metrics"]
        current_perf = self.current_state[perf_seg.start_idx]
        last_perf = self.state_history[-1][perf_seg.start_idx] if self.state_history else current_perf
        
        perf_change = current_perf - last_perf
        
        # 性能下降时增加控制强度，性能提升时减少控制强度
        if perf_change < 0:
            return min(0.8, 0.5 - perf_change)
        else:
            return max(0.2, 0.5 - perf_change * 0.5)
    
    def get_observation(self) -> np.ndarray:
        """
        获取当前观测状态
        
        Returns:
            128维状态向量
        """
        return self.current_state.copy()
    
    def get_state_info(self) -> Dict:
        """
        获取状态管理器的详细信息
        
        Returns:
            状态信息字典
        """
        return {
            'state_dim': self.state_dim,
            'segments': {name: {
                'start': seg.start_idx,
                'end': seg.end_idx,
                'size': seg.size,
                'description': seg.description
            } for name, seg in self.segments.items()},
            'history_length': len(self.state_history),
            'state_stats': self.state_stats
        }
    
    def update_action_history(self, action: Dict):
        """
        更新动作历史
        
        Args:
            action: 动作字典
        """
        self.action_history.append(action.copy())
        if len(self.action_history) > self.config.history_length:
            self.action_history.pop(0)
    
    def get_segment_state(self, segment_name: str) -> np.ndarray:
        """
        获取特定段的状态
        
        Args:
            segment_name: 段名称
            
        Returns:
            该段的状态向量
        """
        if segment_name not in self.segments:
            raise ValueError(f"Unknown segment: {segment_name}")
        
        seg = self.segments[segment_name]
        return self.current_state[seg.start_idx:seg.end_idx].copy()
    
    def parse_state(self, state: np.ndarray) -> Dict:
        """
        解析状态向量为各个段
        
        Args:
            state: 128维状态向量
            
        Returns:
            解析后的状态字典
        """
        if len(state) != self.state_dim:
            raise ValueError(f"状态维度不匹配: 期望{self.state_dim}, 实际{len(state)}")
        
        parsed = {}
        for segment_name, seg in self.segments.items():
            parsed[segment_name] = state[seg.start_idx:seg.end_idx].copy()
        
        return parsed


# 测试状态管理器
if __name__ == "__main__":
    from .config import VerticalQueueConfig
    
    config = VerticalQueueConfig()
    state_manager = StateManager(config)
    
    print("状态管理器创建成功!")
    print(f"状态维度: {state_manager.state_dim}")
    print(f"状态段数: {len(state_manager.segments)}")
    
    # 显示状态段信息
    for name, seg in state_manager.segments.items():
        print(f"{name}: [{seg.start_idx}:{seg.end_idx}] ({seg.size}维) - {seg.description}")
    
    # 测试状态更新
    dummy_info = {
        'queue_lengths': [2, 3, 1, 2, 1],
        'waiting_times': [5, 8, 3, 6, 2],
        'priority_distributions': [[0.3, 0.4, 0.3]] * 5,
        'load_factors': [0.5, 0.6, 0.4, 0.7, 0.3]
    }
    
    state_manager.update_state(dummy_info, {}, {}, {})
    observation = state_manager.get_observation()
    
    print(f"\n观测状态维度: {observation.shape}")
    print(f"状态值范围: [{observation.min():.3f}, {observation.max():.3f}]")
    print("状态管理器测试完成!")