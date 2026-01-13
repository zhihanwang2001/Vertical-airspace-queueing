"""
垂直分层队列环境
Vertical Stratified Queuing Environment

基于01理论文档的完整实现：
- 垂直分层队列系统 V = (L, C, T, S)
- 5层倒金字塔容量结构 
- 层间转移动力学
- 多目标优化目标函数
"""

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from .config import VerticalQueueConfig
from .state_manager import StateManager
from .queue_dynamics import QueueDynamics
from .delivery_cabinet import DeliveryCabinet
from .reward_function import RewardFunction
from .utils import MathUtils


class VerticalQueueEnv(gym.Env):
    """
    垂直分层队列环境
    
    实现01理论中的核心概念：
    1. 垂直分层队列系统 V = (L, C, T, S)
    2. 倒金字塔容量结构 C = {8, 6, 4, 3, 2}
    3. 分层转移动力学 T(li, li-1 | Q(t))
    4. 多目标优化 J(π) = w1·Throughput - w2·W̄ - w3·Var(W) + w4·Fairness
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, config: Optional[VerticalQueueConfig] = None, render_mode: str = None, 
                 reward_weights: Optional['RewardWeights'] = None, **kwargs):
        super().__init__()
        
        # 配置参数
        self.config = config or VerticalQueueConfig()
        self.render_mode = render_mode
        
        # 核心组件初始化
        self.state_manager = StateManager(self.config)
        self.queue_dynamics = QueueDynamics(self.config)
        self.delivery_cabinet = DeliveryCabinet(self.config)
        self.reward_function = RewardFunction(self.config, weights=reward_weights)
        self.math_utils = MathUtils()
        
        # 环境状态
        self.current_step = 0
        self.max_steps = self.config.max_episode_steps
        
        # 性能统计
        self.performance_stats = {
            'total_throughput': 0,
            'total_waiting_time': 0,
            'layer_utilizations': [0] * self.config.num_layers,
            'successful_transfers': 0,
            'blocked_arrivals': 0
        }
        
        # 定义动作和观测空间
        self._setup_spaces()
        
        # 初始化状态
        self.reset()
    
    def _setup_spaces(self):
        """
        设置动作空间和观测空间
        
        基于01理论设计：
        - 观测空间：128维状态向量 (6个维度段)
        - 动作空间：混合动作空间 (连续移动 + 离散队列决策)
        """
        # 观测空间: 128维状态向量
        # 按照state_manager.py中的6个维度段设计
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(128,),
            dtype=np.float32
        )
        
        # 动作空间: 混合动作空间
        # 1. 层间转移决策 (5层，每层可选择转移到下一层或留在当前层)
        # 2. 服务调度策略 (优先级调整)
        # 3. 到达率控制 (动态调整各层到达权重)
        self.action_space = spaces.Dict({
            # 层间转移决策 (5层，每层二选一：转移/不转移)
            'transfer_decisions': spaces.MultiBinary(self.config.num_layers),
            
            # 服务优先级权重 (5层连续权重，归一化后使用)
            'service_priorities': spaces.Box(
                low=0.0, high=1.0, 
                shape=(self.config.num_layers,), 
                dtype=np.float32
            ),
            
            # 到达分配权重 (5层到达权重调整)
            'arrival_weights': spaces.Box(
                low=0.0, high=1.0,
                shape=(self.config.num_layers,),
                dtype=np.float32
            )
        })
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        重置环境到初始状态
        
        Returns:
            observation: 128维初始状态向量
            info: 环境信息字典
        """
        super().reset(seed=seed)
        
        # 重置时间步
        self.current_step = 0
        
        # 重置所有组件
        self.state_manager.reset()
        self.queue_dynamics.reset()
        self.delivery_cabinet.reset()
        
        # 重置性能统计
        self.performance_stats = {
            'total_throughput': 0,
            'total_waiting_time': 0,
            'layer_utilizations': [0] * self.config.num_layers,
            'successful_transfers': 0,
            'blocked_arrivals': 0
        }
        
        # 获取初始观测
        observation = self.state_manager.get_observation()
        
        # 环境信息
        info = {
            'step': self.current_step,
            'queue_lengths': self.queue_dynamics.get_queue_lengths(),
            'cabinet_occupancy': self.delivery_cabinet.get_occupancy_rate(),
            'system_utilization': self._calculate_system_utilization()
        }
        
        return observation, info
    
    def step(self, action: Dict):
        """
        环境步进函数
        
        实现01理论中的系统动力学：
        1. 执行动作 (层间转移、服务调度)
        2. 更新队列状态 (到达、服务、转移)
        3. 计算奖励 (多目标优化函数)
        4. 更新外卖柜状态
        5. 检查终止条件
        
        Args:
            action: 动作字典，包含transfer_decisions, service_priorities, arrival_weights
            
        Returns:
            observation: 新的128维状态向量
            reward: 标量奖励值
            terminated: 是否达到终止条件
            truncated: 是否达到时间限制
            info: 环境信息字典
        """
        # 增加时间步
        self.current_step += 1
        
        # 1. 处理动作
        self._process_action(action)
        
        # 2. 更新队列动力学 (基于01理论的状态转移)
        queue_info = self.queue_dynamics.step(action)
        
        # 3. 更新外卖柜状态
        cabinet_info = self.delivery_cabinet.step(queue_info['service_requests'])
        
        # 4. 更新性能统计
        self._update_performance_stats(queue_info, cabinet_info)
        
        # 5. 计算奖励 (多目标优化函数)
        reward = self.reward_function.calculate_reward(
            queue_info, cabinet_info, self.performance_stats
        )
        
        # 6. 获取新的观测状态
        observation = self.state_manager.get_observation()
        
        # 7. 检查终止条件
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_steps
        
        # 8. 准备返回信息
        info = {
            'step': self.current_step,
            'queue_lengths': queue_info['queue_lengths'],
            'waiting_times': queue_info['waiting_times'],
            'throughput': queue_info['throughput'],
            'cabinet_occupancy': cabinet_info['occupancy_rate'],
            'system_utilization': self._calculate_system_utilization(),
            'stability_condition': self._check_stability_condition(),
            'performance_stats': self.performance_stats.copy()
        }
        
        return observation, reward, terminated, truncated, info
    
    def _process_action(self, action: Dict):
        """
        处理智能体动作
        
        将动作转换为系统参数调整：
        1. transfer_decisions -> 层间转移概率调整
        2. service_priorities -> 服务率权重调整  
        3. arrival_weights -> 到达率分配调整
        """
        # 归一化动作参数
        service_priorities = self.math_utils.normalize_vector(action['service_priorities'])
        arrival_weights = self.math_utils.normalize_vector(action['arrival_weights'])
        
        # 更新队列动力学参数
        self.queue_dynamics.update_transfer_probabilities(action['transfer_decisions'])
        self.queue_dynamics.update_service_priorities(service_priorities)
        self.queue_dynamics.update_arrival_weights(arrival_weights)
        
        # 更新状态管理器
        self.state_manager.update_action_history(action)
    
    def _update_performance_stats(self, queue_info: Dict, cabinet_info: Dict):
        """
        更新性能统计信息
        
        跟踪关键性能指标用于奖励计算和分析
        """
        self.performance_stats['total_throughput'] += queue_info['throughput']
        self.performance_stats['total_waiting_time'] += sum(queue_info['waiting_times'])
        
        # 更新层利用率
        for i, length in enumerate(queue_info['queue_lengths']):
            capacity = self.config.layer_capacities[i]
            utilization = length / capacity if capacity > 0 else 0
            self.performance_stats['layer_utilizations'][i] += utilization
        
        self.performance_stats['successful_transfers'] += queue_info.get('transfers', 0)
        self.performance_stats['blocked_arrivals'] += queue_info.get('blocked', 0)
    
    def _calculate_system_utilization(self) -> float:
        """
        计算系统整体利用率
        
        基于01理论的利用率定义：
        Utilization = (1/2)(Σni/N_total + Σcj/C_total)
        """
        queue_lengths = self.queue_dynamics.get_queue_lengths()
        total_queue = sum(queue_lengths)
        total_capacity = sum(self.config.layer_capacities)
        
        cabinet_occupancy = self.delivery_cabinet.get_occupancy_rate()
        
        queue_util = total_queue / total_capacity if total_capacity > 0 else 0
        system_util = (queue_util + cabinet_occupancy) / 2
        
        return min(system_util, 1.0)
    
    def _check_stability_condition(self) -> Dict[str, float]:
        """
        检查01理论中的稳定性条件
        
        稳定性条件：ρi = λi^eff/(μi·ci) < 1, ∀i
        
        Returns:
            各层的负载系数ρi
        """
        return self.queue_dynamics.get_load_factors()
    
    def _check_termination(self) -> bool:
        """
        检查环境终止条件
        
        终止条件:
        1. 系统不稳定 (任一层ρi >= 1)
        2. 外卖柜故障
        3. 性能极度恶化
        """
        # 检查稳定性
        load_factors = self._check_stability_condition()
        if any(rho >= 1.0 for rho in load_factors.values()):
            return True
        
        # 检查外卖柜状态
        if self.delivery_cabinet.is_failed():
            return True
        
        # 检查性能恶化
        if self._is_performance_degraded():
            return True
        
        return False
    
    def _is_performance_degraded(self) -> bool:
        """
        检查性能是否严重恶化
        
        基于以下指标判断：
        1. 平均等待时间过长
        2. 吞吐量过低
        3. 阻塞率过高
        """
        if self.current_step < 100:  # 给系统一些预热时间
            return False
        
        avg_waiting_time = self.performance_stats['total_waiting_time'] / self.current_step
        avg_throughput = self.performance_stats['total_throughput'] / self.current_step
        block_rate = self.performance_stats['blocked_arrivals'] / self.current_step
        
        # 性能阈值 (可以根据需要调整)
        MAX_WAITING_TIME = 50  # 最大平均等待时间
        MIN_THROUGHPUT = 0.1   # 最小平均吞吐量
        MAX_BLOCK_RATE = 0.5   # 最大阻塞率
        
        return (avg_waiting_time > MAX_WAITING_TIME or 
                avg_throughput < MIN_THROUGHPUT or 
                block_rate > MAX_BLOCK_RATE)
    
    def render(self):
        """
        渲染环境状态
        
        显示：
        1. 5层队列状态
        2. 外卖柜占用情况  
        3. 性能指标
        """
        if self.render_mode == "human":
            self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()
    
    def _render_human(self):
        """
        控制台文本渲染
        """
        print(f"\n=== 垂直分层队列环境 Step {self.current_step} ===")
        
        # 显示队列状态
        queue_lengths = self.queue_dynamics.get_queue_lengths()
        waiting_times = self.queue_dynamics.get_waiting_times()
        load_factors = self._check_stability_condition()
        
        print("队列状态:")
        for i in range(self.config.num_layers):
            height = self.config.layer_heights[i]
            capacity = self.config.layer_capacities[i]
            length = queue_lengths[i]
            wait_time = waiting_times[i]
            rho = load_factors.get(f'layer_{i}', 0)
            
            print(f"  L{i+1}({height}m): {length:2d}/{capacity} UAVs, "
                  f"Wait={wait_time:5.1f}, ρ={rho:.3f}")
        
        # 显示外卖柜状态
        occupancy = self.delivery_cabinet.get_occupancy_rate()
        print(f"外卖柜占用率: {occupancy:.1%}")
        
        # 显示性能指标
        system_util = self._calculate_system_utilization()
        print(f"系统利用率: {system_util:.1%}")
        
        avg_throughput = (self.performance_stats['total_throughput'] / 
                         max(self.current_step, 1))
        print(f"平均吞吐量: {avg_throughput:.3f} 订单/步")
    
    def _render_rgb_array(self):
        """
        图像渲染 (简单实现)
        """
        # 这里可以实现更复杂的可视化
        # 目前返回简单的状态表示
        return np.zeros((400, 600, 3), dtype=np.uint8)
    
    def close(self):
        """
        关闭环境，清理资源
        """
        pass
    
    def get_theoretical_performance(self) -> Dict:
        """
        获取基于01理论的性能预测
        
        Returns:
            理论性能指标字典
        """
        return {
            'theoretical_waiting_time_range': (15, 25),  # 01理论预测
            'theoretical_throughput_range': (0.8, 1.2),  # 01理论预测
            'theoretical_utilization_range': (0.75, 0.85),  # 01理论预测
            'theoretical_efficiency_improvement': (0.4, 0.6)  # 相比传统方法
        }


# 测试环境类
if __name__ == "__main__":
    # 创建环境实例
    env = VerticalQueueEnv()
    
    # 简单测试
    print("环境创建成功!")
    print(f"观测空间: {env.observation_space}")
    print(f"动作空间: {env.action_space}")
    
    # 重置环境
    obs, info = env.reset()
    print(f"初始观测维度: {obs.shape}")
    print(f"初始信息: {info}")
    
    # 随机动作测试
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"\nStep {step + 1}:")
        print(f"  Reward: {reward:.3f}")
        print(f"  Terminated: {terminated}")
        print(f"  队列长度: {info['queue_lengths']}")
        
        if terminated or truncated:
            break
    
    env.close()
    print("\n环境测试完成!")