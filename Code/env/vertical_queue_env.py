"""
Vertical Stratified Queuing Environment

Complete implementation based on theoretical framework:
- Vertical layered queue system V = (L, C, T, S)
- 5-layer inverted pyramid capacity structure
- Inter-layer transfer dynamics
- Multi-objective optimization objective function
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
    Vertical Layered Queue Environment

    Implements core concepts from theory:
    1. Vertical layered queue system V = (L, C, T, S)
    2. Inverted pyramid capacity structure C = {8, 6, 4, 3, 2}
    3. Layered transfer dynamics T(li, li-1 | Q(t))
    4. Multi-objective optimization J(π) = w1·Throughput - w2·W̄ - w3·Var(W) + w4·Fairness
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, config: Optional[VerticalQueueConfig] = None, render_mode: str = None,
                 reward_weights: Optional['RewardWeights'] = None, **kwargs):
        super().__init__()

        # Configuration parameters
        self.config = config or VerticalQueueConfig()
        self.render_mode = render_mode

        # Core component initialization
        self.state_manager = StateManager(self.config)
        self.queue_dynamics = QueueDynamics(self.config)
        self.delivery_cabinet = DeliveryCabinet(self.config)
        self.reward_function = RewardFunction(self.config, weights=reward_weights)
        self.math_utils = MathUtils()

        # Environment state
        self.current_step = 0
        self.max_steps = self.config.max_episode_steps

        # Performance statistics
        self.performance_stats = {
            'total_throughput': 0,
            'total_waiting_time': 0,
            'layer_utilizations': [0] * self.config.num_layers,
            'successful_transfers': 0,
            'blocked_arrivals': 0
        }

        # Define action and observation spaces
        self._setup_spaces()

        # Initialize state
        self.reset()
    
    def _setup_spaces(self):
        """
        Setup action space and observation space

        Based on theoretical design:
        - Observation space: 128-dim state vector (6 dimension segments)
        - Action space: Hybrid action space (continuous movement + discrete queue decisions)
        """
        # Observation space: 128-dim state vector
        # Designed according to 6 dimension segments in state_manager.py
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(128,),
            dtype=np.float32
        )

        # Action space: Hybrid action space
        # 1. Inter-layer transfer decisions (5 layers, each can choose to transfer to next layer or stay)
        # 2. Service scheduling strategy (priority adjustment)
        # 3. Arrival rate control (dynamically adjust arrival weights for each layer)
        self.action_space = spaces.Dict({
            # Inter-layer transfer decisions (5 layers, binary choice: transfer/no transfer)
            'transfer_decisions': spaces.MultiBinary(self.config.num_layers),

            # Service priority weights (5 layers continuous weights, normalized before use)
            'service_priorities': spaces.Box(
                low=0.0, high=1.0,
                shape=(self.config.num_layers,),
                dtype=np.float32
            ),

            # Arrival allocation weights (5 layers arrival weight adjustment)
            'arrival_weights': spaces.Box(
                low=0.0, high=1.0,
                shape=(self.config.num_layers,),
                dtype=np.float32
            )
        })

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset environment to initial state

        Returns:
            observation: 128-dim initial state vector
            info: Environment information dictionary
        """
        super().reset(seed=seed)

        # Reset time step
        self.current_step = 0

        # Reset all components
        self.state_manager.reset()
        self.queue_dynamics.reset()
        self.delivery_cabinet.reset()

        # Reset performance statistics
        self.performance_stats = {
            'total_throughput': 0,
            'total_waiting_time': 0,
            'layer_utilizations': [0] * self.config.num_layers,
            'successful_transfers': 0,
            'blocked_arrivals': 0
        }

        # Get initial observation
        observation = self.state_manager.get_observation()

        # Environment information
        info = {
            'step': self.current_step,
            'queue_lengths': self.queue_dynamics.get_queue_lengths(),
            'cabinet_occupancy': self.delivery_cabinet.get_occupancy_rate(),
            'system_utilization': self._calculate_system_utilization()
        }

        return observation, info
    
    def step(self, action: Dict):
        """
        Environment step function

        Implements system dynamics from theory:
        1. Execute action (inter-layer transfer, service scheduling)
        2. Update queue state (arrival, service, transfer)
        3. Calculate reward (multi-objective optimization function)
        4. Update delivery cabinet state
        5. Check termination condition

        Args:
            action: Action dictionary containing transfer_decisions, service_priorities, arrival_weights

        Returns:
            observation: New 128-dim state vector
            reward: Scalar reward value
            terminated: Whether termination condition is reached
            truncated: Whether time limit is reached
            info: Environment information dictionary
        """
        # Increment time step
        self.current_step += 1

        # 1. Process action
        self._process_action(action)

        # 2. Update queue dynamics (state transition based on theory)
        queue_info = self.queue_dynamics.step(action)

        # 3. Update delivery cabinet state
        cabinet_info = self.delivery_cabinet.step(queue_info['service_requests'])

        # 4. Update performance statistics
        self._update_performance_stats(queue_info, cabinet_info)

        # 5. Calculate reward (multi-objective optimization function)
        reward = self.reward_function.calculate_reward(
            queue_info, cabinet_info, self.performance_stats
        )

        # 6. Get new observation state
        observation = self.state_manager.get_observation()

        # 7. Check termination condition
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_steps

        # 8. Prepare return information
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
        Process agent action

        Convert action to system parameter adjustments:
        1. transfer_decisions -> Inter-layer transfer probability adjustment
        2. service_priorities -> Service rate weight adjustment
        3. arrival_weights -> Arrival rate allocation adjustment
        """
        # Normalize action parameters
        service_priorities = self.math_utils.normalize_vector(action['service_priorities'])
        arrival_weights = self.math_utils.normalize_vector(action['arrival_weights'])

        # Update queue dynamics parameters
        self.queue_dynamics.update_transfer_probabilities(action['transfer_decisions'])
        self.queue_dynamics.update_service_priorities(service_priorities)
        self.queue_dynamics.update_arrival_weights(arrival_weights)

        # Update state manager
        self.state_manager.update_action_history(action)

    def _update_performance_stats(self, queue_info: Dict, cabinet_info: Dict):
        """
        Update performance statistics

        Track key performance indicators for reward calculation and analysis
        """
        self.performance_stats['total_throughput'] += queue_info['throughput']
        self.performance_stats['total_waiting_time'] += sum(queue_info['waiting_times'])

        # Update layer utilization
        for i, length in enumerate(queue_info['queue_lengths']):
            capacity = self.config.layer_capacities[i]
            utilization = length / capacity if capacity > 0 else 0
            self.performance_stats['layer_utilizations'][i] += utilization

        self.performance_stats['successful_transfers'] += queue_info.get('transfers', 0)
        self.performance_stats['blocked_arrivals'] += queue_info.get('blocked', 0)

    def _calculate_system_utilization(self) -> float:
        """
        Calculate overall system utilization

        Based on utilization definition from theory:
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
        Check stability condition from theory

        Stability condition: ρi = λi^eff/(μi·ci) < 1, ∀i

        Returns:
            Load factor ρi for each layer
        """
        return self.queue_dynamics.get_load_factors()

    def _check_termination(self) -> bool:
        """
        Check environment termination condition

        Termination conditions:
        1. System unstable (any layer ρi >= 1)
        2. Delivery cabinet failure
        3. Severe performance degradation
        """
        # Check stability
        load_factors = self._check_stability_condition()
        if any(rho >= 1.0 for rho in load_factors.values()):
            return True

        # Check delivery cabinet status
        if self.delivery_cabinet.is_failed():
            return True

        # Check performance degradation
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