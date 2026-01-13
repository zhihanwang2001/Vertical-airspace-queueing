"""
DRL优化垂直分层队列环境 - 修复版本
DRL-Optimized Vertical Stratified Queuing Environment - Fixed Version

修复的关键问题：
1. 奖励函数数学逻辑错误
2. 能耗计算不合理
3. 负载率计算错误
4. 紧急转移机制优化
5. 观测空间信息增强
"""

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

class DRLOptimizedQueueEnvFixed(gym.Env):
    """
    DRL优化垂直分层队列环境 - 修复版本
    
    核心修复：
    1. 数学上更合理的奖励函数
    2. 正确的负载率和能耗计算
    3. 改进的观测空间设计
    4. 更稳定的紧急转移机制
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 4}
    
    def __init__(self, render_mode: str = None, max_episode_steps: int = 10000):
        super().__init__()

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps  # 添加可配置的最大步数
        
        # ==== 系统参数 ====
        self.n_layers = 5
        self.heights = np.array([100, 80, 60, 40, 20], dtype=np.float32)
        self.capacities = np.array([8, 6, 4, 3, 2], dtype=np.int32)
        
        # 基础参数
        self.base_arrival_rate = 0.3
        self.arrival_weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1], dtype=np.float32)
        self.base_service_rates = np.array([1.2, 1.0, 0.8, 0.6, 0.4], dtype=np.float32)
        
        # ==== 动作空间 ====
        self.action_space = spaces.Dict({
            'service_intensities': spaces.Box(
                low=0.1, high=2.0, shape=(self.n_layers,), dtype=np.float32
            ),
            'arrival_multiplier': spaces.Box(
                low=0.5, high=5.0, shape=(1,), dtype=np.float32
            ),
            'emergency_transfers': spaces.MultiBinary(self.n_layers)
        })
        
        # ==== 增强观测空间 ====
        self.observation_space = spaces.Dict({
            'queue_lengths': spaces.Box(
                low=0, high=max(self.capacities), shape=(self.n_layers,), dtype=np.float32
            ),
            'utilization_rates': spaces.Box(
                low=0.0, high=1.0, shape=(self.n_layers,), dtype=np.float32
            ),
            'queue_changes': spaces.Box(
                low=-1.0, high=1.0, shape=(self.n_layers,), dtype=np.float32
            ),
            'load_rates': spaces.Box(  # 新增：实际负载率
                low=0.0, high=5.0, shape=(self.n_layers,), dtype=np.float32
            ),
            'service_rates': spaces.Box(  # 新增：当前服务率
                low=0.0, high=10.0, shape=(self.n_layers,), dtype=np.float32
            ),
            'prev_reward': spaces.Box(
                low=-100.0, high=100.0, shape=(1,), dtype=np.float32
            ),
            'system_metrics': spaces.Box(  # 新增：系统级指标
                low=0.0, high=10.0, shape=(3,), dtype=np.float32  # [总负载, 总利用率, 稳定性指标]
            )
        })
        
        self.reset()
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """重置环境状态"""
        if seed is not None:
            np.random.seed(seed)
        
        # 初始化状态
        self.queue_lengths = np.zeros(self.n_layers, dtype=np.float32)
        self.prev_queue_lengths = np.zeros(self.n_layers, dtype=np.float32)
        
        # 系统指标
        self.step_count = 0
        self.total_served = 0
        self.total_arrived = 0
        self.prev_reward = 0.0
        
        # 服务统计
        self.service_counts = np.zeros(self.n_layers, dtype=np.float32)
        self.transfer_counts = np.zeros(self.n_layers, dtype=np.float32)  # 新增：转移统计
        
        # 当前动作状态
        self.current_service_rates = self.base_service_rates.copy()
        self.current_arrival_rate = self.base_arrival_rate
        
        # 稳定性监控
        self.stability_history = []  # 记录最近的系统状态用于稳定性计算
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
        
    def step(self, action: Dict[str, Union[np.ndarray, float]]):
        """执行一步动作"""
        self.step_count += 1
        self.prev_queue_lengths = self.queue_lengths.copy()
        
        # ==== 动作执行 ====
        service_intensities = np.clip(action['service_intensities'], 0.1, 2.0)
        self.current_service_rates = self.base_service_rates * service_intensities
        
        arrival_multiplier = np.clip(action['arrival_multiplier'][0], 0.5, 5.0)
        self.current_arrival_rate = self.base_arrival_rate * arrival_multiplier
        
        emergency_transfers = action['emergency_transfers'].astype(bool)
        
        # ==== 系统动力学模拟 ====
        
        # 1. 到达过程
        total_arrivals = np.random.poisson(self.current_arrival_rate)
        self.total_arrived += total_arrivals
        
        if total_arrivals > 0:
            layer_arrivals = np.random.multinomial(total_arrivals, self.arrival_weights)
            # 考虑容量限制的到达
            for i in range(self.n_layers):
                available_capacity = max(0, self.capacities[i] - self.queue_lengths[i])
                actual_arrivals = min(layer_arrivals[i], available_capacity)
                self.queue_lengths[i] += actual_arrivals
        
        # 2. 服务过程
        self.service_counts.fill(0)
        for i in range(self.n_layers):
            if self.queue_lengths[i] > 0:
                # 修正：服务能力基于当前服务率，不乘以容量
                max_service = min(
                    np.random.poisson(self.current_service_rates[i]) + 1,  # +1避免服务为0
                    int(self.queue_lengths[i])
                )
                self.queue_lengths[i] -= max_service
                self.service_counts[i] = max_service
                self.total_served += max_service
        
        # 3. 改进的紧急转移机制
        self.transfer_counts.fill(0)
        for i in range(self.n_layers - 1):  # 最底层无法转移
            if emergency_transfers[i] and self.queue_lengths[i] > 0:
                # 修正：转移率基于队列压力和下层容量
                queue_pressure = self.queue_lengths[i] / self.capacities[i]
                target_available = max(0, self.capacities[i+1] - self.queue_lengths[i+1])
                
                if target_available > 0:
                    # 转移数量基于压力和可用容量
                    max_transfer = min(
                        int(self.queue_lengths[i] * min(0.8, queue_pressure)),
                        target_available
                    )
                    
                    if max_transfer > 0:
                        self.queue_lengths[i] -= max_transfer
                        self.queue_lengths[i+1] += max_transfer
                        self.transfer_counts[i] = max_transfer
        
        # ==== 修正的奖励函数 ====
        reward = self._calculate_fixed_reward(action)
        self.prev_reward = reward
        
        # 更新稳定性历史
        self._update_stability_history()
        
        # 终止条件
        terminated = self._check_termination()
        truncated = self.step_count >= self.max_episode_steps
        
        obs = self._get_observation()
        info = self._get_info(action)  # 传入action参数

        return obs, reward, terminated, truncated, info
    
    def _calculate_fixed_reward(self, action: Dict) -> float:
        """
        修正的奖励函数
        解决数学逻辑错误和计算不合理问题
        """
        
        # 1. 吞吐量奖励 (保持不变，这部分是正确的)
        R_throughput = 10.0 * np.sum(self.service_counts)
        
        # 2. 修正的负载均衡奖励
        utilization_rates = self.queue_lengths / self.capacities
        
        # 修正：使用更稳定的负载均衡度量
        if np.sum(utilization_rates) > 1e-6:
            # 使用基尼系数的倒数作为均衡度度量
            sorted_util = np.sort(utilization_rates)
            n = len(sorted_util)
            cumsum = np.cumsum(sorted_util)
            # 基尼系数: G = (2*sum(i*y_i))/(n*sum(y_i)) - (n+1)/n
            if np.sum(sorted_util) > 1e-6:
                gini = (2 * np.sum((np.arange(n) + 1) * sorted_util)) / (n * np.sum(sorted_util)) - (n + 1) / n
                R_balance = 5.0 * (1.0 - gini)  # 基尼系数越小，均衡度越高
            else:
                R_balance = 5.0
        else:
            R_balance = 5.0  # 空队列时认为是完美均衡
        
        # 3. 修正的效率奖励
        service_total = np.sum(self.service_counts)
        
        # 修正：能耗计算应该基于使用的资源，而不是容量
        base_energy = 1.0  # 基础能耗
        service_energy = np.sum(action['service_intensities'])  # 服务强度总和
        arrival_energy = action['arrival_multiplier'][0] * 0.5  # 到达控制能耗
        transfer_energy = np.sum(action['emergency_transfers']) * 0.2  # 转移能耗
        
        total_energy = base_energy + service_energy + arrival_energy + transfer_energy
        
        if total_energy > 1e-6:
            R_efficiency = 3.0 * service_total / total_energy
        else:
            R_efficiency = 0.0
        
        # 4. 拥堵惩罚 (保持不变，这部分是正确的)
        congestion_levels = np.maximum(0, (self.queue_lengths - 0.8 * self.capacities) / self.capacities)
        P_congestion = -20.0 * np.sum(congestion_levels)
        
        # 5. 修正的系统不稳定惩罚
        # 修正：正确计算负载率
        current_arrivals = self.current_arrival_rate * self.arrival_weights
        load_rates = current_arrivals / np.maximum(self.current_service_rates, 1e-6)  # 移除容量乘积
        
        instability_levels = np.maximum(0, load_rates - 0.95)
        P_instability = -15.0 * np.sum(instability_levels)
        
        # 6. 新增：转移效率奖励
        transfer_benefit = 0.0
        for i in range(self.n_layers - 1):
            if self.transfer_counts[i] > 0:
                # 如果转移缓解了上层压力，给予奖励
                upper_pressure = self.queue_lengths[i] / self.capacities[i]
                lower_util = self.queue_lengths[i+1] / self.capacities[i+1]
                if upper_pressure > lower_util:  # 转移是有益的
                    transfer_benefit += 2.0 * self.transfer_counts[i]
        
        # 7. 新增：系统稳定性奖励
        stability_bonus = self._calculate_stability_bonus()
        
        # 总奖励
        total_reward = (R_throughput + R_balance + R_efficiency + transfer_benefit + 
                       stability_bonus + P_congestion + P_instability)
        
        return float(total_reward)
    
    def _calculate_stability_bonus(self) -> float:
        """计算系统稳定性奖励"""
        if len(self.stability_history) < 5:
            return 0.0

        # 计算最近几步的队列长度变化
        recent_changes = []
        for i in range(len(self.stability_history) - 1):
            change = np.abs(self.stability_history[i+1] - self.stability_history[i])
            recent_changes.append(np.sum(change))

        if len(recent_changes) > 0:
            avg_change = np.mean(recent_changes)
            # 变化越小，稳定性奖励越高
            stability_bonus = 2.0 * np.exp(-avg_change / 2.0)
            return stability_bonus

        return 0.0

    def _get_reward_components(self, action: Dict = None) -> Dict[str, float]:
        """
        获取奖励组件分解（用于帕累托分析）
        与_calculate_fixed_reward完全一致的计算逻辑
        """
        # 1. 吞吐量奖励
        R_throughput = 10.0 * np.sum(self.service_counts)

        # 2. 负载均衡奖励
        utilization_rates = self.queue_lengths / self.capacities
        if np.sum(utilization_rates) > 1e-6:
            sorted_util = np.sort(utilization_rates)
            n = len(sorted_util)
            if np.sum(sorted_util) > 1e-6:
                gini = (2 * np.sum((np.arange(n) + 1) * sorted_util)) / (n * np.sum(sorted_util)) - (n + 1) / n
                R_balance = 5.0 * (1.0 - gini)
            else:
                R_balance = 5.0
        else:
            R_balance = 5.0

        # 3. 效率奖励
        service_total = np.sum(self.service_counts)

        if action is not None:
            # 使用原始action计算能耗（与_calculate_fixed_reward完全一致）
            base_energy = 1.0
            service_energy = np.sum(action['service_intensities'])
            arrival_energy = action['arrival_multiplier'][0] * 0.5
            transfer_energy = np.sum(action['emergency_transfers']) * 0.2
            total_energy = base_energy + service_energy + arrival_energy + transfer_energy
        else:
            # 备用方案：反推action参数
            base_energy = 1.0
            service_energy = np.sum(self.current_service_rates / self.base_service_rates)
            arrival_energy = self.current_arrival_rate / self.base_arrival_rate * 0.5
            transfer_energy = np.sum(self.transfer_counts) * 0.2
            total_energy = base_energy + service_energy + arrival_energy + transfer_energy

        if total_energy > 1e-6:
            R_efficiency = 3.0 * service_total / total_energy
        else:
            R_efficiency = 0.0

        # 4. 拥堵惩罚
        congestion_levels = np.maximum(0, (self.queue_lengths - 0.8 * self.capacities) / self.capacities)
        P_congestion = -20.0 * np.sum(congestion_levels)

        # 5. 不稳定惩罚
        current_arrivals = self.current_arrival_rate * self.arrival_weights
        load_rates = current_arrivals / np.maximum(self.current_service_rates, 1e-6)
        instability_levels = np.maximum(0, load_rates - 0.95)
        P_instability = -15.0 * np.sum(instability_levels)

        # 6. 转移效益
        transfer_benefit = 0.0
        for i in range(self.n_layers - 1):
            if self.transfer_counts[i] > 0:
                upper_pressure = self.queue_lengths[i] / self.capacities[i]
                lower_util = self.queue_lengths[i+1] / self.capacities[i+1]
                if upper_pressure > lower_util:
                    transfer_benefit += 2.0 * self.transfer_counts[i]

        # 7. 稳定性奖励
        stability_bonus = self._calculate_stability_bonus()

        return {
            'throughput': float(R_throughput),
            'balance': float(R_balance),
            'efficiency': float(R_efficiency),
            'transfer': float(transfer_benefit),
            'stability': float(stability_bonus),
            'congestion': float(P_congestion),
            'instability': float(P_instability)
        }

    def _update_stability_history(self):
        """更新稳定性历史"""
        self.stability_history.append(self.queue_lengths.copy())
        # 只保留最近10步的历史
        if len(self.stability_history) > 10:
            self.stability_history.pop(0)
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """构建增强状态观测"""
        
        # 队列状态变化率
        queue_changes = np.zeros(self.n_layers, dtype=np.float32)
        if self.step_count > 0:
            queue_changes = (self.queue_lengths - self.prev_queue_lengths) / np.maximum(self.capacities, 1)
        
        # 利用率
        utilization_rates = self.queue_lengths / self.capacities
        
        # 实际负载率 (修正计算)
        current_arrivals = self.current_arrival_rate * self.arrival_weights
        load_rates = current_arrivals / np.maximum(self.current_service_rates, 1e-6)
        
        # 系统级指标
        total_load = np.sum(self.queue_lengths) / np.sum(self.capacities)
        avg_utilization = np.mean(utilization_rates)
        stability_metric = self._calculate_stability_bonus() / 2.0  # 归一化到[0,1]
        
        system_metrics = np.array([total_load, avg_utilization, stability_metric], dtype=np.float32)
        
        return {
            'queue_lengths': self.queue_lengths.astype(np.float32),
            'utilization_rates': utilization_rates.astype(np.float32),
            'queue_changes': queue_changes.astype(np.float32),
            'load_rates': np.clip(load_rates, 0, 5).astype(np.float32),
            'service_rates': self.current_service_rates.astype(np.float32),
            'prev_reward': np.array([self.prev_reward], dtype=np.float32),
            'system_metrics': system_metrics
        }
    
    def _check_termination(self) -> bool:
        """检查终止条件"""
        # 队列溢出检测
        if np.any(self.queue_lengths >= self.capacities * 1.1):  # 允许10%超载
            return True
        
        # 系统崩溃检测
        current_arrivals = self.current_arrival_rate * self.arrival_weights
        load_rates = current_arrivals / np.maximum(self.current_service_rates, 1e-6)
        if np.any(load_rates > 2.0):  # 负载率过高
            return True
            
        return False
    
    def _get_info(self, action: Dict = None) -> Dict:
        """获取环境信息"""
        current_arrivals = self.current_arrival_rate * self.arrival_weights
        load_rates = current_arrivals / np.maximum(self.current_service_rates, 1e-6)

        return {
            'step_count': self.step_count,
            'total_served': self.total_served,
            'total_arrived': self.total_arrived,
            'throughput': self.total_served / max(1, self.step_count),
            'utilization_rates': (self.queue_lengths / self.capacities).tolist(),
            'load_rates': load_rates.tolist(),
            'service_counts': self.service_counts.tolist(),
            'transfer_counts': self.transfer_counts.tolist(),
            'current_arrival_rate': self.current_arrival_rate,
            'current_service_rates': self.current_service_rates.tolist(),
            'stability_score': self._calculate_stability_bonus(),
            # 新增：奖励组件分解（用于帕累托分析）
            'reward_components': self._get_reward_components(action)
        }
    
    def render(self, mode: str = "human"):
        """渲染环境状态"""
        if mode == "human":
            print(f"\n=== Step {self.step_count} ===")
            print(f"队列状态: {self.queue_lengths}")
            print(f"容量: {self.capacities}")
            print(f"利用率: {self.queue_lengths/self.capacities:.3f}")
            print(f"服务率: {self.current_service_rates:.3f}")
            print(f"当前奖励: {self.prev_reward:.3f}")
            print(f"稳定性: {self._calculate_stability_bonus():.3f}")


def create_fixed_drl_environment():
    """创建修复版DRL环境"""
    return DRLOptimizedQueueEnvFixed()


if __name__ == "__main__":
    # 测试修复版环境
    env = create_fixed_drl_environment()
    
    print("🛠️ DRL优化环境修复版测试")
    print("=" * 60)
    print(f"动作空间: {env.action_space}")
    print(f"观测空间键: {list(env.observation_space.spaces.keys())}")
    
    # 运行测试
    obs, info = env.reset()
    print(f"\n初始观测维度:")
    for key, value in obs.items():
        print(f"  {key}: {value.shape}")
    
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        
        print(f"\nStep {step+1}:")
        print(f"动作: 服务强度{action['service_intensities'][:3]:.2f}..., 到达倍数{action['arrival_multiplier'][0]:.2f}")
        print(f"奖励: {reward:.3f}")
        print(f"队列: {obs['queue_lengths']}")
        print(f"负载率: {obs['load_rates']:.2f}")
        print(f"稳定性: {info['stability_score']:.3f}")
        
        if term or trunc:
            break
    
    env.close()
    print("\n✅ 修复版环境测试完成")