"""
DRL-Optimized Vertical Stratified Queuing Environment - Fixed Version

Key issues fixed:
1. Mathematical logic errors in reward function
2. Unreasonable energy consumption calculation
3. Load rate calculation errors
4. Emergency transfer mechanism optimization
5. Enhanced observation space information
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
    DRL-Optimized Vertical Stratified Queuing Environment - Fixed Version

    Core fixes:
    1. Mathematically more reasonable reward function
    2. Correct load rate and energy consumption calculation
    3. Improved observation space design
    4. More stable emergency transfer mechanism
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, render_mode: str = None, max_episode_steps: int = 10000):
        super().__init__()

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps  # Add configurable maximum steps

        # ==== System parameters ====
        self.n_layers = 5
        self.heights = np.array([100, 80, 60, 40, 20], dtype=np.float32)
        self.capacities = np.array([8, 6, 4, 3, 2], dtype=np.int32)

        # Base parameters
        self.base_arrival_rate = 0.3
        self.arrival_weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1], dtype=np.float32)
        self.base_service_rates = np.array([1.2, 1.0, 0.8, 0.6, 0.4], dtype=np.float32)

        # ==== Action space ====
        self.action_space = spaces.Dict({
            'service_intensities': spaces.Box(
                low=0.1, high=2.0, shape=(self.n_layers,), dtype=np.float32
            ),
            'arrival_multiplier': spaces.Box(
                low=0.5, high=5.0, shape=(1,), dtype=np.float32
            ),
            'emergency_transfers': spaces.MultiBinary(self.n_layers)
        })

        # ==== Enhanced observation space ====
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
            'load_rates': spaces.Box(  # New: actual load rates
                low=0.0, high=5.0, shape=(self.n_layers,), dtype=np.float32
            ),
            'service_rates': spaces.Box(  # New: current service rates
                low=0.0, high=10.0, shape=(self.n_layers,), dtype=np.float32
            ),
            'prev_reward': spaces.Box(
                low=-100.0, high=100.0, shape=(1,), dtype=np.float32
            ),
            'system_metrics': spaces.Box(  # New: system-level metrics
                low=0.0, high=10.0, shape=(3,), dtype=np.float32  # [total load, total utilization, stability indicator]
            )
        })
        
        self.reset()
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment state"""
        if seed is not None:
            np.random.seed(seed)

        # Initialize state
        self.queue_lengths = np.zeros(self.n_layers, dtype=np.float32)
        self.prev_queue_lengths = np.zeros(self.n_layers, dtype=np.float32)

        # System metrics
        self.step_count = 0
        self.total_served = 0
        self.total_arrived = 0
        self.prev_reward = 0.0

        # Service statistics
        self.service_counts = np.zeros(self.n_layers, dtype=np.float32)
        self.transfer_counts = np.zeros(self.n_layers, dtype=np.float32)  # New: transfer statistics

        # Current action state
        self.current_service_rates = self.base_service_rates.copy()
        self.current_arrival_rate = self.base_arrival_rate

        # Stability monitoring
        self.stability_history = []  # Record recent system states for stability calculation

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: Dict[str, Union[np.ndarray, float]]):
        """Execute one step action"""
        self.step_count += 1
        self.prev_queue_lengths = self.queue_lengths.copy()

        # ==== Action execution ====
        service_intensities = np.clip(action['service_intensities'], 0.1, 2.0)
        self.current_service_rates = self.base_service_rates * service_intensities

        arrival_multiplier = np.clip(action['arrival_multiplier'][0], 0.5, 5.0)
        self.current_arrival_rate = self.base_arrival_rate * arrival_multiplier

        emergency_transfers = action['emergency_transfers'].astype(bool)

        # ==== System dynamics simulation ====

        # 1. Arrival process
        total_arrivals = np.random.poisson(self.current_arrival_rate)
        self.total_arrived += total_arrivals

        if total_arrivals > 0:
            layer_arrivals = np.random.multinomial(total_arrivals, self.arrival_weights)
            # Consider capacity-limited arrivals
            for i in range(self.n_layers):
                available_capacity = max(0, self.capacities[i] - self.queue_lengths[i])
                actual_arrivals = min(layer_arrivals[i], available_capacity)
                self.queue_lengths[i] += actual_arrivals

        # 2. Service process
        self.service_counts.fill(0)
        for i in range(self.n_layers):
            if self.queue_lengths[i] > 0:
                # Fix: service capacity based on current service rate, not multiplied by capacity
                max_service = min(
                    np.random.poisson(self.current_service_rates[i]) + 1,  # +1 to avoid zero service
                    int(self.queue_lengths[i])
                )
                self.queue_lengths[i] -= max_service
                self.service_counts[i] = max_service
                self.total_served += max_service

        # 3. Improved emergency transfer mechanism
        self.transfer_counts.fill(0)
        for i in range(self.n_layers - 1):  # Bottom layer cannot transfer
            if emergency_transfers[i] and self.queue_lengths[i] > 0:
                # Fix: transfer rate based on queue pressure and lower layer capacity
                queue_pressure = self.queue_lengths[i] / self.capacities[i]
                target_available = max(0, self.capacities[i+1] - self.queue_lengths[i+1])

                if target_available > 0:
                    # Transfer amount based on pressure and available capacity
                    max_transfer = min(
                        int(self.queue_lengths[i] * min(0.8, queue_pressure)),
                        target_available
                    )

                    if max_transfer > 0:
                        self.queue_lengths[i] -= max_transfer
                        self.queue_lengths[i+1] += max_transfer
                        self.transfer_counts[i] = max_transfer

        # ==== Fixed reward function ====
        reward = self._calculate_fixed_reward(action)
        self.prev_reward = reward

        # Update stability history
        self._update_stability_history()

        # Termination condition
        terminated = self._check_termination()
        truncated = self.step_count >= self.max_episode_steps

        obs = self._get_observation()
        info = self._get_info(action)  # Pass action parameter

        return obs, reward, terminated, truncated, info

    def _calculate_fixed_reward(self, action: Dict) -> float:
        """
        Fixed reward function
        Solves mathematical logic errors and unreasonable calculation issues
        """

        # 1. Throughput reward (unchanged, this part is correct)
        R_throughput = 10.0 * np.sum(self.service_counts)

        # 2. Fixed load balancing reward
        utilization_rates = self.queue_lengths / self.capacities

        # Fix: use more stable load balancing metric
        if np.sum(utilization_rates) > 1e-6:
            # Use inverse of Gini coefficient as balance metric
            sorted_util = np.sort(utilization_rates)
            n = len(sorted_util)
            cumsum = np.cumsum(sorted_util)
            # Gini coefficient: G = (2*sum(i*y_i))/(n*sum(y_i)) - (n+1)/n
            if np.sum(sorted_util) > 1e-6:
                gini = (2 * np.sum((np.arange(n) + 1) * sorted_util)) / (n * np.sum(sorted_util)) - (n + 1) / n
                R_balance = 5.0 * (1.0 - gini)  # Lower Gini coefficient means higher balance
            else:
                R_balance = 5.0
        else:
            R_balance = 5.0  # Empty queue is considered perfect balance

        # 3. Fixed efficiency reward
        service_total = np.sum(self.service_counts)

        # Fix: energy calculation should be based on resources used, not capacity
        base_energy = 1.0  # Base energy consumption
        service_energy = np.sum(action['service_intensities'])  # Total service intensity
        arrival_energy = action['arrival_multiplier'][0] * 0.5  # Arrival control energy
        transfer_energy = np.sum(action['emergency_transfers']) * 0.2  # Transfer energy

        total_energy = base_energy + service_energy + arrival_energy + transfer_energy

        if total_energy > 1e-6:
            R_efficiency = 3.0 * service_total / total_energy
        else:
            R_efficiency = 0.0

        # 4. Congestion penalty (unchanged, this part is correct)
        congestion_levels = np.maximum(0, (self.queue_lengths - 0.8 * self.capacities) / self.capacities)
        P_congestion = -20.0 * np.sum(congestion_levels)

        # 5. Fixed system instability penalty
        # Fix: correctly calculate load rate
        current_arrivals = self.current_arrival_rate * self.arrival_weights
        load_rates = current_arrivals / np.maximum(self.current_service_rates, 1e-6)  # Remove capacity product

        instability_levels = np.maximum(0, load_rates - 0.95)
        P_instability = -15.0 * np.sum(instability_levels)

        # 6. New: transfer efficiency reward
        transfer_benefit = 0.0
        for i in range(self.n_layers - 1):
            if self.transfer_counts[i] > 0:
                # If transfer relieves upper layer pressure, give reward
                upper_pressure = self.queue_lengths[i] / self.capacities[i]
                lower_util = self.queue_lengths[i+1] / self.capacities[i+1]
                if upper_pressure > lower_util:  # Transfer is beneficial
                    transfer_benefit += 2.0 * self.transfer_counts[i]

        # 7. New: system stability reward
        stability_bonus = self._calculate_stability_bonus()

        # Total reward
        total_reward = (R_throughput + R_balance + R_efficiency + transfer_benefit +
                       stability_bonus + P_congestion + P_instability)

        return float(total_reward)

    def _calculate_stability_bonus(self) -> float:
        """Calculate system stability reward"""
        if len(self.stability_history) < 5:
            return 0.0

        # Calculate queue length changes in recent steps
        recent_changes = []
        for i in range(len(self.stability_history) - 1):
            change = np.abs(self.stability_history[i+1] - self.stability_history[i])
            recent_changes.append(np.sum(change))

        if len(recent_changes) > 0:
            avg_change = np.mean(recent_changes)
            # Smaller change means higher stability reward
            stability_bonus = 2.0 * np.exp(-avg_change / 2.0)
            return stability_bonus

        return 0.0

    def _get_reward_components(self, action: Dict = None) -> Dict[str, float]:
        """
        Get reward component breakdown (for Pareto analysis)
        Calculation logic completely consistent with _calculate_fixed_reward
        """
        # 1. Throughput reward
        R_throughput = 10.0 * np.sum(self.service_counts)

        # 2. Load balancing reward
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

        # 3. Efficiency reward
        service_total = np.sum(self.service_counts)

        if action is not None:
            # Use original action to calculate energy (completely consistent with _calculate_fixed_reward)
            base_energy = 1.0
            service_energy = np.sum(action['service_intensities'])
            arrival_energy = action['arrival_multiplier'][0] * 0.5
            transfer_energy = np.sum(action['emergency_transfers']) * 0.2
            total_energy = base_energy + service_energy + arrival_energy + transfer_energy
        else:
            # Fallback: reverse engineer action parameters
            base_energy = 1.0
            service_energy = np.sum(self.current_service_rates / self.base_service_rates)
            arrival_energy = self.current_arrival_rate / self.base_arrival_rate * 0.5
            transfer_energy = np.sum(self.transfer_counts) * 0.2
            total_energy = base_energy + service_energy + arrival_energy + transfer_energy

        if total_energy > 1e-6:
            R_efficiency = 3.0 * service_total / total_energy
        else:
            R_efficiency = 0.0

        # 4. Congestion penalty
        congestion_levels = np.maximum(0, (self.queue_lengths - 0.8 * self.capacities) / self.capacities)
        P_congestion = -20.0 * np.sum(congestion_levels)

        # 5. Instability penalty
        current_arrivals = self.current_arrival_rate * self.arrival_weights
        load_rates = current_arrivals / np.maximum(self.current_service_rates, 1e-6)
        instability_levels = np.maximum(0, load_rates - 0.95)
        P_instability = -15.0 * np.sum(instability_levels)

        # 6. Transfer benefit
        transfer_benefit = 0.0
        for i in range(self.n_layers - 1):
            if self.transfer_counts[i] > 0:
                upper_pressure = self.queue_lengths[i] / self.capacities[i]
                lower_util = self.queue_lengths[i+1] / self.capacities[i+1]
                if upper_pressure > lower_util:
                    transfer_benefit += 2.0 * self.transfer_counts[i]

        # 7. Stability reward
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
        """Update stability history"""
        self.stability_history.append(self.queue_lengths.copy())
        # Keep only the last 10 steps of history
        if len(self.stability_history) > 10:
            self.stability_history.pop(0)

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Build enhanced state observation"""

        # Queue state change rate
        queue_changes = np.zeros(self.n_layers, dtype=np.float32)
        if self.step_count > 0:
            queue_changes = (self.queue_lengths - self.prev_queue_lengths) / np.maximum(self.capacities, 1)

        # Utilization rate
        utilization_rates = self.queue_lengths / self.capacities

        # Actual load rate (fixed calculation)
        current_arrivals = self.current_arrival_rate * self.arrival_weights
        load_rates = current_arrivals / np.maximum(self.current_service_rates, 1e-6)

        # System-level metrics
        total_load = np.sum(self.queue_lengths) / np.sum(self.capacities)
        avg_utilization = np.mean(utilization_rates)
        stability_metric = self._calculate_stability_bonus() / 2.0  # Normalize to [0,1]

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
        """Check termination condition"""
        # Queue overflow detection
        if np.any(self.queue_lengths >= self.capacities * 1.1):  # Allow 10% overload
            return True

        # System crash detection
        current_arrivals = self.current_arrival_rate * self.arrival_weights
        load_rates = current_arrivals / np.maximum(self.current_service_rates, 1e-6)
        if np.any(load_rates > 2.0):  # Load rate too high
            return True

        return False

    def _get_info(self, action: Dict = None) -> Dict:
        """Get environment information"""
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
            # New: reward component breakdown (for Pareto analysis)
            'reward_components': self._get_reward_components(action)
        }

    def render(self, mode: str = "human"):
        """Render environment state"""
        if mode == "human":
            print(f"\n=== Step {self.step_count} ===")
            print(f"Queue state: {self.queue_lengths}")
            print(f"Capacity: {self.capacities}")
            print(f"Utilization: {self.queue_lengths/self.capacities:.3f}")
            print(f"Service rate: {self.current_service_rates:.3f}")
            print(f"Current reward: {self.prev_reward:.3f}")
            print(f"Stability: {self._calculate_stability_bonus():.3f}")


def create_fixed_drl_environment():
    """Create fixed version DRL environment"""
    return DRLOptimizedQueueEnvFixed()


if __name__ == "__main__":
    # Test fixed version environment
    env = create_fixed_drl_environment()

    print("🛠️ DRL Optimized Environment Fixed Version Test")
    print("=" * 60)
    print(f"Action space: {env.action_space}")
    print(f"Observation space keys: {list(env.observation_space.spaces.keys())}")

    # Run test
    obs, info = env.reset()
    print(f"\nInitial observation dimensions:")
    for key, value in obs.items():
        print(f"  {key}: {value.shape}")

    for step in range(5):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)

        print(f"\nStep {step+1}:")
        print(f"Action: service intensity{action['service_intensities'][:3]:.2f}..., arrival multiplier{action['arrival_multiplier'][0]:.2f}")
        print(f"Reward: {reward:.3f}")
        print(f"Queue: {obs['queue_lengths']}")
        print(f"Load rate: {obs['load_rates']:.2f}")
        print(f"Stability: {info['stability_score']:.3f}")

        if term or trunc:
            break

    env.close()
    print("\n✅ Fixed version environment testing complete")