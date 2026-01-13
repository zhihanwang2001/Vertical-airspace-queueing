"""
Configuration File

Complete parameter configuration based on theoretical framework:
- Vertical layered queue standard parameters
- Inverted pyramid capacity structure
- Performance prediction benchmarks
- Environment experimental settings
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np


@dataclass
class VerticalQueueConfig:
    """
    Vertical Layered Queue Configuration Class

    Standard parameter configuration based on theoretical framework:
    - Height layers: L = {100m, 80m, 60m, 40m, 20m}
    - Capacity configuration: C = {8, 6, 4, 3, 2} (inverted pyramid structure)
    - Arrival parameters: λ₀ = 0.25/step, α = [0.1, 0.15, 0.25, 0.3, 0.2]
    - Service parameters: μ = [0.8, 0.9, 1.0, 1.2, 1.5]
    """

    # ============= Basic System Parameters =============
    num_layers: int = 5
    layer_heights: List[float] = None  # [100, 80, 60, 40, 20] meters (L5→L1, high→low)
    layer_capacities: List[int] = None  # [8, 6, 4, 3, 2] inverted pyramid (L5→L1)

    # ============= Arrival Process Parameters =============
    base_arrival_rate: float = 0.25  # λ₀ = 0.25/step
    arrival_weights: List[float] = None  # α = [0.3, 0.25, 0.2, 0.15, 0.1] (L5→L1)

    # ============= Service Process Parameters =============
    layer_service_rates: List[float] = None  # μ = [1.2, 1.0, 0.8, 0.6, 0.4] (L5→L1)
    min_wait_times: List[int] = None  # τ_min = [10, 8, 6, 4, 2] steps

    # ============= Delivery Cabinet Parameters =============
    total_cabinet_cells: int = 24
    cells_per_zone: int = 8
    temperature_zones: List[str] = None  # ['cold', 'hot', 'normal']

    # ============= Environment Parameters =============
    max_episode_steps: int = 1000
    max_waiting_time: int = 50  # Maximum waiting time (steps)
    history_length: int = 20    # History record length
    random_seed: int = 42

    # ============= Performance Prediction Benchmarks =============
    theoretical_performance: Dict = None

    # ============= Experimental Settings Parameters =============
    experiment_configs: Dict = None
    
    def __post_init__(self):
        """Initialize default values"""

        # Height layer configuration (based on standard configuration)
        if self.layer_heights is None:
            self.layer_heights = [100.0, 80.0, 60.0, 40.0, 20.0]

        # Inverted pyramid capacity structure (core innovation)
        if self.layer_capacities is None:
            self.layer_capacities = [8, 6, 4, 3, 2]

        # Arrival weights (higher layers handle main traffic)
        # Array order: index 0→4 corresponds to L5→L1 (100m→20m)
        if self.arrival_weights is None:
            self.arrival_weights = [0.3, 0.25, 0.2, 0.15, 0.1]  # L5→L1

        # Service rates (higher layers serve faster: L5 fastest 1.2, L1 slowest 0.4)
        # Array order: index 0→4 corresponds to L5→L1 (100m→20m)
        if self.layer_service_rates is None:
            self.layer_service_rates = [1.2, 1.0, 0.8, 0.6, 0.4]  # L5→L1

        # Minimum waiting time (transfer trigger condition)
        if self.min_wait_times is None:
            self.min_wait_times = [10, 8, 6, 4, 2]

        # Temperature zone configuration
        if self.temperature_zones is None:
            self.temperature_zones = ['cold', 'hot', 'normal']

        # Theoretical performance prediction
        if self.theoretical_performance is None:
            self.theoretical_performance = {
                'avg_waiting_time_range': (15, 25),      # Average waiting time: 15-25 steps
                'system_throughput_range': (0.8, 1.2),  # System throughput: 0.8-1.2 orders/step
                'space_utilization_range': (0.75, 0.85), # Space utilization: 75-85%
                'efficiency_improvement': (0.4, 0.6),    # Vertical efficiency improvement: 40-60%
                'stability_threshold': 1.0,              # Stability threshold: ρᵢ < 1
                'optimal_load_factor': 0.8               # Optimal load factor
            }

        # Experiment configurations
        if self.experiment_configs is None:
            self.experiment_configs = {
                # H1 validation: Inverted pyramid capacity allocation optimality
                'H1_capacity_configurations': {
                    'pyramid': [8, 6, 4, 3, 2],         # Inverted pyramid (our theory)
                    'uniform': [5, 5, 5, 5, 5],         # Uniform allocation
                    'reverse_pyramid': [2, 3, 4, 6, 8], # Regular pyramid
                    'random': [4, 7, 3, 5, 6]           # Random allocation
                },

                # H2 validation: Layered vs single-layer queue performance comparison
                'H2_baseline_comparisons': {
                    'single_layer_capacity': 23,         # Single layer total capacity
                    'fifo_queue': True,                  # FIFO queueing
                    'priority_queue': True,              # Priority queueing
                    'mm1_model': True                    # M/M/1 theoretical model
                },

                # H3 validation: System stability testing
                'H3_stability_tests': {
                    'load_test_factors': [0.3, 0.5, 0.7, 0.8, 0.9, 0.95], # Load factor tests
                    'arrival_spike_tests': [2.0, 3.0, 5.0],                # Arrival rate spike tests
                    'service_degradation': [0.8, 0.6, 0.4],                # Service rate degradation tests
                },

                # H4 validation: Multi-objective Pareto optimality
                'H4_pareto_analysis': {
                    'objectives': ['throughput', 'fairness', 'efficiency', 'stability'],
                    'weight_combinations': [
                        [0.4, 0.2, 0.2, 0.2],  # Throughput priority
                        [0.2, 0.4, 0.2, 0.2],  # Fairness priority
                        [0.2, 0.2, 0.4, 0.2],  # Efficiency priority
                        [0.2, 0.2, 0.2, 0.4],  # Stability priority
                        [0.25, 0.25, 0.25, 0.25] # Balanced configuration
                    ]
                }
            }

        # Validate configuration validity
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameter validity"""

        # Check array length consistency
        assert len(self.layer_heights) == self.num_layers, "Layer height count mismatch"
        assert len(self.layer_capacities) == self.num_layers, "Capacity configuration count mismatch"
        assert len(self.arrival_weights) == self.num_layers, "Arrival weight count mismatch"
        assert len(self.layer_service_rates) == self.num_layers, "Service rate count mismatch"
        assert len(self.min_wait_times) == self.num_layers, "Minimum wait time count mismatch"

        # Check inverted pyramid property (core theory)
        for i in range(len(self.layer_capacities) - 1):
            assert self.layer_capacities[i] >= self.layer_capacities[i+1], \
                f"Violates inverted pyramid structure: C[{i}]={self.layer_capacities[i]} < C[{i+1}]={self.layer_capacities[i+1]}"

        # Check higher layers serve faster (theory requirement: L5 fastest, L1 slowest)
        # Since array index 0→4 corresponds to L5→L1, service rates should decrease
        for i in range(len(self.layer_service_rates) - 1):
            assert self.layer_service_rates[i] >= self.layer_service_rates[i+1], \
                f"Violates higher layer faster service principle: μ[{i}]={self.layer_service_rates[i]} < μ[{i+1}]={self.layer_service_rates[i+1]} (index {i} is L{5-i}, should be ≥ index {i+1}'s L{5-i-1})"

        # Check height decreases
        for i in range(len(self.layer_heights) - 1):
            assert self.layer_heights[i] > self.layer_heights[i+1], \
                f"Height layers should decrease: H[{i}]={self.layer_heights[i]} <= H[{i+1}]={self.layer_heights[i+1]}"

        # Check probability sum equals 1
        weight_sum = sum(self.arrival_weights)
        assert abs(weight_sum - 1.0) < 0.01, f"Arrival weight sum should be 1.0, currently {weight_sum}"

        print("✅ Configuration validation passed: meets theoretical requirements")
    
    @property
    def theoretical_max_throughput(self) -> float:
        """
        Theoretical maximum throughput

        Based on theory: Λ_max = Σ μᵢ·cᵢ
        """
        return sum(rate * cap for rate, cap in zip(self.layer_service_rates, self.layer_capacities))

    @property
    def total_system_capacity(self) -> int:
        """
        Total system capacity
        """
        return sum(self.layer_capacities)

    def get_stability_condition_params(self) -> Dict:
        """
        Get stability condition parameters

        Used to calculate ρᵢ = λᵢᵉᶠᶠ/(μᵢ·cᵢ) < 1
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
        Get experiment configuration for specific hypothesis

        Args:
            hypothesis: 'H1', 'H2', 'H3', 'H4'
        """
        hypothesis_key = f'{hypothesis}_'
        configs = {}

        for key, value in self.experiment_configs.items():
            if key.startswith(hypothesis_key):
                configs[key] = value

        if not configs:
            raise ValueError(f"Configuration for hypothesis {hypothesis} not found")

        return configs

    def create_baseline_config(self, baseline_type: str) -> 'VerticalQueueConfig':
        """
        Create baseline comparison configuration

        Args:
            baseline_type: 'uniform', 'reverse_pyramid', 'single_layer'
        """
        config = VerticalQueueConfig()

        if baseline_type == 'uniform':
            # Uniform capacity allocation
            avg_capacity = sum(self.layer_capacities) // self.num_layers
            config.layer_capacities = [avg_capacity] * self.num_layers

        elif baseline_type == 'reverse_pyramid':
            # Regular pyramid structure (opposite of our theory)
            config.layer_capacities = list(reversed(self.layer_capacities))

        elif baseline_type == 'single_layer':
            # Single layer queue
            config.num_layers = 1
            config.layer_heights = [self.layer_heights[0]]
            config.layer_capacities = [sum(self.layer_capacities)]
            config.arrival_weights = [1.0]
            config.layer_service_rates = [self.layer_service_rates[0]]
            config.min_wait_times = [self.min_wait_times[0]]

        return config

    def get_performance_bounds(self) -> Dict:
        """
        Get theoretical performance bounds

        Used to evaluate difference between actual performance and theoretical prediction
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
        Export configuration as dictionary format
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
        String representation of configuration information
        """
        return f"""
Vertical Layered Queue Configuration (Based on Theory):
================================
System Structure:
  - Layers: {self.num_layers}
  - Heights: {self.layer_heights} (m)
  - Capacities: {self.layer_capacities} (inverted pyramid)
  - Total Capacity: {self.total_system_capacity}

Dynamics Parameters:
  - Base Arrival Rate: {self.base_arrival_rate}/step
  - Arrival Weights: {self.arrival_weights}
  - Service Rates: {self.layer_service_rates} (ascending priority: higher layers faster)
  - Minimum Wait: {self.min_wait_times} steps

Theoretical Predictions:
  - Waiting Time: {self.theoretical_performance['avg_waiting_time_range']} steps
  - Throughput: {self.theoretical_performance['system_throughput_range']} orders/step
  - Utilization: {self.theoretical_performance['space_utilization_range']}
  - Efficiency Improvement: {self.theoretical_performance['efficiency_improvement']}

Delivery Cabinet Configuration:
  - Total Cells: {self.total_cabinet_cells}
  - Per Zone: {self.cells_per_zone}
  - Zone Types: {self.temperature_zones}
"""


# Predefined configurations
class ConfigPresets:
    """Predefined configuration templates"""

    @staticmethod
    def get_standard_config() -> VerticalQueueConfig:
        """Get standard configuration (default parameters)"""
        return VerticalQueueConfig()

    @staticmethod
    def get_high_load_config() -> VerticalQueueConfig:
        """High load test configuration"""
        config = VerticalQueueConfig()
        config.base_arrival_rate = 0.4  # Increase arrival rate
        config.max_episode_steps = 2000  # Extend test duration
        return config

    @staticmethod
    def get_stability_test_config() -> VerticalQueueConfig:
        """Stability test configuration"""
        config = VerticalQueueConfig()
        config.base_arrival_rate = 0.35  # Near stability boundary
        config.max_waiting_time = 100   # Increase waiting tolerance
        return config

    @staticmethod
    def get_efficiency_test_config() -> VerticalQueueConfig:
        """Efficiency test configuration"""
        config = VerticalQueueConfig()
        # Optimize service rate configuration
        config.layer_service_rates = [0.9, 1.0, 1.1, 1.3, 1.6]
        config.min_wait_times = [8, 6, 4, 3, 1]  # Faster transfers
        return config

    @staticmethod
    def get_debug_config() -> VerticalQueueConfig:
        """Debug configuration (quick test)"""
        config = VerticalQueueConfig()
        config.max_episode_steps = 100
        config.base_arrival_rate = 0.1
        config.history_length = 5
        return config


# Test configuration
if __name__ == "__main__":
    print("Testing configuration system...")

    # Create standard configuration
    config = VerticalQueueConfig()
    print(config)

    # Test stability parameters
    stability_params = config.get_stability_condition_params()
    print(f"Service capacities: {stability_params['service_capacities']}")
    print(f"Theoretical maximum throughput: {config.theoretical_max_throughput:.2f}")

    # Test experiment configuration
    h1_config = config.get_experiment_config('H1')
    print(f"H1 experiment configuration: {list(h1_config.keys())}")

    # Test baseline configuration
    uniform_config = config.create_baseline_config('uniform')
    print(f"Uniform configuration capacities: {uniform_config.layer_capacities}")

    # Test performance bounds
    bounds = config.get_performance_bounds()
    print(f"Waiting time target: {bounds['waiting_time']['target']}")

    print("\n✅ Configuration system test complete!")