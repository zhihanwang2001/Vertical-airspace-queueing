"""
Queue Dynamics Module

Implements vertical layered queue dynamics in 01 theory:
- Non-homogeneous Poisson arrival process Ni(t) ~ NHPP(λi(t))
- Layered service rules Si ~ Gi(μi, σi²)
- Inter-layer transfer dynamics T(li, li-1 | Q(t))
- Stability condition verification ρi = λi^eff/(μi·ci) < 1
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
    """UAV order object"""
    id: int
    arrival_time: int
    priority: str  # 'high', 'medium', 'low'
    temperature_zone: str  # 'hot', 'cold', 'normal'
    size: str  # 'small', 'medium', 'large'
    urgency_level: float  # [0,1]
    wait_tolerance: int  # Maximum tolerable waiting time
    current_layer: int  # Current layer
    total_wait_time: int = 0


@dataclass
class LayerState:
    """Single layer queue state"""
    layer_id: int
    height: float  # Height (m)
    capacity: int  # Capacity
    current_length: int = 0  # Current queue length
    waiting_orders: deque = None  # Waiting queue
    service_rate: float = 1.0  # Current service rate
    arrival_rate: float = 0.25  # Current arrival rate
    total_arrivals: int = 0  # Total arrivals
    total_departures: int = 0  # Total departures
    total_transfers_in: int = 0  # Transfers in
    total_transfers_out: int = 0  # Transfers out
    
    def __post_init__(self):
        if self.waiting_orders is None:
            self.waiting_orders = deque()


class QueueDynamics:
    """
    Queue Dynamics Manager

    Implements core dynamics of 01 theory:
    1. Arrival process: λi(t) = λ0 · αi · f(urgency) · g(saturation_{i-1})
    2. Service process: Si ~ Gi(μi, σi²), μi < μi+1 (ascending priority, higher layers serve faster)
    3. Transfer process: T(li, li-1 | Q(t)) based on waiting time and capacity state
    4. Stability: ρi = λi^eff/(μi·ci) < 1
    """
    
    def __init__(self, config: VerticalQueueConfig):
        self.config = config
        self.math_utils = MathUtils()
        
        # Dynamics parameters (based on 01 theory standard configuration)
        self.base_arrival_rate = config.base_arrival_rate  # λ0 = 0.25
        self.arrival_weights = np.array(config.arrival_weights)  # α = [0.1, 0.15, 0.25, 0.3, 0.2]
        self.service_rates = np.array(config.layer_service_rates)  # μ = [0.8, 0.9, 1.0, 1.2, 1.5]
        self.min_wait_times = np.array(config.min_wait_times)  # τ_min = [10, 8, 6, 4, 2]
        
        # Initialize layer states
        self.layers = self._initialize_layers()
        
        # System parameters
        self.current_step = 0
        self.order_counter = 0
        
        # Transfer parameters
        self.transfer_probabilities = np.zeros(config.num_layers - 1)  # 4 transfer probabilities
        self.transfer_enable = np.ones(config.num_layers, dtype=bool)  # Transfer switches
        
        # Performance statistics
        self.performance_history = {
            'throughput': [],
            'waiting_times': [],
            'queue_lengths': [],
            'load_factors': [],
            'transfer_counts': []
        }
        
        # Random number generator
        self.rng = np.random.RandomState(config.random_seed)
    
    def _initialize_layers(self) -> List[LayerState]:
        """
        Initialize 5-layer queue states
        
        Based on 01 theory inverted pyramid capacity structure: C = {8, 6, 4, 3, 2}
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
        Reset queue dynamics system
        """
        self.current_step = 0
        self.order_counter = 0
        
        # Clear all queues
        for layer in self.layers:
            layer.current_length = 0
            layer.waiting_orders.clear()
            layer.total_arrivals = 0
            layer.total_departures = 0
            layer.total_transfers_in = 0
            layer.total_transfers_out = 0
        
        # Reset statistics
        for key in self.performance_history:
            self.performance_history[key].clear()
    
    def step(self, action: Dict) -> Dict:
        """
        Queue dynamics single step update
        
        Implements complete dynamics process of 01 theory:
        1. Generate new arrivals (NHPP process)
        2. Process service departures
        3. Execute inter-layer transfers
        4. Update waiting times
        5. Calculate performance metrics
        
        Args:
            action: Action dictionary containing control decisions
            
        Returns:
            Queue state information dictionary
        """
        self.current_step += 1
        
        # 1. Process new arrivals (based on 01 theory arrival process)
        new_arrivals = self._process_arrivals()
        
        # 2. Process service departures (layered service rules)
        service_info = self._process_service()
        
        # 3. Process inter-layer transfers (transfer dynamics)
        transfer_info = self._process_transfers()
        
        # 4. Update waiting times
        self._update_waiting_times()
        
        # 5. Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics()
        
        # 6. Update performance history
        self._update_performance_history(performance_metrics)
        
        # 7. Prepare return information
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
            'service_requests': service_info.get('completed_orders', []),  # Provide service requests for delivery cabinet
            'transfers': transfer_info['total_transfers'],
            'blocked': sum(performance_metrics.get('blocked_arrivals', []))
        }
    
    def _process_arrivals(self) -> List[int]:
        """
        Process new arrival process
        
        Implements 01 theory arrival model:
        λi(t) = λ0 · αi · f(urgency) · g(saturation_{i-1})
        
        Returns:
            List of new arrival counts for each layer
        """
        new_arrivals = []
        
        for i, layer in enumerate(self.layers):
            # Calculate effective arrival rate for current layer
            effective_rate = self._calculate_effective_arrival_rate(i)
            
            # Poisson sampling
            arrivals = self.rng.poisson(effective_rate)
            
            # Capacity check and arrival processing
            actual_arrivals = 0
            for _ in range(arrivals):
                if layer.current_length < layer.capacity:
                    # Create new order
                    order = self._create_order(i)
                    layer.waiting_orders.append(order)
                    layer.current_length += 1
                    layer.total_arrivals += 1
                    actual_arrivals += 1
                else:
                    # Capacity full, order blocked
                    # Consider transferring to upper layer
                    if i < len(self.layers) - 1 and self.layers[i + 1].current_length < self.layers[i + 1].capacity:
                        # Transfer to upper layer
                        order = self._create_order(i + 1)
                        self.layers[i + 1].waiting_orders.append(order)
                        self.layers[i + 1].current_length += 1
                        self.layers[i + 1].total_arrivals += 1
                        actual_arrivals += 1
            
            new_arrivals.append(actual_arrivals)
        
        return new_arrivals
    
    def _calculate_effective_arrival_rate(self, layer_idx: int) -> float:
        """
        Calculate effective arrival rate for layer i
        
        Based on 01 theory formula:
        λi(t) = λ0 · αi · f(urgency) · g(saturation_{i-1})
        """
        # Base arrival rate
        base_rate = self.base_arrival_rate * self.arrival_weights[layer_idx]
        
        # Urgency factor f(urgency)
        urgency_factor = self._calculate_urgency_factor()
        
        # Lower layer saturation factor g(saturation_{i-1})
        saturation_factor = 1.0
        if layer_idx > 0:
            prev_layer = self.layers[layer_idx - 1]
            saturation = prev_layer.current_length / prev_layer.capacity if prev_layer.capacity > 0 else 0
            saturation_factor = max(0, 1 - saturation)  # More full lower layer, less arrivals to current layer
        
        return base_rate * urgency_factor * saturation_factor
    
    def _calculate_urgency_factor(self) -> float:
        """
        Calculate urgency factor
        
        f(urgency) = Σ βu · P(urgency = u)
        """
        # Simplified implementation: dynamically adjust based on current system load
        total_load = sum(layer.current_length for layer in self.layers)
        max_load = sum(layer.capacity for layer in self.layers)
        
        system_pressure = total_load / max_load if max_load > 0 else 0
        
        # Higher system pressure, higher proportion of urgent orders, overall arrival rate may increase
        urgency_factor = 1.0 + 0.5 * system_pressure
        return min(urgency_factor, 2.0)  # Limit maximum growth
    
    def _create_order(self, layer_idx: int) -> UAVOrder:
        """
        Create new order
        """
        self.order_counter += 1
        
        # Randomly generate order attributes
        priorities = ['low', 'medium', 'high']
        priority_probs = [0.5, 0.3, 0.2]  # More low priority, less high priority
        priority = self.rng.choice(priorities, p=priority_probs)
        
        temp_zones = ['normal', 'hot', 'cold']
        temp_probs = [0.5, 0.3, 0.2]
        temp_zone = self.rng.choice(temp_zones, p=temp_probs)
        
        sizes = ['small', 'medium', 'large']
        size_probs = [0.4, 0.4, 0.2]
        size = self.rng.choice(sizes, p=size_probs)
        
        # Urgency level and wait tolerance
        urgency_map = {'low': 0.2, 'medium': 0.5, 'high': 0.8}
        urgency = urgency_map[priority] + self.rng.normal(0, 0.1)
        urgency = np.clip(urgency, 0, 1)
        
        wait_tolerance = int(self.rng.exponential(20)) + 5  # 5-100 steps wait tolerance
        
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
        Process service departure process

        Implements layered service characteristics of 01 theory:
        1. Ascending priority: μi < μi+1 (higher altitude serves faster, e.g., μ1=0.8 < μ5=1.5)
        2. Inverted pyramid capacity: ci < ci+1 (higher altitude has larger capacity, e.g., C1=2 < C5=8)
        3. Urgency priority: High urgency orders served first
        """
        service_completions = []
        
        for i, layer in enumerate(self.layers):
            if layer.current_length == 0:
                service_completions.append(0)
                continue
            
            # Calculate effective service rate for current layer
            effective_service_rate = layer.service_rate
            
            # Service capacity (can serve multiple orders simultaneously)
            service_capacity = min(layer.capacity, layer.current_length)
            
            # Decide whether to complete service for each service position
            completed = 0
            orders_to_remove = []
            
            # Sort orders by priority (urgency priority)
            orders_list = list(layer.waiting_orders)
            orders_list.sort(key=lambda x: (x.priority == 'high', x.priority == 'medium', -x.total_wait_time))
            
            for j, order in enumerate(orders_list[:service_capacity]):
                # Service completion probability (based on service rate)
                service_prob = effective_service_rate
                
                if self.rng.random() < service_prob:
                    orders_to_remove.append(order)
                    completed += 1
            
            # Remove completed orders
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
        Process inter-layer transfer process
        
        Implements transfer dynamics of 01 theory:
        T(li, li-1 | Q(t)) = φ(wi/τi^min) · ψ((ci-1 - ni-1)/ci-1)
        """
        total_transfers = 0
        transfer_details = []
        
        # Process transfers layer by layer from upper to lower
        for i in range(len(self.layers) - 1, 0, -1):  # L5->L4, L4->L3, ..., L2->L1
            source_layer = self.layers[i]
            target_layer = self.layers[i - 1]
            
            if source_layer.current_length == 0 or not self.transfer_enable[i]:
                transfer_details.append(0)
                continue
            
            transfers = 0
            orders_to_transfer = []
            
            # Check if each order meets transfer conditions
            for order in source_layer.waiting_orders:
                # Transfer condition check
                if self._check_transfer_conditions(order, i, i - 1):
                    # Calculate transfer probability
                    transfer_prob = self._calculate_transfer_probability(order, i, i - 1)
                    
                    if self.rng.random() < transfer_prob:
                        orders_to_transfer.append(order)
            
            # Execute transfers (limited by target layer capacity)
            for order in orders_to_transfer:
                if target_layer.current_length < target_layer.capacity:
                    # Execute transfer
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
                    break  # Target layer full, stop transfers
            
            transfer_details.append(transfers)
        
        return {
            'transfer_details': transfer_details,
            'total_transfers': total_transfers
        }
    
    def _check_transfer_conditions(self, order: UAVOrder, from_layer: int, to_layer: int) -> bool:
        """
        Check transfer conditions
        
        Transfer trigger conditions based on 01 theory:
        1. Time condition: wi ≥ τi^min
        2. Space condition: nj < cj  
        3. Priority condition: priority ≥ threshold_j
        """
        # 1. Time condition
        min_wait = self.min_wait_times[from_layer]
        if order.total_wait_time < min_wait:
            return False
        
        # 2. Space condition (checked in _process_transfers)
        target_layer = self.layers[to_layer]
        if target_layer.current_length >= target_layer.capacity:
            return False
        
        # 3. Priority condition (high priority more easily transferred)
        if order.priority == 'low' and order.total_wait_time < min_wait * 2:
            return False
        
        return True
    
    def _calculate_transfer_probability(self, order: UAVOrder, from_layer: int, to_layer: int) -> float:
        """
        Calculate transfer probability
        
        Based on 01 theory formula:
        T(li, li-1 | Q(t)) = φ(wi/τi^min) · ψ((ci-1 - ni-1)/ci-1)
        """
        # φ(wi/τi^min) - waiting time activation function
        min_wait = self.min_wait_times[from_layer]
        wait_ratio = order.total_wait_time / min_wait if min_wait > 0 else 1.0
        phi = min(1.0, max(0.0, wait_ratio))  # φ(x) = min(1, max(0, x))
        
        # ψ((ci-1 - ni-1)/ci-1) - vacancy availability function
        target_layer = self.layers[to_layer]
        available_ratio = (target_layer.capacity - target_layer.current_length) / target_layer.capacity
        available_ratio = max(0, available_ratio)
        gamma = 2.0  # Exponential parameter
        psi = available_ratio ** gamma  # ψ(x) = x^γ
        
        # Final transfer probability
        transfer_prob = phi * psi
        
        # Priority adjustment
        priority_multiplier = {'low': 0.8, 'medium': 1.0, 'high': 1.2}
        transfer_prob *= priority_multiplier.get(order.priority, 1.0)
        
        return min(transfer_prob, 1.0)
    
    def _update_waiting_times(self):
        """
        Update waiting times for all orders
        """
        for layer in self.layers:
            for order in layer.waiting_orders:
                order.total_wait_time += 1
    
    def _calculate_performance_metrics(self) -> Dict:
        """
        Calculate performance metrics
        
        Performance measures based on 01 theory:
        1. System throughput: Λ_system = Σ λi · (1 - P_block,i)
        2. Average waiting time: Wi = (ρi/(1-ρi)) · (1/μi) · (1+Cs²)/2
        3. Load factor: ρi = λi^eff/(μi·ci)
        """
        # System throughput (service completions in current step)
        throughput = sum(layer.total_departures for layer in self.layers)
        if self.current_step > 0:
            throughput = throughput / self.current_step  # Average throughput per step
        
        # Average waiting time
        total_wait = 0
        total_orders = 0
        for layer in self.layers:
            for order in layer.waiting_orders:
                total_wait += order.total_wait_time
                total_orders += 1
        
        avg_waiting_time = total_wait / total_orders if total_orders > 0 else 0
        
        # Load factors
        load_factors = self._calculate_load_factors()
        
        # Layer utilization rates
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
        Calculate load factors for each layer
        
        ρi = λi^eff/(μi·ci)
        """
        load_factors = {}
        
        for i, layer in enumerate(self.layers):
            # Effective arrival rate (including transfers from upper layer)
            effective_arrival = self._calculate_effective_arrival_rate(i)
            
            # Add arrival rate from upper layer transfers
            if i < len(self.layers) - 1:
                transfer_rate = self.transfer_probabilities[i] * effective_arrival
                effective_arrival += transfer_rate
            
            # Load factor
            service_capacity = layer.service_rate * layer.capacity
            rho = effective_arrival / service_capacity if service_capacity > 0 else 0
            
            load_factors[f'layer_{i}'] = min(rho, 2.0)  # Limit maximum value to avoid numerical issues
        
        return load_factors
    
    def _get_average_waiting_times(self) -> List[float]:
        """
        Get average waiting times for each layer
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
        Get priority distributions for each layer
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
                dist = [0.33, 0.33, 0.34]  # Default uniform distribution
            
            distributions.append(dist)
        return distributions
    
    def _get_transfer_states(self) -> List[float]:
        """
        Get transfer states for each layer
        """
        transfer_states = []
        for i, layer in enumerate(self.layers):
            if i == 0:  # Bottom layer cannot transfer
                transfer_states.append(0.0)
            else:
                # Calculate proportion of orders ready to transfer in current layer
                ready_to_transfer = 0
                for order in layer.waiting_orders:
                    if self._check_transfer_conditions(order, i, i-1):
                        ready_to_transfer += 1
                
                ratio = ready_to_transfer / len(layer.waiting_orders) if len(layer.waiting_orders) > 0 else 0
                transfer_states.append(ratio)
        
        return transfer_states
    
    def _get_service_states(self) -> List[float]:
        """
        Get service states for each layer
        """
        service_states = []
        for layer in self.layers:
            # Service activity = current length / capacity * service rate
            if layer.capacity > 0:
                activity = (layer.current_length / layer.capacity) * layer.service_rate
            else:
                activity = 0.0
            service_states.append(min(activity, 1.0))
        
        return service_states
    
    def _update_performance_history(self, metrics: Dict):
        """
        Update performance history
        """
        self.performance_history['throughput'].append(metrics['throughput'])
        self.performance_history['waiting_times'].append(metrics['avg_waiting_time'])
        self.performance_history['queue_lengths'].append([layer.current_length for layer in self.layers])
        self.performance_history['load_factors'].append(list(metrics['load_factors'].values()))
        
        # Maintain history length
        max_history = 1000
        for key in self.performance_history:
            if len(self.performance_history[key]) > max_history:
                self.performance_history[key].pop(0)
    
    def update_transfer_probabilities(self, transfer_decisions: np.ndarray):
        """
        Update transfer probabilities (based on agent actions)
        """
        # transfer_decisions is 5-dim binary vector indicating whether each layer allows transfer
        self.transfer_enable = transfer_decisions.astype(bool)
    
    def update_service_priorities(self, service_priorities: np.ndarray):
        """
        Update service priority weights
        """
        # Adjust service rates for each layer
        normalized_priorities = service_priorities / (np.sum(service_priorities) + 1e-8)
        for i, layer in enumerate(self.layers):
            if i < len(normalized_priorities):
                # Adjust based on base service rate
                layer.service_rate = self.service_rates[i] * (0.5 + normalized_priorities[i])
    
    def update_arrival_weights(self, arrival_weights: np.ndarray):
        """
        Update arrival weight allocation
        """
        # Update arrival weights
        self.arrival_weights = arrival_weights / (np.sum(arrival_weights) + 1e-8)
        
        # Update arrival rates for each layer
        for i, layer in enumerate(self.layers):
            if i < len(self.arrival_weights):
                layer.arrival_rate = self.base_arrival_rate * self.arrival_weights[i]
    
    def get_queue_lengths(self) -> List[int]:
        """Get current queue lengths"""
        return [layer.current_length for layer in self.layers]
    
    def get_waiting_times(self) -> List[float]:
        """Get average waiting times"""
        return self._get_average_waiting_times()
    
    def get_load_factors(self) -> Dict[str, float]:
        """Get load factors"""
        return self._calculate_load_factors()
    
    def get_system_info(self) -> Dict:
        """
        Get detailed system information
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


# Test queue dynamics
if __name__ == "__main__":
    from .config import VerticalQueueConfig
    
    config = VerticalQueueConfig()
    queue_dynamics = QueueDynamics(config)
    
    print("Queue dynamics module created successfully!")
    print(f"Number of layers: {len(queue_dynamics.layers)}")
    
    # Display layer information
    for i, layer in enumerate(queue_dynamics.layers):
        print(f"L{i+1}({layer.height}m): Capacity {layer.capacity}, Service rate {layer.service_rate:.2f}")
    
    # Test dynamics stepping
    dummy_action = {
        'transfer_decisions': np.array([1, 1, 1, 1, 0]),
        'service_priorities': np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
        'arrival_weights': np.array([0.1, 0.15, 0.25, 0.3, 0.2])
    }
    
    print("\nStarting dynamics simulation...")
    for step in range(10):
        info = queue_dynamics.step(dummy_action)
        print(f"Step {step + 1}: Queue lengths {info['queue_lengths']}, Throughput {info['throughput']:.3f}")
    
    print("\nQueue dynamics test completed!")
