"""
State Manager

Implements 128-dimensional state space design in 01 theory:
- Hierarchical encoding of 6 dimension segments
- Semantically separated observation space
- State representation guaranteeing Markov property
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from .config import VerticalQueueConfig


@dataclass
class StateSegment:
    """State segment definition"""
    name: str
    start_idx: int
    end_idx: int
    size: int
    description: str


class StateManager:
    """
    State Manager
    
    Responsible for constructing and managing 128-dimensional state space, implementing:
    1. Hierarchical encoding: 6 semantic dimension segments
    2. Markov property: Contains complete decision-relevant information
    3. Normalization: All state values normalized to [0,1] interval
    4. Semantic separation: Different types of information encoded separately
    """
    
    def __init__(self, config: VerticalQueueConfig):
        self.config = config
        self.state_dim = 128
        
        # Define 6 state dimension segments
        self._define_state_segments()
        
        # Initialize state cache
        self.current_state = np.zeros(self.state_dim, dtype=np.float32)
        self.state_history = []
        self.action_history = []
        
        # State statistics (for normalization)
        self.state_stats = self._initialize_state_stats()
    
    def _define_state_segments(self):
        """
        Define 6 dimension segments of 128-dimensional state
        
        State space design based on 01 theory:
        1. Queue states segment (40 dims): Detailed states of 5-layer queues
        2. Cabinet states segment (32 dims): 24-compartment cabinet states + 8-dim temperature zone info  
        3. Performance metrics segment (16 dims): Throughput, waiting time, utilization, etc.
        4. Dynamics parameters segment (16 dims): Arrival rate, service rate, transition probability
        5. History information segment (16 dims): Recent state change trends
        6. Control information segment (8 dims): Current control strategy and action effects
        """
        segments = []
        current_idx = 0
        
        # 1. Queue states segment (40 dims)
        segments.append(StateSegment(
            name="queue_states",
            start_idx=current_idx,
            end_idx=current_idx + 40,
            size=40,
            description="Detailed states of 5-layer queues: length, waiting time, priority distribution, etc."
        ))
        current_idx += 40
        
        # 2. Cabinet states segment (32 dims)
        segments.append(StateSegment(
            name="cabinet_states",
            start_idx=current_idx,
            end_idx=current_idx + 32,
            size=32,
            description="24-compartment cabinet states + 8-dim temperature zone management info"
        ))
        current_idx += 32
        
        # 3. Performance metrics segment (16 dims)
        segments.append(StateSegment(
            name="performance_metrics",
            start_idx=current_idx,
            end_idx=current_idx + 16,
            size=16,
            description="System performance metrics: throughput, utilization, stability, etc."
        ))
        current_idx += 16
        
        # 4. Dynamics parameters segment (16 dims)
        segments.append(StateSegment(
            name="dynamics_params",
            start_idx=current_idx,
            end_idx=current_idx + 16,
            size=16,
            description="Current dynamics parameters: arrival rate, service rate, transition probability"
        ))
        current_idx += 16
        
        # 5. History information segment (16 dims)
        segments.append(StateSegment(
            name="history_info",
            start_idx=current_idx,
            end_idx=current_idx + 16,
            size=16,
            description="Historical state change trends and statistical information"
        ))
        current_idx += 16
        
        # 6. Control information segment (8 dims)
        segments.append(StateSegment(
            name="control_info",
            start_idx=current_idx,
            end_idx=current_idx + 8,
            size=8,
            description="Current control strategy and action effect feedback"
        ))
        current_idx += 8
        
        # Verify total dimensions
        assert current_idx == self.state_dim, f"State dimension mismatch: {current_idx} != {self.state_dim}"
        
        # Store segment definitions
        self.segments = {seg.name: seg for seg in segments}
        self.segment_list = segments
    
    def _initialize_state_stats(self) -> Dict:
        """
        Initialize state statistics
        
        Used for state normalization and anomaly detection
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
        Reset state manager
        """
        self.current_state = np.zeros(self.state_dim, dtype=np.float32)
        self.state_history = []
        self.action_history = []
    
    def update_state(self, queue_info: Dict, cabinet_info: Dict, 
                     performance_info: Dict, dynamics_info: Dict):
        """
        Update complete 128-dimensional state vector
        
        Args:
            queue_info: Queue information dictionary
            cabinet_info: Cabinet information dictionary  
            performance_info: Performance information dictionary
            dynamics_info: Dynamics information dictionary
        """
        # Construct new state vector
        new_state = np.zeros(self.state_dim, dtype=np.float32)
        
        # 1. Update queue states segment
        self._update_queue_states(new_state, queue_info)
        
        # 2. Update cabinet states segment
        self._update_cabinet_states(new_state, cabinet_info)
        
        # 3. Update performance metrics segment
        self._update_performance_metrics(new_state, performance_info)
        
        # 4. Update dynamics parameters segment
        self._update_dynamics_params(new_state, dynamics_info)
        
        # 5. Update history information segment
        self._update_history_info(new_state)
        
        # 6. Update control information segment
        self._update_control_info(new_state)
        
        # Normalize state
        new_state = self._normalize_state(new_state)
        
        # Update state history
        self.state_history.append(self.current_state.copy())
        if len(self.state_history) > self.config.history_length:
            self.state_history.pop(0)
        
        # Update current state
        self.current_state = new_state
        
        # Update state statistics
        self._update_state_stats(new_state)
    
    def _update_queue_states(self, state: np.ndarray, queue_info: Dict):
        """
        Update queue states segment (40 dims)
        
        8 dims per layer:
        - Queue length (1 dim)
        - Average waiting time (1 dim)  
        - Priority distribution (3 dims: high/medium/low)
        - Load factor ρ (1 dim)
        - Transfer state (1 dim)
        - Service state (1 dim)
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
            
            # Queue length (normalized to capacity)
            capacity = self.config.layer_capacities[i]
            state[base_idx] = queue_lengths[i] / capacity if capacity > 0 else 0
            
            # Average waiting time (normalized to max waiting time)
            max_wait = self.config.max_waiting_time
            state[base_idx + 1] = min(waiting_times[i] / max_wait, 1.0)
            
            # Priority distribution (3 dims, already normalized)
            priority_dist = priority_dists[i] if i < len(priority_dists) else [0.33, 0.33, 0.34]
            state[base_idx + 2:base_idx + 5] = priority_dist[:3]
            
            # Load factor (ρi, theoretically should be < 1)
            state[base_idx + 5] = min(load_factors[i] if i < len(load_factors) else 0, 1.0)
            
            # Transfer state (activity of downward transfer from current layer)
            state[base_idx + 6] = transfer_states[i] if i < len(transfer_states) else 0
            
            # Service state (service activity of current layer)  
            state[base_idx + 7] = service_states[i] if i < len(service_states) else 0
    
    def _update_cabinet_states(self, state: np.ndarray, cabinet_info: Dict):
        """
        Update cabinet states segment (32 dims)
        
        24-dim compartment states + 8-dim temperature zone management:
        - 24 compartment occupancy states (24 dims, 0/1)
        - 3 temperature zone temperatures (3 dims, normalized)  
        - 3 temperature zone loads (3 dims, occupancy rate)
        - Temperature control system state (1 dim)
        - Service queue length (1 dim)
        """
        seg = self.segments["cabinet_states"]
        
        # 24 compartment states
        grid_states = cabinet_info.get('grid_states', [0] * 24)
        state[seg.start_idx:seg.start_idx + 24] = grid_states[:24]
        
        # Temperature zone temperatures (normalized to target temperature ranges)
        temperatures = cabinet_info.get('temperatures', [20, 60, 5])  # Normal, hot, cold
        temp_ranges = [(15, 25), (55, 65), (0, 10)]  # Normal ranges for each zone
        
        for i, (temp, (min_t, max_t)) in enumerate(zip(temperatures, temp_ranges)):
            normalized_temp = (temp - min_t) / (max_t - min_t) if max_t > min_t else 0.5
            state[seg.start_idx + 24 + i] = np.clip(normalized_temp, 0, 1)
        
        # Temperature zone loads
        zone_loads = cabinet_info.get('zone_loads', [0, 0, 0])
        state[seg.start_idx + 27:seg.start_idx + 30] = zone_loads[:3]
        
        # Temperature control system state
        thermal_status = cabinet_info.get('thermal_status', 0)
        state[seg.start_idx + 30] = thermal_status
        
        # Service queue length
        service_queue = cabinet_info.get('service_queue_length', 0)
        max_service_queue = 10  # Assume max service queue length
        state[seg.start_idx + 31] = min(service_queue / max_service_queue, 1.0)
    
    def _update_performance_metrics(self, state: np.ndarray, performance_info: Dict):
        """
        Update performance metrics segment (16 dims)
        
        Key performance indicators:
        - System throughput (1 dim)
        - Average waiting time (1 dim)
        - Waiting time variance (1 dim) 
        - Fairness metric (1 dim)
        - Layer utilization rates (5 dims)
        - Stability metrics (3 dims)
        - Efficiency metrics (4 dims)
        """
        seg = self.segments["performance_metrics"]
        
        # System throughput
        throughput = performance_info.get('throughput', 0)
        max_throughput = self.config.theoretical_max_throughput
        state[seg.start_idx] = min(throughput / max_throughput, 1.0)
        
        # Average waiting time
        avg_waiting = performance_info.get('avg_waiting_time', 0)
        max_wait = self.config.max_waiting_time
        state[seg.start_idx + 1] = min(avg_waiting / max_wait, 1.0)
        
        # Waiting time variance  
        wait_variance = performance_info.get('waiting_time_variance', 0)
        max_variance = max_wait * max_wait
        state[seg.start_idx + 2] = min(wait_variance / max_variance, 1.0)
        
        # Fairness metric
        fairness = performance_info.get('fairness', 1.0)
        state[seg.start_idx + 3] = fairness
        
        # Layer utilization rates (5 dims)
        utilizations = performance_info.get('layer_utilizations', [0] * 5)
        state[seg.start_idx + 4:seg.start_idx + 9] = utilizations[:5]
        
        # Stability metrics (3 dims)
        stability_metrics = performance_info.get('stability_metrics', [0, 0, 0])
        state[seg.start_idx + 9:seg.start_idx + 12] = stability_metrics[:3]
        
        # Efficiency metrics (4 dims)
        efficiency_metrics = performance_info.get('efficiency_metrics', [0, 0, 0, 0])
        state[seg.start_idx + 12:seg.start_idx + 16] = efficiency_metrics[:4]
    
    def _update_dynamics_params(self, state: np.ndarray, dynamics_info: Dict):
        """
        Update dynamics parameters segment (16 dims)
        
        Current system dynamics parameters:
        - Layer arrival rates (5 dims)
        - Layer service rates (5 dims)  
        - Inter-layer transition probabilities (4 dims)
        - System load (1 dim)
        - Control parameter (1 dim)
        """
        seg = self.segments["dynamics_params"]
        
        # Layer arrival rates (normalized to base arrival rate)
        arrival_rates = dynamics_info.get('arrival_rates', [0] * 5)
        base_rate = self.config.base_arrival_rate
        for i in range(5):
            rate = arrival_rates[i] if i < len(arrival_rates) else 0
            state[seg.start_idx + i] = min(rate / (base_rate * 2), 1.0)  # Assume max is 2x base rate
        
        # Layer service rates (normalized to theoretical max service rate)
        service_rates = dynamics_info.get('service_rates', [0] * 5)
        max_service = max(self.config.layer_service_rates) * 2  # Assume max is 2x theoretical value
        for i in range(5):
            rate = service_rates[i] if i < len(service_rates) else 0
            state[seg.start_idx + 5 + i] = min(rate / max_service, 1.0)
        
        # Inter-layer transition probabilities (4 dims, L5->L4, L4->L3, L3->L2, L2->L1)
        transfer_probs = dynamics_info.get('transfer_probabilities', [0] * 4)
        state[seg.start_idx + 10:seg.start_idx + 14] = transfer_probs[:4]
        
        # System load
        system_load = dynamics_info.get('system_load', 0)
        state[seg.start_idx + 14] = min(system_load, 1.0)
        
        # Control parameter (current control intensity)
        control_intensity = dynamics_info.get('control_intensity', 0)
        state[seg.start_idx + 15] = control_intensity
    
    def _update_history_info(self, state: np.ndarray):
        """
        Update history information segment (16 dims)
        
        Historical trends and statistical information:
        - Recent performance trends (8 dims)
        - State change rates (4 dims)
        - Anomaly detection metrics (2 dims)
        - Learning progress metrics (2 dims)
        """
        seg = self.segments["history_info"]
        
        if len(self.state_history) > 0:
            # Recent performance trends (calculate performance changes in recent steps)
            recent_states = self.state_history[-min(8, len(self.state_history)):]
            if len(recent_states) > 1:
                # Calculate trend in performance segment
                perf_seg = self.segments["performance_metrics"]
                recent_perf = [s[perf_seg.start_idx:perf_seg.start_idx+8] for s in recent_states]
                
                # Calculate trend (linear fit slope)
                for i in range(8):
                    if len(recent_perf) > 1:
                        values = [perf[i] if i < len(perf) else 0 for perf in recent_perf]
                        trend = self._calculate_trend(values)
                        state[seg.start_idx + i] = np.clip((trend + 1) / 2, 0, 1)  # Normalize to [0,1]
            
            # State change rates
            if len(self.state_history) > 0:
                last_state = self.state_history[-1]
                state_diff = np.abs(self.current_state - last_state)
                # Take average change rate of several key segments
                key_segments = ["queue_states", "performance_metrics", "dynamics_params"]
                for i, seg_name in enumerate(key_segments[:4]):
                    if seg_name in self.segments:
                        seg_info = self.segments[seg_name]
                        avg_change = np.mean(state_diff[seg_info.start_idx:seg_info.end_idx])
                        state[seg.start_idx + 8 + i] = min(avg_change * 10, 1.0)  # Amplify change rate
            
            # Anomaly detection metrics (simple implementation)
            state[seg.start_idx + 12] = self._detect_anomaly()
            state[seg.start_idx + 13] = self._calculate_stability_score()
            
            # Learning progress metrics (can implement more complex logic as needed)
            state[seg.start_idx + 14] = min(len(self.state_history) / 1000, 1.0)  # Experience accumulation
            state[seg.start_idx + 15] = self._calculate_learning_progress()
    
    def _update_control_info(self, state: np.ndarray):
        """
        Update control information segment (8 dims)
        
        Current control strategy and effects:
        - Recent action effects (4 dims)
        - Control strategy parameters (3 dims)  
        - Adaptive parameter (1 dim)
        """
        seg = self.segments["control_info"]
        
        if len(self.action_history) > 0:
            last_action = self.action_history[-1]
            
            # Recent action effects (simplified evaluation)
            # More complex action effect evaluation logic can be implemented here
            action_effects = [0.5, 0.5, 0.5, 0.5]  # Placeholder
            state[seg.start_idx:seg.start_idx + 4] = action_effects
            
            # Control strategy parameters
            if 'service_priorities' in last_action:
                priorities = last_action['service_priorities'][:3]
                state[seg.start_idx + 4:seg.start_idx + 7] = priorities
            
            # Adaptive parameter (control intensity adaptation)
            adaptive_param = self._calculate_adaptive_param()
            state[seg.start_idx + 7] = adaptive_param
    
    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """
        Normalize state vector
        
        Ensure all state values are in [0,1] interval
        """
        # Simple clip operation, more complex normalization can be based on state statistics
        return np.clip(state, 0.0, 1.0)
    
    def _update_state_stats(self, state: np.ndarray):
        """
        Update state statistics
        """
        stats = self.state_stats
        stats['update_count'] += 1
        
        # Update min/max values
        stats['min_values'] = np.minimum(stats['min_values'], state)
        stats['max_values'] = np.maximum(stats['max_values'], state)
        
        # Update mean and std (online update)
        alpha = 1.0 / min(stats['update_count'], 1000)  # Learning rate decay
        stats['mean_values'] = (1 - alpha) * stats['mean_values'] + alpha * state
        
        diff = state - stats['mean_values']
        stats['std_values'] = (1 - alpha) * stats['std_values'] + alpha * (diff * diff)
    
    def _calculate_trend(self, values: List[float]) -> float:
        """
        Calculate trend of value sequence (simple linear fit)
        
        Returns:
            Trend value, positive indicates upward, negative indicates downward
        """
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x = np.arange(n)
        y = np.array(values)
        
        # Simple linear regression
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
        Simple anomaly detection
        
        Returns:
            Anomaly degree [0,1], 0 is normal, 1 is anomalous
        """
        if len(self.state_history) < 5:
            return 0.0
        
        # Check deviation of recent states from historical mean
        recent_mean = np.mean(self.state_history[-5:], axis=0)
        historical_mean = np.mean(self.state_history[:-5], axis=0) if len(self.state_history) > 5 else recent_mean
        
        deviation = np.mean(np.abs(recent_mean - historical_mean))
        return min(deviation * 5, 1.0)  # Amplify deviation
    
    def _calculate_stability_score(self) -> float:
        """
        Calculate system stability score
        
        Returns:
            Stability score [0,1], 1 is most stable
        """
        if len(self.state_history) < 3:
            return 1.0
        
        # Calculate variance of recent states
        recent_states = np.array(self.state_history[-3:])
        variance = np.mean(np.var(recent_states, axis=0))
        
        # Convert to stability score
        stability = 1.0 / (1.0 + variance * 100)
        return stability
    
    def _calculate_learning_progress(self) -> float:
        """
        Calculate learning progress metric
        
        Returns:
            Learning progress [0,1]
        """
        # Simple implementation: based on performance improvement
        if len(self.state_history) < 10:
            return 0.0
        
        # Compare recent and early performance
        perf_seg = self.segments["performance_metrics"]
        recent_perf = np.mean([s[perf_seg.start_idx] for s in self.state_history[-5:]])
        early_perf = np.mean([s[perf_seg.start_idx] for s in self.state_history[:5]])
        
        improvement = recent_perf - early_perf
        return np.clip((improvement + 1) / 2, 0, 1)
    
    def _calculate_adaptive_param(self) -> float:
        """
        Calculate adaptive control parameter
        
        Returns:
            Adaptive parameter [0,1]
        """
        # Automatically adjust control intensity based on system performance
        if len(self.state_history) < 2:
            return 0.5
        
        # Simple implementation: adjust based on performance change
        perf_seg = self.segments["performance_metrics"]
        current_perf = self.current_state[perf_seg.start_idx]
        last_perf = self.state_history[-1][perf_seg.start_idx] if self.state_history else current_perf
        
        perf_change = current_perf - last_perf
        
        # Increase control intensity when performance drops, decrease when performance improves
        if perf_change < 0:
            return min(0.8, 0.5 - perf_change)
        else:
            return max(0.2, 0.5 - perf_change * 0.5)
    
    def get_observation(self) -> np.ndarray:
        """
        Get current observation state
        
        Returns:
            128-dimensional state vector
        """
        return self.current_state.copy()
    
    def get_state_info(self) -> Dict:
        """
        Get detailed information of state manager
        
        Returns:
            State information dictionary
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
        Update action history
        
        Args:
            action: Action dictionary
        """
        self.action_history.append(action.copy())
        if len(self.action_history) > self.config.history_length:
            self.action_history.pop(0)
    
    def get_segment_state(self, segment_name: str) -> np.ndarray:
        """
        Get state of specific segment
        
        Args:
            segment_name: Segment name
            
        Returns:
            State vector of that segment
        """
        if segment_name not in self.segments:
            raise ValueError(f"Unknown segment: {segment_name}")
        
        seg = self.segments[segment_name]
        return self.current_state[seg.start_idx:seg.end_idx].copy()
    
    def parse_state(self, state: np.ndarray) -> Dict:
        """
        Parse state vector into segments
        
        Args:
            state: 128-dimensional state vector
            
        Returns:
            Parsed state dictionary
        """
        if len(state) != self.state_dim:
            raise ValueError(f"State dimension mismatch: expected {self.state_dim}, got {len(state)}")
        
        parsed = {}
        for segment_name, seg in self.segments.items():
            parsed[segment_name] = state[seg.start_idx:seg.end_idx].copy()
        
        return parsed


# Test state manager
if __name__ == "__main__":
    from .config import VerticalQueueConfig
    
    config = VerticalQueueConfig()
    state_manager = StateManager(config)
    
    print("State manager created successfully!")
    print(f"State dimension: {state_manager.state_dim}")
    print(f"Number of state segments: {len(state_manager.segments)}")
    
    # Display state segment information
    for name, seg in state_manager.segments.items():
        print(f"{name}: [{seg.start_idx}:{seg.end_idx}] ({seg.size} dims) - {seg.description}")
    
    # Test state update
    dummy_info = {
        'queue_lengths': [2, 3, 1, 2, 1],
        'waiting_times': [5, 8, 3, 6, 2],
        'priority_distributions': [[0.3, 0.4, 0.3]] * 5,
        'load_factors': [0.5, 0.6, 0.4, 0.7, 0.3]
    }
    
    state_manager.update_state(dummy_info, {}, {}, {})
    observation = state_manager.get_observation()
    
    print(f"\nObservation state dimension: {observation.shape}")
    print(f"State value range: [{observation.min():.3f}, {observation.max():.3f}]")
    print("State manager test completed!")
