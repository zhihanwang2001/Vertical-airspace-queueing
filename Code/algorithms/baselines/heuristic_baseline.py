"""
Heuristic Baseline
Heuristic baseline algorithm - strategy based on queuing theory and system expertise
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import time

from .base_baseline import BaseBaseline


class HeuristicBaseline(BaseBaseline):
    """Heuristic baseline algorithm implementation"""
    
    def __init__(self, 
                 env,
                 algorithm_name: str = "Heuristic",
                 config: Optional[Dict] = None):
        
        default_config = {
            # Heuristic strategy parameters
            'load_balance_threshold': 0.8,  # Load balance threshold
            'utilization_target': 0.7,     # Target utilization
            'emergency_threshold': 0.9,     # Emergency transfer threshold
            'service_rate_bounds': (0.1, 2.0),
            'arrival_rate_bounds': (0.5, 5.0),
            'adaptive_factor': 0.1,         # Adaptive adjustment factor
            'priority_weights': [1.0, 0.8, 0.6, 0.4, 0.2]  # Layer priority weights
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(env, algorithm_name, default_config)
        
        # Record historical states for adaptive adjustment
        self.history_window = 10
        self.utilization_history = []
        self.reward_history = []
        
        print(f"Heuristic Baseline initialized")
        print(f"Load balance threshold: {self.config['load_balance_threshold']}")
        print(f"Target utilization: {self.config['utilization_target']}")
    
    def _extract_state_info(self, observation) -> Dict:
        """Extract state information from observation"""
        if isinstance(observation, dict):
            # Extract information from Dict observation space
            extracted = {}
            extracted['queue_lengths'] = observation.get('queue_lengths', np.zeros(5))
            extracted['utilization_rates'] = observation.get('utilization_rates', np.zeros(5))
            extracted['service_rates'] = observation.get('service_rates', np.ones(5))
            extracted['load_rates'] = observation.get('load_rates', np.ones(5))
            
            # Infer arrival rates (from load rates and service rates)
            load_rates = extracted['load_rates']
            service_rates = extracted['service_rates']
            arrival_rates = load_rates * service_rates
            extracted['arrival_rates'] = arrival_rates
            
            # System metrics
            system_metrics = observation.get('system_metrics', np.zeros(3))
            extracted['throughput'] = system_metrics[0] if len(system_metrics) > 0 else 0
            extracted['system_load'] = system_metrics[1] if len(system_metrics) > 1 else np.mean(extracted['utilization_rates'])
            extracted['emergency_flags'] = [0, 0]  # Simplified
            extracted['waiting_times'] = np.zeros(5)  # Simplified
            
            return extracted
            
        elif isinstance(observation, (list, np.ndarray)):
            # Handle flattened observation
            obs_array = np.array(observation).flatten()
            
            # Based on 29-dimensional observation space structure
            if len(obs_array) >= 29:
                return {
                    'queue_lengths': obs_array[0:5],
                    'utilization_rates': obs_array[5:10],
                    'load_rates': obs_array[10:15],
                    'service_rates': obs_array[15:20],
                    'arrival_rates': obs_array[10:15] * obs_array[15:20],  # load_rates * service_rates
                    'waiting_times': np.zeros(5),
                    'throughput': obs_array[25] if len(obs_array) > 25 else 0,
                    'system_load': obs_array[26] if len(obs_array) > 26 else np.mean(obs_array[5:10]),
                    'emergency_flags': [0, 0]
                }
            else:
                # Simplified handling
                n_layers = 5
                return {
                    'queue_lengths': obs_array[0:n_layers] if len(obs_array) >= n_layers else np.zeros(n_layers),
                    'utilization_rates': obs_array[0:n_layers] * 0.5 if len(obs_array) >= n_layers else np.ones(n_layers) * 0.5,
                    'arrival_rates': np.ones(n_layers),
                    'service_rates': np.ones(n_layers),
                    'load_rates': np.ones(n_layers),
                    'waiting_times': np.zeros(n_layers),
                    'throughput': 0,
                    'system_load': np.mean(obs_array),
                    'emergency_flags': [0, 0]
                }
        else:
            # Default state
            return {
                'queue_lengths': np.zeros(5),
                'utilization_rates': np.zeros(5),
                'arrival_rates': np.ones(5),
                'service_rates': np.ones(5),
                'load_rates': np.ones(5),
                'waiting_times': np.zeros(5),
                'throughput': 0,
                'system_load': 0,
                'emergency_flags': [0, 0]
            }
    
    def predict(self, observation, deterministic: bool = True) -> Tuple[np.ndarray, Optional[Dict]]:
        """Predict action based on heuristic rules"""
        state_info = self._extract_state_info(observation)
        
        # Extract key state information
        queue_lengths = np.array(state_info['queue_lengths'])
        utilization_rates = np.array(state_info['utilization_rates'])
        arrival_rates = np.array(state_info['arrival_rates'])
        service_rates = np.array(state_info['service_rates'])
        system_load = state_info['system_load']
        
        # Heuristic decision
        action = self._make_heuristic_decision(
            queue_lengths, utilization_rates, arrival_rates, 
            service_rates, system_load
        )
        
        return action, {'strategy': 'heuristic'}
    
    def _make_heuristic_decision(self, queue_lengths, utilization_rates, 
                               arrival_rates, service_rates, system_load) -> Dict:
        """Make decision based on heuristic rules"""
        
        # 1. Service intensity adjustment strategy
        service_intensities = self._adjust_service_intensities(
            queue_lengths, utilization_rates, service_rates
        )
        
        # 2. Arrival rate adjustment strategy
        arrival_multiplier = self._adjust_arrival_rate(
            utilization_rates, system_load
        )
        
        # 3. Emergency transfer strategy
        emergency_transfers = self._decide_emergency_transfers(
            queue_lengths, utilization_rates
        )
        
        return {
            'service_intensities': service_intensities.astype(np.float32),
            'arrival_multiplier': np.array([arrival_multiplier], dtype=np.float32),
            'emergency_transfers': emergency_transfers.astype(np.int8)
        }
    
    def _adjust_service_intensities(self, queue_lengths, utilization_rates, service_rates):
        """Adjust service intensities"""
        service_intensities = np.ones(5)
        
        target_util = self.config['utilization_target']
        priority_weights = self.config['priority_weights']
        
        for i in range(5):
            current_util = utilization_rates[i]
            queue_len = queue_lengths[i]
            priority = priority_weights[i]
            
            # Adjust based on utilization and queue length
            if current_util > target_util or queue_len > 10:
                # Increase service intensity
                adjustment = min(1.5, 1 + (current_util - target_util) * 2 + queue_len * 0.1)
            elif current_util < target_util * 0.5:
                # Reduce service intensity to save resources
                adjustment = max(0.5, 1 - (target_util - current_util) * 1.5)
            else:
                adjustment = 1.0
            
            # Consider layer priority
            adjustment *= (0.8 + 0.4 * priority)
            
            # Apply boundary constraints
            service_intensities[i] = np.clip(
                adjustment, 
                self.config['service_rate_bounds'][0], 
                self.config['service_rate_bounds'][1]
            )
        
        return service_intensities
    
    def _adjust_arrival_rate(self, utilization_rates, system_load):
        """Adjust system arrival rate"""
        avg_utilization = np.mean(utilization_rates)
        target_util = self.config['utilization_target']
        
        if avg_utilization > self.config['load_balance_threshold']:
            # System overloaded, reduce arrival rate
            multiplier = max(0.5, 1.0 - (avg_utilization - target_util) * 2)
        elif avg_utilization < target_util * 0.6:
            # System load low, can increase arrival rate
            multiplier = min(3.0, 1.0 + (target_util - avg_utilization) * 1.5)
        else:
            # Normal range
            multiplier = 1.0
        
        # Consider overall system load
        if system_load > 0.8:
            multiplier *= 0.8
        elif system_load < 0.3:
            multiplier *= 1.2
        
        return np.clip(
            multiplier,
            self.config['arrival_rate_bounds'][0],
            self.config['arrival_rate_bounds'][1]
        )
    
    def _decide_emergency_transfers(self, queue_lengths, utilization_rates):
        """Decide emergency transfers"""
        emergency_transfers = np.zeros(5, dtype=int)
        emergency_threshold = self.config['emergency_threshold']
        
        for i in range(4):  # Only first 4 layers can transfer down
            current_util = utilization_rates[i]
            current_queue = queue_lengths[i]
            next_util = utilization_rates[i + 1]
            
            # Emergency transfer conditions
            should_transfer = (
                current_util > emergency_threshold or  # Current layer overloaded
                current_queue > 15 or                  # Queue too long
                (current_util > 0.8 and next_util < 0.5)  # Current high load and next layer has capacity
            )
            
            if should_transfer:
                emergency_transfers[i] = 1
        
        return emergency_transfers
    
    def train(self, total_timesteps: int, **kwargs) -> Dict:
        """Training process (heuristic policy can be adaptive by collecting data)"""
        print(f"Running Heuristic Baseline for {total_timesteps} timesteps...")
        
        # Reset training records
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'avg_rewards': [],
            'training_time': [],
            'loss_values': []
        }
        
        start_time = time.time()
        
        state, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        episode_count = 0
        
        for timestep in range(total_timesteps):
            # Predict action
            action, _ = self.predict(state, deterministic=True)
            
            # Execute action
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Record state for adaptation
            state_info = self._extract_state_info(state)
            self._update_adaptive_parameters(state_info, reward)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            if done:
                # Record episode information
                self.training_history['episode_rewards'].append(episode_reward)
                self.training_history['episode_lengths'].append(episode_length)
                
                # Calculate average reward
                if len(self.training_history['episode_rewards']) >= 100:
                    avg_reward = np.mean(self.training_history['episode_rewards'][-100:])
                    self.training_history['avg_rewards'].append(avg_reward)
                
                if episode_count % 100 == 0:
                    print(f"Episode {episode_count}, Timestep {timestep}, Reward: {episode_reward:.2f}")
                
                # Reset environment
                state, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0
                episode_count += 1
        
        end_time = time.time()
        training_time = end_time - start_time
        self.training_history['training_time'].append(training_time)
        
        self.total_timesteps = total_timesteps
        self.episode_count = episode_count
        
        print(f"Heuristic Baseline completed in {training_time:.2f} seconds")
        
        return {
            'total_timesteps': total_timesteps,
            'episodes': episode_count,
            'training_time': training_time,
            'final_reward': self.training_history['episode_rewards'][-1] if self.training_history['episode_rewards'] else 0
        }
    
    def _update_adaptive_parameters(self, state_info, reward):
        """Adaptively adjust parameters based on historical performance"""
        utilization_rates = state_info['utilization_rates']
        avg_utilization = np.mean(utilization_rates)
        
        # Update historical records
        self.utilization_history.append(avg_utilization)
        self.reward_history.append(reward)
        
        # Maintain history window size
        if len(self.utilization_history) > self.history_window:
            self.utilization_history.pop(0)
            self.reward_history.pop(0)
        
        # Adaptive adjustment (simple version)
        if len(self.reward_history) >= self.history_window:
            recent_reward = np.mean(self.reward_history[-5:])
            early_reward = np.mean(self.reward_history[:5])
            
            if recent_reward < early_reward:
                # Performance degradation, adjust thresholds
                self.config['load_balance_threshold'] *= (1 - self.config['adaptive_factor'])
                self.config['utilization_target'] *= (1 - self.config['adaptive_factor'] * 0.5)
            else:
                # Performance improvement, maintain or fine-tune
                self.config['load_balance_threshold'] *= (1 + self.config['adaptive_factor'] * 0.5)
                self.config['utilization_target'] *= (1 + self.config['adaptive_factor'] * 0.3)
        
        # Boundary constraints
        self.config['load_balance_threshold'] = np.clip(self.config['load_balance_threshold'], 0.6, 0.95)
        self.config['utilization_target'] = np.clip(self.config['utilization_target'], 0.5, 0.8)
    
    def save(self, path: str) -> None:
        """Save model"""
        import json
        
        # Convert numpy types to Python native types
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        save_data = {
            'algorithm_name': self.algorithm_name,
            'config': convert_numpy(self.config),
            'total_timesteps': int(self.total_timesteps),
            'episode_count': int(self.episode_count),
            'training_history': convert_numpy(self.training_history),
            'utilization_history': [float(x) for x in self.utilization_history],
            'reward_history': [float(x) for x in self.reward_history]
        }
        
        with open(path, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"Heuristic Baseline saved to: {path}")
    
    def load(self, path: str) -> None:
        """Load model"""
        import json
        
        with open(path, 'r') as f:
            save_data = json.load(f)
        
        self.config.update(save_data['config'])
        self.total_timesteps = save_data.get('total_timesteps', 0)
        self.episode_count = save_data.get('episode_count', 0)
        self.training_history = save_data.get('training_history', {
            'episode_rewards': [],
            'episode_lengths': [],
            'avg_rewards': [],
            'training_time': [],
            'loss_values': []
        })
        self.utilization_history = save_data.get('utilization_history', [])
        self.reward_history = save_data.get('reward_history', [])
        
        print(f"Heuristic Baseline loaded from: {path}")
    
    def get_info(self) -> Dict:
        """Get algorithm information"""
        info = super().get_info()
        info.update({
            'description': 'Heuristic policy based on queuing theory',
            'deterministic': True,
            'adaptive': True,
            'current_thresholds': {
                'load_balance': self.config['load_balance_threshold'],
                'utilization_target': self.config['utilization_target']
            }
        })
        return info
