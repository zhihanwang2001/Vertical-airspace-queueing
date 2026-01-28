"""
Traditional Scheduling Algorithms (Fixed Version)
Traditional scheduling algorithms: FCFS, SJF, Priority, etc. - runs complete timesteps
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import time
import os
import json

from .base_baseline import BaseBaseline


class FCFSBaseline(BaseBaseline):
    """First Come First Served baseline algorithm"""
    
    def __init__(self, env, algorithm_name: str = "FCFS", config: Optional[Dict] = None):
        default_config = {
            'service_rate': 1.0,  # Uniform service intensity
            'arrival_multiplier': 1.0  # Uniform arrival multiplier
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(env, algorithm_name, default_config)
        print(f"FCFS Baseline initialized")
    
    def get_action(self, observation) -> Dict:
        """FCFS strategy: uniform processing for all queues"""
        action = {
            'service_intensities': np.full(5, self.config['service_rate'], dtype=np.float32),
            'arrival_multiplier': np.array([self.config['arrival_multiplier']], dtype=np.float32),
            'emergency_transfers': np.zeros(5, dtype=np.int32)
        }
        return action

    def train(self, total_timesteps: int, **kwargs) -> Dict:
        """FCFS algorithm runs complete timesteps"""
        print(f"Running FCFS for {total_timesteps:,} timesteps...")
        
        start_time = time.time()
        episode_rewards = []
        current_timesteps = 0
        episode = 0
        
        while current_timesteps < total_timesteps:
            obs, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done and current_timesteps < total_timesteps:
                action = self.get_action(obs)
                step_result = self.env.step(action)
                
                if len(step_result) == 5:  # Gymnasium format
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:  # Gym format
                    obs, reward, done, info = step_result
                
                episode_reward += reward
                current_timesteps += 1
                
                if isinstance(done, (list, tuple)):
                    done = any(done)
            
            episode_rewards.append(episode_reward)
            
            if episode % 1000 == 0:
                print(f"Episode {episode}, Timestep {current_timesteps}, Reward: {episode_reward:.2f}")
            
            episode += 1
        
        training_time = time.time() - start_time
        print(f"FCFS completed {current_timesteps:,} timesteps in {training_time:.2f} seconds")
        
        eval_results = self.evaluate()
        
        return {
            'eval_results': eval_results,
            'training_time': training_time,
            'episode_rewards': episode_rewards,
            'total_timesteps': current_timesteps
        }

    def predict(self, observation, deterministic: bool = True):
        """Predict action"""
        action = self.get_action(observation)
        return action, None

    def save(self, path: str) -> None:
        """Save model parameters"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.config, f)
        print(f"FCFS config saved to {path}")

    def load(self, path: str) -> None:
        """Load model parameters"""
        with open(path, 'r') as f:
            self.config = json.load(f)
        print(f"FCFS config loaded from {path}")


class SJFBaseline(BaseBaseline):
    """Shortest Job First baseline algorithm"""
    
    def __init__(self, env, algorithm_name: str = "SJF", config: Optional[Dict] = None):
        default_config = {
            'priority_factor': 2.0,  # Priority factor
            'base_service_rate': 0.5
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(env, algorithm_name, default_config)
        print(f"SJF Baseline initialized")
    
    def get_action(self, observation) -> Dict:
        """SJF strategy: prioritize higher-layer queues (assuming higher-layer jobs are shorter)"""
        if isinstance(observation, dict) and 'queue_lengths' in observation:
            queue_lengths = observation['queue_lengths']
        else:
            queue_lengths = np.ones(5)
        
        # Higher-layer queues get higher service intensity
        service_intensities = np.zeros(5, dtype=np.float32)
        for i in range(5):
            # Higher layer, higher service intensity
            service_intensities[i] = self.config['base_service_rate'] * (5 - i) * self.config['priority_factor']
            service_intensities[i] = np.clip(service_intensities[i], 0.1, 2.0)
        
        action = {
            'service_intensities': service_intensities,
            'arrival_multiplier': np.array([1.0], dtype=np.float32),
            'emergency_transfers': np.zeros(5, dtype=np.int32)
        }
        return action

    def train(self, total_timesteps: int, **kwargs) -> Dict:
        """SJF algorithm runs complete timesteps"""
        print(f"Running SJF for {total_timesteps:,} timesteps...")
        
        start_time = time.time()
        episode_rewards = []
        current_timesteps = 0
        episode = 0
        
        while current_timesteps < total_timesteps:
            obs, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done and current_timesteps < total_timesteps:
                action = self.get_action(obs)
                step_result = self.env.step(action)
                
                if len(step_result) == 5:  # Gymnasium format
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:  # Gym format
                    obs, reward, done, info = step_result
                
                episode_reward += reward
                current_timesteps += 1
                
                if isinstance(done, (list, tuple)):
                    done = any(done)
            
            episode_rewards.append(episode_reward)
            
            if episode % 1000 == 0:
                print(f"Episode {episode}, Timestep {current_timesteps}, Reward: {episode_reward:.2f}")
            
            episode += 1
        
        training_time = time.time() - start_time
        print(f"SJF completed {current_timesteps:,} timesteps in {training_time:.2f} seconds")
        
        eval_results = self.evaluate()
        
        return {
            'eval_results': eval_results,
            'training_time': training_time,
            'episode_rewards': episode_rewards,
            'total_timesteps': current_timesteps
        }

    def predict(self, observation, deterministic: bool = True):
        """Predict action"""
        action = self.get_action(observation)
        return action, None

    def save(self, path: str) -> None:
        """Save model parameters"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.config, f)
        print(f"SJF config saved to {path}")

    def load(self, path: str) -> None:
        """Load model parameters"""
        with open(path, 'r') as f:
            self.config = json.load(f)
        print(f"SJF config loaded from {path}")


class PriorityBaseline(BaseBaseline):
    """Priority scheduling baseline algorithm"""
    
    def __init__(self, env, algorithm_name: str = "Priority", config: Optional[Dict] = None):
        default_config = {
            'priority_weights': [2.0, 1.8, 1.5, 1.2, 1.0],  # Priority weights for each layer
            'emergency_threshold': 0.8,  # Emergency transfer threshold
            'load_balance_factor': 0.3
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(env, algorithm_name, default_config)
        print(f"Priority Baseline initialized")
    
    def get_action(self, observation) -> Dict:
        """Priority strategy: dynamically adjust based on priority and queue state"""
        if isinstance(observation, dict):
            queue_lengths = observation.get('queue_lengths', np.ones(5))
            utilization_rates = observation.get('utilization_rates', np.ones(5) * 0.5)
        else:
            queue_lengths = np.ones(5)
            utilization_rates = np.ones(5) * 0.5
        
        # Calculate dynamic service intensities
        service_intensities = np.zeros(5, dtype=np.float32)
        for i in range(5):
            # Base priority weight
            base_intensity = self.config['priority_weights'][i] * 0.5
            
            # Adjust based on queue length
            if queue_lengths[i] > 2.0:
                base_intensity *= 1.5
            
            # Adjust based on utilization rate
            if utilization_rates[i] > self.config['emergency_threshold']:
                base_intensity *= 1.3
            
            service_intensities[i] = np.clip(base_intensity, 0.1, 2.0)
        
        # Emergency transfer decision
        emergency_transfers = np.zeros(5, dtype=np.int32)
        for i in range(4):  # First 4 layers can transfer down
            if (utilization_rates[i] > self.config['emergency_threshold'] and 
                utilization_rates[i+1] < 0.6):
                emergency_transfers[i] = 1
        
        action = {
            'service_intensities': service_intensities,
            'arrival_multiplier': np.array([1.0], dtype=np.float32),
            'emergency_transfers': emergency_transfers
        }
        return action

    def train(self, total_timesteps: int, **kwargs) -> Dict:
        """Priority algorithm runs complete timesteps"""
        print(f"Running Priority scheduling for {total_timesteps:,} timesteps...")
        
        start_time = time.time()
        episode_rewards = []
        current_timesteps = 0
        episode = 0
        
        while current_timesteps < total_timesteps:
            obs, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done and current_timesteps < total_timesteps:
                action = self.get_action(obs)
                step_result = self.env.step(action)
                
                if len(step_result) == 5:  # Gymnasium format
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:  # Gym format
                    obs, reward, done, info = step_result
                
                episode_reward += reward
                current_timesteps += 1
                
                if isinstance(done, (list, tuple)):
                    done = any(done)
            
            episode_rewards.append(episode_reward)
            
            if episode % 1000 == 0:
                print(f"Episode {episode}, Timestep {current_timesteps}, Reward: {episode_reward:.2f}")
            
            episode += 1
        
        training_time = time.time() - start_time
        print(f"Priority completed {current_timesteps:,} timesteps in {training_time:.2f} seconds")
        
        eval_results = self.evaluate()
        
        return {
            'eval_results': eval_results,
            'training_time': training_time,
            'episode_rewards': episode_rewards,
            'total_timesteps': current_timesteps
        }

    def predict(self, observation, deterministic: bool = True):
        """Predict action"""
        action = self.get_action(observation)
        return action, None

    def save(self, path: str) -> None:
        """Save model parameters"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.config, f)
        print(f"Priority config saved to {path}")

    def load(self, path: str) -> None:
        """Load model parameters"""
        with open(path, 'r') as f:
            self.config = json.load(f)
        print(f"Priority config loaded from {path}")
