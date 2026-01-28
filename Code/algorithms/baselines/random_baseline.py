"""
Random Baseline
Random baseline algorithm - used as comparison benchmark
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import time
import random

from .base_baseline import BaseBaseline


class RandomBaseline(BaseBaseline):
    """Random baseline algorithm implementation"""
    
    def __init__(self, 
                 env,
                 algorithm_name: str = "Random",
                 config: Optional[Dict] = None):
        
        default_config = {
            'seed': None,  # Random seed
            'action_bounds': {
                'service_intensities': (0.1, 2.0),
                'arrival_multiplier': (0.5, 5.0),
                'emergency_transfers': (0, 1)
            }
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(env, algorithm_name, default_config)
        
        # Set random seed
        if self.config['seed'] is not None:
            random.seed(self.config['seed'])
            np.random.seed(self.config['seed'])
        
        print(f"Random Baseline initialized")
    
    def predict(self, observation, deterministic: bool = True) -> Tuple[np.ndarray, Optional[Dict]]:
        """Randomly predict action"""
        # Ignore deterministic parameter, always random
        
        # Generate random action based on environment's action_space
        if hasattr(self.env.action_space, 'sample'):
            action = self.env.action_space.sample()
        else:
            # Manually generate random action
            action = self._generate_random_action()
        
        return action, None
    
    def _generate_random_action(self) -> Dict:
        """Generate random action"""
        bounds = self.config['action_bounds']
        
        action = {}
        
        # service_intensities: 5-dimensional continuous action
        service_min, service_max = bounds['service_intensities']
        action['service_intensities'] = np.random.uniform(
            service_min, service_max, size=5
        )
        
        # arrival_multiplier: 1-dimensional continuous action
        arrival_min, arrival_max = bounds['arrival_multiplier']
        action['arrival_multiplier'] = np.random.uniform(
            arrival_min, arrival_max
        )
        
        # emergency_transfers: 5-dimensional binary action
        action['emergency_transfers'] = np.random.randint(
            0, 2, size=5
        )
        
        return action
    
    def train(self, total_timesteps: int, **kwargs) -> Dict:
        """Training process (actually just random execution)"""
        print(f"Running Random Baseline for {total_timesteps} timesteps...")
        
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
            # Randomly select action
            action, _ = self.predict(state, deterministic=False)
            
            # Execute action
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
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
        
        print(f"Random Baseline completed in {training_time:.2f} seconds")
        
        return {
            'total_timesteps': total_timesteps,
            'episodes': episode_count,
            'training_time': training_time,
            'final_reward': self.training_history['episode_rewards'][-1] if self.training_history['episode_rewards'] else 0
        }
    
    def save(self, path: str) -> None:
        """Save model (random baseline has no parameters to save)"""
        import json
        
        save_data = {
            'algorithm_name': self.algorithm_name,
            'config': self.config,
            'total_timesteps': self.total_timesteps,
            'episode_count': self.episode_count,
            'training_history': self.training_history
        }
        
        with open(path, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"Random Baseline saved to: {path}")
    
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
        
        print(f"Random Baseline loaded from: {path}")
    
    def get_info(self) -> Dict:
        """Get algorithm information"""
        info = super().get_info()
        info.update({
            'description': 'Random policy baseline for comparison',
            'deterministic': False,
            'requires_training': False
        })
        return info
