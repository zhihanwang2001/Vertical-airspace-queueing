"""
IMPALA Baseline for Vertical Stratified Queue System
IMPALA implementation integrated into existing baseline algorithm framework
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import time
from typing import Dict, Any, Optional

from env.drl_optimized_env_fixed import DRLOptimizedQueueEnvFixed
from baselines.space_utils import SB3DictWrapper
from .impala_agent import IMPALAAgent


class IMPALABaseline:
    """IMPALA baseline algorithm, compatible with existing framework"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize IMPALA baseline

        Args:
            config: Configuration parameter dictionary
        """
        # Optimized configuration - consistent with impala_agent.py
        default_config = {
            # Network configuration
            'hidden_dim': 512,
            'num_layers': 2,

            # Learning parameters - further reduced learning rate to prevent late-stage collapse
            'learning_rate': 3e-5,      # Optimization v2: 5e-5 → 3e-5 (consistent with agent)
            'gamma': 0.99,
            'entropy_coeff': 0.01,
            'value_loss_coeff': 0.5,
            'gradient_clip': 20.0,      # Optimization: 40.0 → 20.0 (consistent with agent)

            # V-trace parameters - extremely conservative to avoid importance sampling explosion
            'rho_bar': 0.7,             # Optimization v2: 0.9 → 0.7 (consistent with agent)
            'c_bar': 0.7,               # Optimization v2: 0.9 → 0.7 (consistent with agent)

            # Replay buffer - reduced buffer to decrease policy staleness
            'buffer_size': 30000,       # Optimization v2: 50000 → 30000 (consistent with agent)
            'sequence_length': 10,      # Optimization: 20 → 10 (consistent with agent)
            'batch_size': 32,           # Optimization: 16 → 32 (consistent with agent)

            # Training parameters - more frequent updates but delayed start
            'learning_starts': 2000,    # Optimization: 1000 → 2000 (consistent with agent)
            'train_freq': 2,            # Optimization: 4 → 2 (consistent with agent)
            'update_freq': 100,

            # TensorBoard logging
            'tensorboard_log': "./tensorboard_logs/",
            'verbose': 1,
            'seed': 42
        }
        
        if config:
            default_config.update(config)
        
        self.config = default_config
        self.agent = None
        self.env = None
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'avg_rewards': [],
            'loss_values': [],
            'training_steps': []
        }
        
        print("IMPALA Baseline initialized")
    
    def setup_env(self):
        """Setup environment"""
        base_env = DRLOptimizedQueueEnvFixed()
        self.env = SB3DictWrapper(base_env)
        
        print(f"Environment setup completed")
        print(f"   Observation space: {self.env.observation_space}")
        print(f"   Action space: {self.env.action_space}")
        
        return self.env
    
    def create_agent(self):
        """Create IMPALA agent"""
        if self.env is None:
            self.setup_env()
        
        self.agent = IMPALAAgent(
            state_space=self.env.observation_space,
            action_space=self.env.action_space,
            config=self.config
        )
        
        print("IMPALA Agent created successfully")
        return self.agent
    
    def train(self, total_timesteps: int, eval_freq: int = 10000, save_freq: int = 50000):
        """
        Train IMPALA model

        Args:
            total_timesteps: Total training steps
            eval_freq: Evaluation frequency
            save_freq: Save frequency

        Returns:
            Training history dictionary
        """
        if self.agent is None:
            self.create_agent()

        # Create TensorBoard writer
        tb_log_name = f"IMPALA_{int(time.time())}"
        writer = SummaryWriter(
            log_dir=os.path.join(self.config['tensorboard_log'], tb_log_name)
        )

        print(f"Starting IMPALA training for {total_timesteps:,} timesteps...")
        print(f"   TensorBoard log: {tb_log_name}")

        # Training variables
        episode = 0
        timestep = 0
        episode_reward = 0.0
        episode_length = 0

        # Reset environment
        state, _ = self.env.reset()
        
        start_time = time.time()

        while timestep < total_timesteps:
            # Select action
            action = self.agent.act(state, training=True)

            # Execute action
            try:
                step_result = self.env.step(action)
                if len(step_result) == 5:  # Gymnasium format
                    next_state, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:  # Gym format
                    next_state, reward, done, info = step_result
            except Exception as e:
                print(f"Environment step error: {e}")
                break

            # Store experience
            self.agent.store_transition(state, action, reward, next_state, done)

            # Update statistics
            episode_reward += reward
            episode_length += 1
            timestep += 1

            # Train agent
            if timestep >= self.config['learning_starts']:
                train_info = self.agent.train()

                if train_info and timestep % 1000 == 0:
                    # Log training information
                    writer.add_scalar('train/total_loss', train_info['total_loss'], timestep)
                    writer.add_scalar('train/pg_loss', train_info['pg_loss'], timestep)
                    writer.add_scalar('train/value_loss', train_info['value_loss'], timestep)
                    writer.add_scalar('train/entropy_loss', train_info['entropy_loss'], timestep)
                    writer.add_scalar('train/mean_advantage', train_info['mean_advantage'], timestep)
                    writer.add_scalar('train/buffer_size', train_info['buffer_size'], timestep)

            # Check if episode ended
            if done:
                # Log episode information
                self.training_history['episode_rewards'].append(episode_reward)
                self.training_history['episode_lengths'].append(episode_length)

                # TensorBoard logging
                writer.add_scalar('train/episode_reward', episode_reward, episode)
                writer.add_scalar('train/episode_length', episode_length, episode)

                # Calculate moving average
                if len(self.training_history['episode_rewards']) >= 100:
                    avg_reward = np.mean(self.training_history['episode_rewards'][-100:])
                    self.training_history['avg_rewards'].append(avg_reward)
                    writer.add_scalar('train/avg_reward_100', avg_reward, episode)

                # Print progress
                if self.config['verbose'] and episode % 100 == 0:
                    elapsed_time = time.time() - start_time
                    print(f"Episode {episode:5d} | "
                          f"Timestep {timestep:8d} | "
                          f"Reward: {episode_reward:8.2f} | "
                          f"Length: {episode_length:4d} | "
                          f"Time: {elapsed_time:.1f}s")

                # Reset episode
                episode += 1
                episode_reward = 0.0
                episode_length = 0
                state, _ = self.env.reset()
            else:
                state = next_state

            # Evaluation
            if eval_freq > 0 and timestep % eval_freq == 0 and timestep > 0:
                eval_results = self.evaluate(n_episodes=5, deterministic=True, verbose=False)
                writer.add_scalar('eval/mean_reward', eval_results['mean_reward'], timestep)
                writer.add_scalar('eval/std_reward', eval_results['std_reward'], timestep)

                print(f"Evaluation at step {timestep}: "
                      f"Mean reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")

            # Save model
            if save_freq > 0 and timestep % save_freq == 0 and timestep > 0:
                save_path = f"../../../../Models/impala_step_{timestep}.pt"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                self.agent.save(save_path)
                print(f"Model saved at step {timestep}: {save_path}")

        # Training completed
        total_time = time.time() - start_time
        writer.close()

        print(f"IMPALA training completed!")
        print(f"   Total episodes: {episode}")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Average reward (last 100): {np.mean(self.training_history['episode_rewards'][-100:]):.2f}")

        # Save final model
        final_save_path = "../../../../Models/impala_final.pt"
        os.makedirs(os.path.dirname(final_save_path), exist_ok=True)
        self.agent.save(final_save_path)
        
        return {
            'episodes': episode,
            'total_timesteps': timestep,
            'final_reward': np.mean(self.training_history['episode_rewards'][-10:]) if self.training_history['episode_rewards'] else 0,
            'training_time': total_time
        }
    
    def evaluate(self, n_episodes: int = 10, deterministic: bool = True, verbose: bool = True):
        """
        Evaluate model performance

        Args:
            n_episodes: Number of evaluation episodes
            deterministic: Whether to use deterministic policy
            verbose: Whether to print detailed information

        Returns:
            Evaluation results dictionary
        """
        if self.agent is None:
            raise ValueError("Agent not initialized. Please train first.")
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(n_episodes):
            state, _ = self.env.reset()
            episode_reward = 0.0
            episode_length = 0
            done = False
            
            while not done:
                action = self.agent.act(state, training=False)
                
                try:
                    step_result = self.env.step(action)
                    if len(step_result) == 5:
                        next_state, reward, terminated, truncated, info = step_result
                        done = terminated or truncated
                    else:
                        next_state, reward, done, info = step_result
                except Exception as e:
                    print(f"Evaluation error: {e}")
                    break
                
                episode_reward += reward
                episode_length += 1
                state = next_state

                # Prevent infinite loop
                if episode_length >= 1000:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            if verbose:
                print(f"  Episode {episode+1}/{n_episodes}: Reward = {episode_reward:.2f}, Length = {episode_length}")
        
        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'system_metrics': []  # IMPALA-specific metrics can be added here
        }
        
        if verbose:
            print(f"IMPALA Evaluation Results:")
            print(f"   Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
            print(f"   Mean length: {results['mean_length']:.1f}")
        
        return results
    
    def save_results(self, path_prefix: str):
        """Save training results"""
        os.makedirs(os.path.dirname(path_prefix) if os.path.dirname(path_prefix) else ".", exist_ok=True)

        # Save training history
        import json
        with open(f"{path_prefix}_history.json", 'w') as f:
            # Convert numpy types to Python native types
            serializable_history = {}
            for key, value in self.training_history.items():
                if isinstance(value, list):
                    serializable_history[key] = [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in value]
                else:
                    serializable_history[key] = value
            
            json.dump(serializable_history, f, indent=2)
        
        print(f"IMPALA results saved to: {path_prefix}")
    
    def save(self, path: str):
        """Save model"""
        if self.agent is None:
            raise ValueError("Agent not trained yet!")

        self.agent.save(path)
        print(f"IMPALA model saved to: {path}")

    def load(self, path: str):
        """Load model"""
        if self.env is None:
            self.setup_env()
        
        if self.agent is None:
            self.create_agent()
        
        self.agent.load(path)
        print(f"IMPALA model loaded from: {path}")
        
        return self.agent


def test_impala():
    """Test IMPALA"""
    print("Testing IMPALA...")

    # Create baseline
    baseline = IMPALABaseline()

    # Quick training test
    results = baseline.train(total_timesteps=5000)
    print(f"Training results: {results}")

    # Evaluation test
    eval_results = baseline.evaluate(n_episodes=3)
    print(f"Evaluation results: {eval_results}")

    # Save test
    baseline.save("../../../../Models/impala_test.pt")

    print("IMPALA test completed!")


if __name__ == "__main__":
    test_impala()
