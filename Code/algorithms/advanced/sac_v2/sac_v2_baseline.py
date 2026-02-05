"""
SAC v2 Baseline for Vertical Stratified Queue System
SAC v2 implementation integrated into existing baseline algorithm framework
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
from .sac_v2_agent import SAC_v2_Agent


class SAC_v2_Baseline:
    """SAC v2 baseline algorithm, compatible with existing framework"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize SAC v2 baseline

        Args:
            config: Configuration parameter dictionary
        """
        # Default configuration
        default_config = {
            # Network configuration
            'hidden_dim': 256,
            'max_action': 1.0,

            # Learning parameters
            'actor_lr': 3e-4,
            'critic_lr': 3e-4,
            'alpha_lr': 3e-4,
            'gamma': 0.99,
            'tau': 0.005,

            # SAC-specific parameters
            'alpha': 0.2,
            'automatic_entropy_tuning': True,
            'target_entropy_scale': 1.0,

            # Replay buffer
            'buffer_size': 100000,
            'batch_size': 256,

            # Training parameters
            'learning_starts': 10000,
            'train_freq': 1,
            'gradient_steps': 1,

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
        
        print("SAC v2 Baseline initialized")
    
    def setup_env(self):
        """Setup environment"""
        base_env = DRLOptimizedQueueEnvFixed()
        self.env = SB3DictWrapper(base_env)
        
        print(f"Environment setup completed")
        print(f"   Observation space: {self.env.observation_space}")
        print(f"   Action space: {self.env.action_space}")
        
        return self.env
    
    def create_agent(self):
        """Create SAC v2 agent"""
        if self.env is None:
            self.setup_env()
        
        self.agent = SAC_v2_Agent(
            state_space=self.env.observation_space,
            action_space=self.env.action_space,
            config=self.config
        )
        
        print("SAC v2 Agent created successfully")
        return self.agent
    
    def train(self, total_timesteps: int, eval_freq: int = 10000, save_freq: int = 50000):
        """
        Train SAC v2 model

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
        tb_log_name = f"SAC_v2_{int(time.time())}"
        writer = SummaryWriter(
            log_dir=os.path.join(self.config['tensorboard_log'], tb_log_name)
        )
        
        print(f"Starting SAC v2 training for {total_timesteps:,} timesteps...")
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
            train_info = self.agent.train()

            if train_info and timestep % 1000 == 0:
                # Log training information
                writer.add_scalar('train/actor_loss', train_info['actor_loss'], timestep)
                writer.add_scalar('train/critic1_loss', train_info['critic1_loss'], timestep)
                writer.add_scalar('train/critic2_loss', train_info['critic2_loss'], timestep)
                writer.add_scalar('train/alpha_loss', train_info['alpha_loss'], timestep)
                writer.add_scalar('train/alpha', train_info['alpha'], timestep)
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
                    current_alpha = self.agent.networks.alpha.item()
                    print(f"Episode {episode:5d} | "
                          f"Timestep {timestep:8d} | "
                          f"Reward: {episode_reward:8.2f} | "
                          f"Length: {episode_length:4d} | "
                          f"Alpha: {current_alpha:.4f} | "
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
                save_path = f"../../../../Models/sac_v2_step_{timestep}.pt"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                self.agent.save(save_path)
                print(f"Model saved at step {timestep}: {save_path}")

        # Training completed
        total_time = time.time() - start_time
        writer.close()

        print(f"SAC v2 training completed!")
        print(f"   Total episodes: {episode}")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Average reward (last 100): {np.mean(self.training_history['episode_rewards'][-100:]):.2f}")

        # Save final model
        final_save_path = "../../../../Models/sac_v2_final.pt"
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
            'system_metrics': []  # SAC-specific metrics can be added here
        }
        
        if verbose:
            print(f"SAC v2 Evaluation Results:")
            print(f"   Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
            print(f"   Mean length: {results['mean_length']:.1f}")
            print(f"   Alpha: {self.agent.networks.alpha.item():.4f}")
        
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
        
        print(f"SAC v2 results saved to: {path_prefix}")
    
    def save(self, path: str):
        """Save model"""
        if self.agent is None:
            raise ValueError("Agent not trained yet!")
        
        self.agent.save(path)
        print(f"SAC v2 model saved to: {path}")
    
    def load(self, path: str):
        """Load model"""
        if self.env is None:
            self.setup_env()
        
        if self.agent is None:
            self.create_agent()
        
        self.agent.load(path)
        print(f"SAC v2 model loaded from: {path}")
        
        return self.agent


def test_sac_v2():
    """Test SAC v2"""
    print("Testing SAC v2...")

    # Create baseline
    baseline = SAC_v2_Baseline()

    # Quick training test
    results = baseline.train(total_timesteps=5000)
    print(f"Training results: {results}")

    # Evaluation test
    eval_results = baseline.evaluate(n_episodes=3)
    print(f"Evaluation results: {eval_results}")

    # Save test
    baseline.save("../../../../Models/sac_v2_test.pt")
    
    print("SAC v2 test completed!")


if __name__ == "__main__":
    test_sac_v2()
