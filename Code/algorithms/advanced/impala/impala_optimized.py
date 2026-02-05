"""
Optimized IMPALA implementation
Specifically optimized for queue system environment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Tuple
import time
import os
from torch.utils.tensorboard import SummaryWriter

# Import environment
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from Code.env.drl_optimized_env_fixed import DRLOptimizedQueueEnvFixed
from Code.env.sb3_dict_wrapper import SB3DictWrapper


class QueueSpecificNetwork(nn.Module):
    """Queue-specific network architecture
    
    Designed for 5-layer queue system:
    - Continuous actions: service_intensities (5) + arrival_multiplier (1) = 6
    - Discrete actions: emergency_transfers (5 binary choices)
    """

    def __init__(self, state_dim: int, config: Dict):
        super().__init__()
        self.state_dim = state_dim
        self.config = config

        hidden_dim = config.get('hidden_dim', 512)
        num_layers = config.get('num_layers', 3)

        # Hierarchical feature extraction
        # Layer 1: Extract basic features
        self.feature_layer1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Layer 2: Extract advanced features
        self.feature_layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Layer 3: Extract abstract features
        self.feature_layer3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU()
        )

        # Continuous action head (service_intensities + arrival_multiplier)
        self.continuous_mean = nn.Linear(hidden_dim // 2, 6)
        self.continuous_logstd = nn.Linear(hidden_dim // 2, 6)

        # Discrete action head (emergency_transfers)
        self.discrete_logits = nn.Linear(hidden_dim // 2, 5)

        # Value head
        self.value_head = nn.Linear(hidden_dim // 2, 1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Special initialization for action heads
        nn.init.orthogonal_(self.continuous_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.continuous_logstd.weight, gain=0.01)
        nn.init.orthogonal_(self.discrete_logits.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(self, state: torch.Tensor):
        """Forward pass"""
        # Hierarchical feature extraction
        x = self.feature_layer1(state)
        x = self.feature_layer2(x)
        features = self.feature_layer3(x)

        # Continuous action distribution parameters
        continuous_mean = torch.tanh(self.continuous_mean(features))
        continuous_logstd = torch.clamp(self.continuous_logstd(features), -5, 2)

        # Discrete action logits
        discrete_logits = self.discrete_logits(features)

        # Value
        value = self.value_head(features)

        return continuous_mean, continuous_logstd, discrete_logits, value

    def get_action_and_value(self, state: torch.Tensor, deterministic: bool = False):
        """Get action and value"""
        continuous_mean, continuous_logstd, discrete_logits, value = self.forward(state)

        if deterministic:
            # Deterministic action
            continuous_action = continuous_mean
            discrete_action = (torch.sigmoid(discrete_logits) > 0.5).float()
        else:
            # Sample continuous action
            continuous_std = torch.exp(continuous_logstd)
            continuous_dist = torch.distributions.Normal(continuous_mean, continuous_std)
            continuous_action = continuous_dist.sample()
            continuous_action = torch.clamp(continuous_action, -1, 1)
            continuous_log_prob = continuous_dist.log_prob(continuous_action).sum(dim=-1, keepdim=True)

            # Sample discrete action
            discrete_dist = torch.distributions.Bernoulli(logits=discrete_logits)
            discrete_action = discrete_dist.sample()
            discrete_log_prob = discrete_dist.log_prob(discrete_action).sum(dim=-1, keepdim=True)

        # Merge log_prob
        total_log_prob = continuous_log_prob + discrete_log_prob

        # Combine actions
        action = torch.cat([continuous_action, discrete_action], dim=-1)

        return action, total_log_prob, value

    def evaluate_action(self, state: torch.Tensor, action: torch.Tensor):
        """Evaluate given state and action"""
        continuous_mean, continuous_logstd, discrete_logits, value = self.forward(state)

        # Separate continuous and discrete actions
        continuous_action = action[:, :6]
        discrete_action = action[:, 6:]

        # Compute log_prob and entropy for continuous actions
        continuous_std = torch.exp(continuous_logstd)
        continuous_dist = torch.distributions.Normal(continuous_mean, continuous_std)
        continuous_log_prob = continuous_dist.log_prob(continuous_action).sum(dim=-1, keepdim=True)
        continuous_entropy = continuous_dist.entropy().sum(dim=-1, keepdim=True)

        # Compute log_prob and entropy for discrete actions
        discrete_dist = torch.distributions.Bernoulli(logits=discrete_logits)
        discrete_log_prob = discrete_dist.log_prob(discrete_action).sum(dim=-1, keepdim=True)
        discrete_entropy = discrete_dist.entropy().sum(dim=-1, keepdim=True)

        # Merge
        total_log_prob = continuous_log_prob + discrete_log_prob
        total_entropy = continuous_entropy + discrete_entropy

        return total_log_prob, value, total_entropy


class OptimizedIMPALAAgent:
    """Optimized IMPALA agent"""

    def __init__(self, state_space, action_space, config: Dict = None):
        # Conservative optimization configuration
        default_config = {
            # Network configuration - increased network capacity
            'hidden_dim': 512,
            'num_layers': 3,

            # Learning parameters - more conservative settings
            'learning_rate': 5e-5,  # Lower learning rate
            'gamma': 0.99,
            'entropy_coeff': 0.02,  # Increase exploration
            'value_loss_coeff': 0.5,
            'gradient_clip': 10.0,  # Stricter gradient clipping

            # V-trace parameters - conservative settings to avoid training collapse
            'rho_bar': 0.8,  # Lower importance weight clipping
            'c_bar': 0.8,    # Lower TD weight clipping

            # Replay buffer - increased capacity and sequence length
            'buffer_size': 50000,  # Larger buffer
            'sequence_length': 32,  # Longer sequences to capture long-term dependencies
            'batch_size': 32,       # Larger batch size

            # Training parameters - more frequent updates
            'learning_starts': 2000,
            'train_freq': 2,  # More frequent training
            'update_freq': 50,

            # Other
            'seed': 42,
            'device': 'auto'
        }

        if config:
            default_config.update(config)
        self.config = default_config

        # Set device
        if self.config['device'] == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.config['device'])

        # Set random seed
        if self.config['seed'] is not None:
            torch.manual_seed(self.config['seed'])
            np.random.seed(self.config['seed'])

        self.state_space = state_space
        self.action_space = action_space

        # Get state dimension
        if hasattr(state_space, 'shape'):
            self.state_dim = state_space.shape[0]
        else:
            # Handle Dict state space
            self.state_dim = sum([space.shape[0] for space in state_space.spaces.values()])

        # Create specialized network
        self.network = QueueSpecificNetwork(
            state_dim=self.state_dim,
            config=self.config
        ).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.config['learning_rate'],
            eps=1e-8  # Increase numerical stability
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100000, eta_min=1e-6
        )

        # Simple experience storage
        self.memory = []
        self.max_memory_size = self.config['buffer_size']

        # Training statistics
        self.training_step = 0
        self.episode_count = 0

        print(f"Optimized IMPALA Agent initialized on {self.device}")
        print(f"   - Conservative V-trace: rho_bar={self.config['rho_bar']}, c_bar={self.config['c_bar']}")
        print(f"   - Lower learning rate: {self.config['learning_rate']}")
        print(f"   - Larger buffer: {self.config['buffer_size']}")
        print(f"   - Longer sequences: {self.config['sequence_length']}")

    def act(self, state, training: bool = True):
        """Select action"""
        if isinstance(state, dict):
            # Convert Dict state to flat vector
            state_vector = []
            for key in ['queue_lengths', 'utilization_rates', 'queue_changes',
                       'load_rates', 'service_rates', 'prev_reward', 'system_metrics']:
                if key in state:
                    value = state[key]
                    if isinstance(value, np.ndarray):
                        state_vector.extend(value.flatten())
                    elif hasattr(value, 'flatten'):
                        state_vector.extend(value.flatten())
                    elif isinstance(value, (list, tuple)):
                        state_vector.extend(value)
                    else:
                        state_vector.append(float(value))
            state = np.array(state_vector, dtype=np.float32)

        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)

        # Convert to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob, value = self.network.get_action_and_value(
                state_tensor, deterministic=not training
            )

            action = action.cpu().numpy()[0]
            log_prob = log_prob.cpu().numpy()[0]
            value = value.cpu().numpy()[0]

        # Store raw action and converted action for training
        self._last_raw_action = action
        self._last_log_prob = log_prob[0]
        self._last_value = value[0]

        # Return raw action vector (let SB3DictWrapper perform conversion)
        return action

    def store_transition(self, state, action, reward, next_state, done):
        """Store experience"""
        if hasattr(self, '_last_raw_action'):
            self.memory.append({
                'state': state,
                'action': self._last_raw_action,
                'reward': reward,
                'next_state': next_state,
                'done': done,
                'log_prob': self._last_log_prob,
                'value': self._last_value
            })

            # Limit memory size
            if len(self.memory) > self.max_memory_size:
                self.memory.pop(0)

    def train(self):
        """Train one step"""
        if len(self.memory) < self.config['sequence_length'] * self.config['batch_size']:
            return None

        if self.training_step % self.config['train_freq'] != 0:
            self.training_step += 1
            return None

        # Simplified V-trace training
        batch_size = min(self.config['batch_size'], len(self.memory) // self.config['sequence_length'])

        total_loss = 0.0
        pg_loss_sum = 0.0
        value_loss_sum = 0.0
        entropy_loss_sum = 0.0

        for _ in range(batch_size):
            # Randomly sample sequence
            start_idx = np.random.randint(0, len(self.memory) - self.config['sequence_length'])
            sequence = self.memory[start_idx:start_idx + self.config['sequence_length']]

            # Build batch data
            states = torch.FloatTensor([self._process_state(exp['state']) for exp in sequence]).to(self.device)
            actions = torch.FloatTensor([exp['action'] for exp in sequence]).to(self.device)
            rewards = torch.FloatTensor([exp['reward'] for exp in sequence]).to(self.device)
            dones = torch.FloatTensor([exp['done'] for exp in sequence]).to(self.device)
            old_log_probs = torch.FloatTensor([exp['log_prob'] for exp in sequence]).to(self.device)

            # Compute current policy outputs
            new_log_probs, values, entropies = self.network.evaluate_action(states, actions)
            values = values.squeeze(-1)
            new_log_probs = new_log_probs.squeeze(-1)
            entropies = entropies.squeeze(-1)

            # Simplified V-trace computation
            with torch.no_grad():
                # Compute importance weights
                importance_weights = torch.exp(new_log_probs - old_log_probs)
                clipped_importance_weights = torch.clamp(importance_weights, max=self.config['rho_bar'])

                # Compute V-trace targets
                next_values = torch.cat([values[1:], torch.zeros(1, device=self.device)])
                td_targets = rewards + self.config['gamma'] * next_values * (1 - dones)
                advantages = clipped_importance_weights * (td_targets - values)

            # Compute losses
            pg_loss = -(new_log_probs * advantages.detach()).mean()
            value_loss = F.mse_loss(values, td_targets.detach())
            entropy_loss = -entropies.mean()

            # Combine losses
            loss = pg_loss + self.config['value_loss_coeff'] * value_loss + self.config['entropy_coeff'] * entropy_loss

            total_loss += loss
            pg_loss_sum += pg_loss.item()
            value_loss_sum += value_loss.item()
            entropy_loss_sum += entropy_loss.item()

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config['gradient_clip'])
        self.optimizer.step()
        self.scheduler.step()

        self.training_step += 1

        return {
            'total_loss': total_loss.item() / batch_size,
            'pg_loss': pg_loss_sum / batch_size,
            'value_loss': value_loss_sum / batch_size,
            'entropy_loss': entropy_loss_sum / batch_size,
            'mean_advantage': advantages.mean().item(),
            'buffer_size': len(self.memory),
            'learning_rate': self.scheduler.get_last_lr()[0]
        }

    def _process_state(self, state):
        """Process state to vector format"""
        if isinstance(state, dict):
            state_vector = []
            for key in ['queue_lengths', 'utilization_rates', 'queue_changes',
                       'load_rates', 'service_rates', 'prev_reward', 'system_metrics']:
                if key in state:
                    value = state[key]
                    if isinstance(value, np.ndarray):
                        state_vector.extend(value.flatten())
                    elif hasattr(value, 'flatten'):
                        state_vector.extend(value.flatten())
                    elif isinstance(value, (list, tuple)):
                        state_vector.extend(value)
                    else:
                        state_vector.append(float(value))
            return np.array(state_vector, dtype=np.float32)
        return np.array(state, dtype=np.float32)

    def save(self, filepath: str):
        """Save model"""
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'config': self.config,
            'training_step': self.training_step
        }, filepath)

    def load(self, filepath: str):
        """Load model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.training_step = checkpoint['training_step']


class OptimizedIMPALABaseline:
    """Optimized IMPALA baseline algorithm"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.agent = None
        self.env = None
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'avg_rewards': [],
            'loss_values': [],
            'training_steps': []
        }

        print("Optimized IMPALA Baseline initialized with queue-specific optimizations")

    def setup_env(self):
        """Setup environment"""
        base_env = DRLOptimizedQueueEnvFixed()
        self.env = SB3DictWrapper(base_env)

        print(f"Environment setup completed")
        print(f"   Observation space: {self.env.observation_space}")
        print(f"   Action space: {self.env.action_space}")

        return self.env

    def create_agent(self):
        """Create optimized IMPALA agent"""
        if self.env is None:
            self.setup_env()

        self.agent = OptimizedIMPALAAgent(
            state_space=self.env.observation_space,
            action_space=self.env.action_space,
            config=self.config
        )

        print("Optimized IMPALA Agent created successfully")
        return self.agent

    def train(self, total_timesteps: int, eval_freq: int = 10000, save_freq: int = 50000):
        """Train optimized IMPALA model"""
        if self.agent is None:
            self.create_agent()

        # Create TensorBoard writer
        tb_log_name = f"IMPALA_Optimized_{int(time.time())}"
        writer = SummaryWriter(log_dir=f"./tensorboard_logs/{tb_log_name}")

        print(f"Starting Optimized IMPALA training for {total_timesteps:,} timesteps...")
        print(f"   TensorBoard log: {tb_log_name}")
        print(f"   Key optimizations:")
        print(f"   - Mixed action space support")
        print(f"   - Queue-specific network architecture")
        print(f"   - Conservative V-trace parameters")
        print(f"   - Lower learning rate with scheduling")

        # Training loop
        episode = 0
        timestep = 0
        episode_reward = 0.0
        episode_length = 0

        state, _ = self.env.reset()
        start_time = time.time()

        while timestep < total_timesteps:
            # Select action
            action = self.agent.act(state, training=True)

            # Execute action
            try:
                step_result = self.env.step(action)
                if len(step_result) == 5:
                    next_state, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
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
            if timestep >= self.config.get('learning_starts', 2000):
                train_info = self.agent.train()

                if train_info and timestep % 1000 == 0:
                    # Log training information
                    writer.add_scalar('train/total_loss', train_info['total_loss'], timestep)
                    writer.add_scalar('train/pg_loss', train_info['pg_loss'], timestep)
                    writer.add_scalar('train/value_loss', train_info['value_loss'], timestep)
                    writer.add_scalar('train/entropy_loss', train_info['entropy_loss'], timestep)
                    writer.add_scalar('train/mean_advantage', train_info['mean_advantage'], timestep)
                    writer.add_scalar('train/buffer_size', train_info['buffer_size'], timestep)
                    writer.add_scalar('train/learning_rate', train_info['learning_rate'], timestep)

            # Episode end handling
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
                if episode % 100 == 0:
                    elapsed_time = time.time() - start_time
                    recent_rewards = self.training_history['episode_rewards'][-100:] if len(self.training_history['episode_rewards']) >= 100 else self.training_history['episode_rewards']
                    avg_recent = np.mean(recent_rewards) if recent_rewards else 0

                    print(f"Episode {episode:5d} | "
                          f"Timestep {timestep:8d} | "
                          f"Reward: {episode_reward:8.2f} | "
                          f"Avg(100): {avg_recent:8.2f} | "
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
                      f"Mean reward: {eval_results['mean_reward']:.2f} +/- {eval_results['std_reward']:.2f}")

            # Save model
            if save_freq > 0 and timestep % save_freq == 0 and timestep > 0:
                save_path = f"../../../../Models/impala_optimized_step_{timestep}.pt"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                self.agent.save(save_path)
                print(f"Model saved at step {timestep}: {save_path}")

        # Training completed
        total_time = time.time() - start_time
        writer.close()

        print(f"Optimized IMPALA training completed!")
        print(f"   Total episodes: {episode}")
        print(f"   Total time: {total_time:.2f}s")
        final_avg = np.mean(self.training_history['episode_rewards'][-100:]) if len(self.training_history['episode_rewards']) >= 100 else np.mean(self.training_history['episode_rewards']) if self.training_history['episode_rewards'] else 0
        print(f"   Average reward (last 100): {final_avg:.2f}")

        # Save final model
        final_save_path = "../../../../Models/impala_optimized_final.pt"
        os.makedirs(os.path.dirname(final_save_path), exist_ok=True)
        self.agent.save(final_save_path)

        return {
            'episodes': episode,
            'total_timesteps': timestep,
            'final_reward': final_avg,
            'training_time': total_time
        }

    def evaluate(self, n_episodes: int = 10, deterministic: bool = True, verbose: bool = True):
        """Evaluate model performance"""
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
            'episode_lengths': episode_lengths
        }

        if verbose:
            print(f"Optimized IMPALA Evaluation Results:")
            print(f"   Mean reward: {results['mean_reward']:.2f} +/- {results['std_reward']:.2f}")
            print(f"   Mean length: {results['mean_length']:.1f}")

        return results

    def save_results(self, path_prefix: str):
        """Save training results"""
        os.makedirs(os.path.dirname(path_prefix) if os.path.dirname(path_prefix) else ".", exist_ok=True)

        import json
        with open(f"{path_prefix}_history.json", 'w') as f:
            serializable_history = {}
            for key, value in self.training_history.items():
                if isinstance(value, list):
                    serializable_history[key] = [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in value]
                else:
                    serializable_history[key] = value
            json.dump(serializable_history, f, indent=2)

        print(f"Optimized IMPALA results saved to: {path_prefix}")

    def save(self, path: str):
        """Save model"""
        if self.agent is None:
            raise ValueError("Agent not trained yet!")

        self.agent.save(path)
        print(f"Optimized IMPALA model saved to: {path}")

    def load(self, path: str):
        """Load model"""
        if self.env is None:
            self.setup_env()

        if self.agent is None:
            self.create_agent()

        self.agent.load(path)
        print(f"Optimized IMPALA model loaded from: {path}")

        return self.agent


def test_optimized_impala():
    """Test optimized IMPALA"""
    print("Testing Optimized IMPALA...")

    baseline = OptimizedIMPALABaseline()

    # Quick training test
    results = baseline.train(total_timesteps=5000)
    print(f"Training results: {results}")

    # Evaluation test
    eval_results = baseline.evaluate(n_episodes=3)
    print(f"Evaluation results: {eval_results}")

    baseline.save("../../../../Models/impala_optimized_test.pt")
    print("Optimized IMPALA test completed!")


if __name__ == "__main__":
    test_optimized_impala()
