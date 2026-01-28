"""
HCA2C (Hierarchical Capacity-Aware A2C) Agent

Main algorithm implementation combining:
- Hierarchical policy decomposition
- Capacity-aware action clipping
- Inter-layer coordination
- A2C-style policy gradient updates
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from typing import Dict, List, Tuple, Optional, Union
import time
from collections import deque

from .networks import HierarchicalActorCritic
from .wrapper import HierarchicalEnvWrapper
from .clipper import CapacityAwareClipper


class RolloutBuffer:
    """
    Rollout buffer for storing trajectories

    Stores hierarchical observations, actions, rewards, and values
    for policy gradient updates.
    """

    def __init__(self, buffer_size: int, global_dim: int, layer_dim: int,
                 action_dim: int, n_layers: int = 5, device: str = 'cpu'):
        self.buffer_size = buffer_size
        self.global_dim = global_dim
        self.layer_dim = layer_dim
        self.action_dim = action_dim
        self.n_layers = n_layers
        self.device = device

        self.reset()

    def reset(self):
        """Reset buffer"""
        self.global_states = []
        self.layer_states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.pos = 0

    def add(self, global_state: np.ndarray, layer_states: np.ndarray,
            action: Dict, reward: float, value: float, log_prob: float, done: bool):
        """Add a transition to the buffer"""
        self.global_states.append(global_state)
        self.layer_states.append(layer_states)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.pos += 1

    def get(self) -> Dict[str, torch.Tensor]:
        """Get all data as tensors"""
        return {
            'global_states': torch.tensor(np.array(self.global_states), dtype=torch.float32, device=self.device),
            'layer_states': torch.tensor(np.array(self.layer_states), dtype=torch.float32, device=self.device),
            'rewards': torch.tensor(np.array(self.rewards), dtype=torch.float32, device=self.device),
            'values': torch.tensor(np.array(self.values), dtype=torch.float32, device=self.device),
            'log_probs': torch.tensor(np.array(self.log_probs), dtype=torch.float32, device=self.device),
            'dones': torch.tensor(np.array(self.dones), dtype=torch.float32, device=self.device),
            'actions': self._stack_actions()
        }

    def _stack_actions(self) -> Dict[str, torch.Tensor]:
        """Stack action dictionaries into tensors"""
        stacked = {}
        keys = self.actions[0].keys()

        for key in keys:
            values = [a[key] for a in self.actions]
            stacked[key] = torch.tensor(np.array(values), dtype=torch.float32, device=self.device)

        return stacked

    def __len__(self):
        return self.pos


class HCA2C:
    """
    Hierarchical Capacity-Aware A2C Algorithm

    Key features:
    1. Hierarchical policy: Global + Layer-specific policies
    2. Capacity-aware clipping: Dynamic action bounds
    3. Inter-layer coordination: Message passing between layers
    4. A2C updates: Advantage-based policy gradients
    """

    def __init__(self,
                 env,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 n_steps: int = 32,
                 ent_coef: float = 0.01,
                 vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 global_hidden_dim: int = 256,
                 layer_hidden_dim: int = 128,
                 device: str = 'auto',
                 seed: int = None,
                 verbose: int = 1):
        """
        Initialize HCA2C agent

        Args:
            env: Base environment (will be wrapped with HierarchicalEnvWrapper)
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            gae_lambda: GAE lambda for advantage estimation
            n_steps: Number of steps per rollout
            ent_coef: Entropy coefficient for exploration
            vf_coef: Value function coefficient
            max_grad_norm: Maximum gradient norm for clipping
            global_hidden_dim: Hidden dimension for global policy
            layer_hidden_dim: Hidden dimension for layer policies
            device: Device to use ('auto', 'cpu', 'cuda')
            seed: Random seed
            verbose: Verbosity level
        """
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Set seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.verbose = verbose

        # Wrap environment
        if not isinstance(env, HierarchicalEnvWrapper):
            self.env = HierarchicalEnvWrapper(env)
        else:
            self.env = env

        # Get dimensions
        self.global_dim = self.env.global_state_dim
        self.layer_dim = self.env.layer_state_dim
        self.n_layers = self.env.n_layers
        self.action_dim = 11  # 5 service + 1 arrival + 5 transfer

        # Hyperparameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.n_steps = n_steps
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm

        # Create networks
        self.policy = HierarchicalActorCritic(
            global_state_dim=self.global_dim,
            layer_state_dim=self.layer_dim,
            global_hidden_dim=global_hidden_dim,
            layer_hidden_dim=layer_hidden_dim,
            n_layers=self.n_layers
        ).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=learning_rate,
            eps=1e-5
        )

        # Capacity-aware clipper
        self.clipper = CapacityAwareClipper(
            capacities=self.env.capacities,
            n_layers=self.n_layers
        )

        # Rollout buffer
        self.buffer = RolloutBuffer(
            buffer_size=n_steps,
            global_dim=self.global_dim,
            layer_dim=self.layer_dim,
            action_dim=self.action_dim,
            n_layers=self.n_layers,
            device=self.device
        )

        # Training statistics
        self.num_timesteps = 0
        self.num_episodes = 0
        self._last_obs = None
        self._episode_reward = 0
        self._episode_length = 0

        # Logging
        self.ep_rewards = deque(maxlen=100)
        self.ep_lengths = deque(maxlen=100)

    def predict(self, observation: Dict[str, np.ndarray],
                deterministic: bool = False) -> Tuple[np.ndarray, Dict]:
        """
        Predict action given observation

        Args:
            observation: Hierarchical observation dict
            deterministic: If True, use mean action

        Returns:
            action: Flat action array for environment
            info: Additional information
        """
        with torch.no_grad():
            global_state = torch.tensor(
                observation['global'], dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            layer_states = torch.tensor(
                observation['layers'], dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            actions, log_probs, values = self.policy(
                global_state, layer_states, deterministic=deterministic
            )

        # Convert to numpy
        action_dict = {k: v.cpu().numpy().squeeze(0) for k, v in actions.items()}

        # Clip actions based on state
        clipped_action = self.clipper.clip_action(action_dict, observation)

        # Convert to flat action for environment
        flat_action = self._dict_to_flat_action(clipped_action)

        info = {
            'value': values.cpu().numpy().item(),
            'log_prob': log_probs.cpu().numpy().item(),
            'raw_action': action_dict,
            'clipped_action': clipped_action
        }

        return flat_action, info

    def _dict_to_flat_action(self, action_dict: Dict) -> np.ndarray:
        """Convert action dictionary to flat array"""
        # Service intensities: [0.1, 2.0] -> [-1, 1]
        service = (action_dict['service_intensities'] - 0.1) / 1.9 * 2 - 1

        # Arrival multiplier: [0.5, 5.0] -> [-1, 1]
        arrival = (action_dict['arrival_multiplier'] - 0.5) / 4.5 * 2 - 1

        # Emergency transfers: {0, 1} -> [-1, 1]
        transfers = action_dict['emergency_transfers'] * 2 - 1

        return np.concatenate([service, arrival.flatten(), transfers]).astype(np.float32)

    def collect_rollouts(self) -> bool:
        """
        Collect rollouts for n_steps

        Returns:
            True if rollout collection was successful
        """
        self.buffer.reset()

        if self._last_obs is None:
            self._last_obs, _ = self.env.reset()
            self._episode_reward = 0
            self._episode_length = 0

        for _ in range(self.n_steps):
            # Get action
            with torch.no_grad():
                global_state = torch.tensor(
                    self._last_obs['global'], dtype=torch.float32, device=self.device
                ).unsqueeze(0)

                layer_states = torch.tensor(
                    self._last_obs['layers'], dtype=torch.float32, device=self.device
                ).unsqueeze(0)

                actions, log_probs, values = self.policy(
                    global_state, layer_states, deterministic=False
                )

            # Convert to numpy and clip
            action_dict = {k: v.cpu().numpy().squeeze(0) for k, v in actions.items()}
            clipped_action = self.clipper.clip_action(action_dict, self._last_obs)
            flat_action = self._dict_to_flat_action(clipped_action)

            # Step environment
            new_obs, reward, terminated, truncated, info = self.env.step(flat_action)
            done = terminated or truncated

            self.num_timesteps += 1
            self._episode_reward += reward
            self._episode_length += 1

            # Store transition
            self.buffer.add(
                global_state=self._last_obs['global'],
                layer_states=self._last_obs['layers'],
                action=clipped_action,
                reward=reward,
                value=values.cpu().numpy().item(),
                log_prob=log_probs.cpu().numpy().item(),
                done=done
            )

            self._last_obs = new_obs

            if done:
                self.ep_rewards.append(self._episode_reward)
                self.ep_lengths.append(self._episode_length)
                self.num_episodes += 1

                self._last_obs, _ = self.env.reset()
                self._episode_reward = 0
                self._episode_length = 0

        return True

    def compute_returns_and_advantages(self, last_value: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute returns and advantages using GAE

        Args:
            last_value: Value estimate for the last state

        Returns:
            returns: Discounted returns
            advantages: GAE advantages
        """
        data = self.buffer.get()
        rewards = data['rewards']
        values = data['values']
        dones = data['dones']

        n_steps = len(rewards)
        advantages = torch.zeros(n_steps, device=self.device)
        returns = torch.zeros(n_steps, device=self.device)

        last_gae = 0
        last_return = last_value

        for t in reversed(range(n_steps)):
            if t == n_steps - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            # TD error
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]

            # GAE
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae

            # Returns
            returns[t] = last_return = rewards[t] + self.gamma * (1 - dones[t]) * last_return

        return returns, advantages

    def train_step(self) -> Dict[str, float]:
        """
        Perform one training step

        Returns:
            Dictionary of training metrics
        """
        # Get last value for GAE
        with torch.no_grad():
            global_state = torch.tensor(
                self._last_obs['global'], dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            layer_states = torch.tensor(
                self._last_obs['layers'], dtype=torch.float32, device=self.device
            ).unsqueeze(0)

            _, _, last_value = self.policy(global_state, layer_states, deterministic=True)
            last_value = last_value.cpu().numpy().item()

        # Compute returns and advantages
        returns, advantages = self.compute_returns_and_advantages(last_value)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Get data
        data = self.buffer.get()
        global_states = data['global_states']
        layer_states = data['layer_states']
        old_log_probs = data['log_probs']
        actions = data['actions']

        # Forward pass
        # Reconstruct action dict for evaluation
        action_dict = {
            'service_intensities': actions['service_intensities'],
            'arrival_multiplier': actions['arrival_multiplier'],
            'transfer_decisions': actions['emergency_transfers'].float() * 2 - 1  # Convert back
        }

        log_probs, values, entropy = self.policy.evaluate_actions(
            global_states, layer_states, action_dict
        )

        values = values.squeeze(-1)

        # Policy loss (A2C style)
        policy_loss = -(log_probs * advantages.detach()).mean()

        # Value loss
        value_loss = F.mse_loss(values, returns)

        # Entropy loss
        entropy_loss = -entropy.mean()

        # Total loss
        loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)

        self.optimizer.step()

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': -entropy_loss.item(),
            'total_loss': loss.item()
        }

    def learn(self, total_timesteps: int, callback=None, log_interval: int = 100,
              progress_bar: bool = True) -> 'HCA2C':
        """
        Train the agent

        Args:
            total_timesteps: Total number of timesteps to train
            callback: Optional callback function
            log_interval: Logging interval (in updates)
            progress_bar: Whether to show progress bar

        Returns:
            self
        """
        num_updates = total_timesteps // self.n_steps
        start_time = time.time()

        if self.verbose > 0:
            print(f"Starting HCA2C training for {total_timesteps:,} timesteps...")
            print(f"  Device: {self.device}")
            print(f"  N_steps: {self.n_steps}")
            print(f"  Num updates: {num_updates:,}")

        try:
            from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn
            use_rich = progress_bar
        except ImportError:
            use_rich = False

        if use_rich:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
            ) as progress:
                task = progress.add_task("Training", total=num_updates)

                for update in range(1, num_updates + 1):
                    # Collect rollouts
                    self.collect_rollouts()

                    # Train
                    metrics = self.train_step()

                    # Logging
                    if update % log_interval == 0 and self.verbose > 0:
                        elapsed = time.time() - start_time
                        fps = self.num_timesteps / elapsed

                        mean_reward = np.mean(self.ep_rewards) if self.ep_rewards else 0
                        mean_length = np.mean(self.ep_lengths) if self.ep_lengths else 0

                        print(f"\n  Update {update}/{num_updates}")
                        print(f"    Timesteps: {self.num_timesteps:,}")
                        print(f"    Episodes: {self.num_episodes}")
                        print(f"    Mean reward: {mean_reward:.2f}")
                        print(f"    Mean length: {mean_length:.1f}")
                        print(f"    FPS: {fps:.0f}")
                        print(f"    Policy loss: {metrics['policy_loss']:.4f}")
                        print(f"    Value loss: {metrics['value_loss']:.4f}")
                        print(f"    Entropy: {metrics['entropy']:.4f}")

                    progress.update(task, advance=1)

                    # Callback
                    if callback is not None:
                        callback(self)
        else:
            for update in range(1, num_updates + 1):
                # Collect rollouts
                self.collect_rollouts()

                # Train
                metrics = self.train_step()

                # Logging
                if update % log_interval == 0 and self.verbose > 0:
                    elapsed = time.time() - start_time
                    fps = self.num_timesteps / elapsed

                    mean_reward = np.mean(self.ep_rewards) if self.ep_rewards else 0

                    print(f"Update {update}/{num_updates} | "
                          f"Timesteps: {self.num_timesteps:,} | "
                          f"Mean reward: {mean_reward:.2f} | "
                          f"FPS: {fps:.0f}")

                # Callback
                if callback is not None:
                    callback(self)

        if self.verbose > 0:
            total_time = time.time() - start_time
            print(f"\nTraining completed!")
            print(f"  Total time: {total_time:.1f}s")
            print(f"  Total timesteps: {self.num_timesteps:,}")
            print(f"  Total episodes: {self.num_episodes}")
            print(f"  Final mean reward: {np.mean(self.ep_rewards) if self.ep_rewards else 0:.2f}")

        return self

    def save(self, path: str):
        """Save model to file"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'num_timesteps': self.num_timesteps,
            'num_episodes': self.num_episodes,
            'config': {
                'learning_rate': self.learning_rate,
                'gamma': self.gamma,
                'gae_lambda': self.gae_lambda,
                'n_steps': self.n_steps,
                'ent_coef': self.ent_coef,
                'vf_coef': self.vf_coef,
                'max_grad_norm': self.max_grad_norm,
                'global_dim': self.global_dim,
                'layer_dim': self.layer_dim,
                'n_layers': self.n_layers
            }
        }, path)

        if self.verbose > 0:
            print(f"Model saved to: {path}")

    def load(self, path: str):
        """Load model from file"""
        checkpoint = torch.load(path, map_location=self.device)

        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.num_timesteps = checkpoint['num_timesteps']
        self.num_episodes = checkpoint['num_episodes']

        if self.verbose > 0:
            print(f"Model loaded from: {path}")

    def evaluate(self, n_episodes: int = 10, deterministic: bool = True,
                 verbose: bool = True) -> Dict[str, float]:
        """
        Evaluate the agent

        Args:
            n_episodes: Number of evaluation episodes
            deterministic: Use deterministic actions
            verbose: Print progress

        Returns:
            Dictionary of evaluation metrics
        """
        episode_rewards = []
        episode_lengths = []
        crash_count = 0

        for ep in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            ep_reward = 0
            ep_length = 0

            while not done:
                action, _ = self.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                ep_reward += reward
                ep_length += 1

                if ep_length >= 10000:  # Max episode length
                    break

            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)

            if terminated and ep_length < 10000:
                crash_count += 1

            if verbose:
                print(f"  Episode {ep + 1}/{n_episodes}: Reward = {ep_reward:.2f}, Length = {ep_length}")

        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'crash_rate': crash_count / n_episodes,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths
        }

        if verbose:
            print(f"\nEvaluation Results:")
            print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
            print(f"  Mean Length: {results['mean_length']:.1f}")
            print(f"  Crash Rate: {results['crash_rate']:.1%}")

        return results


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

    from env.drl_optimized_env_fixed import DRLOptimizedQueueEnvFixed

    print("Testing HCA2C Agent...")

    # Create environment
    env = DRLOptimizedQueueEnvFixed()

    # Create agent
    agent = HCA2C(
        env=env,
        learning_rate=3e-4,
        n_steps=32,
        verbose=1
    )

    print(f"\n✅ Agent created successfully!")
    print(f"   Device: {agent.device}")
    print(f"   Global dim: {agent.global_dim}")
    print(f"   Layer dim: {agent.layer_dim}")
    print(f"   N layers: {agent.n_layers}")

    # Test prediction
    obs, _ = agent.env.reset()
    action, info = agent.predict(obs)
    print(f"\n✅ Prediction test passed!")
    print(f"   Action shape: {action.shape}")
    print(f"   Value: {info['value']:.4f}")

    # Test short training
    print("\n\nTesting short training (1000 steps)...")
    agent.learn(total_timesteps=1000, log_interval=10, progress_bar=False)

    # Test evaluation
    print("\n\nTesting evaluation...")
    results = agent.evaluate(n_episodes=3, verbose=True)

    # Test save/load
    print("\n\nTesting save/load...")
    agent.save("/tmp/hca2c_test.pt")
    agent.load("/tmp/hca2c_test.pt")

    print("\n✅ All HCA2C agent tests passed!")
