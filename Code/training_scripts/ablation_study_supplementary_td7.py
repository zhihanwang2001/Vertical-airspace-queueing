"""
Final Ablation Study: CCF-B Journal Version - TD7 Algorithm
Final Ablation Study: CCF-B Journal Version - TD7 Algorithm

Key Parameters:
1. Algorithm: TD7 (self-developed advanced algorithm)
2. Evaluation rounds: 50 episodes (more reliable statistics)
3. High load: 10x arrival rate (system near saturation)
4. Fixed traffic pattern: [0.3, 0.25, 0.2, 0.15, 0.1] (real UAM)
5. 5 capacity structures: inverted pyramid, uniform, high capacity, normal pyramid, low capacity

Goals:
- Verify advantage of inverted pyramid [8,6,4,3,2] under high load
- Compare TD7 performance with A2C/PPO
- Provide solid statistical evidence for CCF-B paper
"""

import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import json
import time
from datetime import datetime

# Import TD7 components
from algorithms.advanced.td7.td7_agent import TD7_Agent
from algorithms.baselines.space_utils import SB3DictWrapper

# Import environment
from env.config import VerticalQueueConfig
from env.configurable_env_wrapper import ConfigurableEnvWrapper


def create_config(config_type='capacity_4x5', high_load_multiplier=10.0):
    """
    Create high load configuration

    high_load_multiplier: Load multiplier (relative to v3)
    """
    config = VerticalQueueConfig()

    # Set capacity
    if config_type == 'capacity_4x5':
        config.layer_capacities = [4, 4, 4, 4, 4]  # Total 20
    elif config_type == 'capacity_6x5':
        config.layer_capacities = [6, 6, 6, 6, 6]  # Total 30
    else:
        raise ValueError(f"Unknown config type: {config_type}")

    # Fixed real UAM traffic pattern
    config.arrival_weights = [0.3, 0.25, 0.2, 0.15, 0.1]

    # Key: Significantly increase arrival rate
    total_capacity = sum(config.layer_capacities)
    avg_service_rate = np.mean(config.layer_service_rates)

    # Base arrival rate × high load multiplier
    base_rate_v3 = 0.75 * total_capacity * avg_service_rate / 5
    config.base_arrival_rate = base_rate_v3 * high_load_multiplier

    # Calculate theoretical load per layer
    layer_loads = []
    for i, (w, c) in enumerate(zip(config.arrival_weights, config.layer_capacities)):
        layer_arrival = config.base_arrival_rate * w
        actual_service_rate = config.layer_service_rates[i]
        layer_load = layer_arrival / (c * actual_service_rate)
        layer_loads.append(layer_load)

    print(f"\n{'='*80}")
    print(f"Configuration: {config_type}")
    print(f"Capacity: {config.layer_capacities} (Total: {total_capacity})")
    print(f"Arrival weights: {config.arrival_weights} (Fixed real UAM pattern)")
    print(f"Total arrival rate: {config.base_arrival_rate:.2f} ({high_load_multiplier:.1f}x of v3)")
    print(f"\nTheoretical load per layer (ρ = λ/(μ·c)):")
    for i, load in enumerate(layer_loads):
        mu = config.layer_service_rates[i]
        status = "OVERLOAD!" if load >= 1.0 else "CRITICAL" if load > 0.8 else "NORMAL"
        print(f"  Layer {i} (capacity {config.layer_capacities[i]}, μ={mu:.1f}): {load*100:.1f}% {status}")
    print(f"Average load: {np.mean(layer_loads)*100:.1f}%")
    print(f"Maximum load: {np.max(layer_loads)*100:.1f}% (Layer {np.argmax(layer_loads)})")

    # Predict crash
    if any(load >= 1.0 for load in layer_loads):
        print(f"Warning: System crash expected (layers with ρ>=1.0 exist)")
    else:
        print(f"System stability expected (all layers ρ<1.0)")
    print(f"{'='*80}\n")

    return config


def create_wrapped_env(config):
    """Create wrapped environment"""
    base_env = ConfigurableEnvWrapper(config=config)
    # TD7 needs SB3DictWrapper to handle observation
    wrapped_env = SB3DictWrapper(base_env)
    return wrapped_env


def train_and_evaluate(config_type='capacity_4x5',
                       timesteps=100000, eval_episodes=50, high_load_multiplier=10.0):
    """Train and evaluate TD7"""

    print(f"\n{'='*80}")
    print(f"Experiment: TD7 + {config_type}")
    print(f"High load multiplier: {high_load_multiplier}x")
    print(f"Evaluation episodes: {eval_episodes}")
    print(f"{'='*80}\n")

    config = create_config(config_type, high_load_multiplier)
    env = create_wrapped_env(config)

    save_dir = Path(project_root) / 'Results' / 'ablation_study_supplementary' / config_type
    save_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # Create TD7 agent (using TD7 recommended configuration)
    td7_config = {
        'embedding_dim': 256,
        'hidden_dim': 256,
        'max_action': 1.0,
        'actor_lr': 3e-4,
        'critic_lr': 3e-4,
        'encoder_lr': 3e-4,
        'gamma': 0.99,
        'tau': 0.005,
        'policy_delay': 2,
        'target_noise': 0.2,
        'noise_clip': 0.5,
        'exploration_noise': 0.1,
        'buffer_size': 1000000,
        'batch_size': 256,
        'alpha': 0.6,
        'beta': 0.4,
        'beta_increment': 0.001,
        'embedding_loss_weight': 1.0,
        'embedding_update_freq': 1,
        'use_checkpoints': True,
        'checkpoint_freq': 10000,
        'max_checkpoints': 5,
        'learning_starts': 25000,
        'train_freq': 1,
        'seed': 42,
        'device': 'cuda'
    }

    agent = TD7_Agent(
        state_space=env.observation_space,
        action_space=env.action_space,
        config=td7_config
    )

    print(f"\nStarting TD7 training...")

    # Training loop
    state, info = env.reset()
    episode_reward = 0
    episode_length = 0
    episode_count = 0

    for step in range(1, timesteps + 1):
        # Select action
        if step < td7_config['learning_starts']:
            action = env.action_space.sample()
        else:
            action = agent.act(state, training=True)

        # Execute action
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        episode_reward += reward
        episode_length += 1

        # Store experience
        agent.store_transition(state, action, reward, next_state, done)

        state = next_state

        # Train
        if step >= td7_config['learning_starts']:
            agent.train()

        # Episode end
        if done:
            episode_count += 1
            if episode_count % 10 == 0:
                print(f"  Episode {episode_count}: Reward = {episode_reward:.2f}, Length = {episode_length}")

            state, info = env.reset()
            episode_reward = 0
            episode_length = 0

        # Progress display
        if step % 10000 == 0:
            print(f"  Training progress: {step}/{timesteps} ({step/timesteps*100:.1f}%)")

    training_time = time.time() - start_time

    # Save model
    model_path = save_dir / 'TD7_model.pt'
    agent.save(str(model_path))

    # Evaluation
    print(f"\nEvaluation ({eval_episodes} episodes)...")
    eval_rewards = []
    eval_lengths = []
    eval_terminated_count = 0  # Real crashes
    eval_truncated_count = 0   # Normal truncation
    eval_waiting_times = []
    eval_utilizations = []

    for ep in range(eval_episodes):
        obs, info = env.reset()
        done = False
        ep_reward = 0
        ep_len = 0
        ep_waiting = []
        ep_utils = []
        episode_terminated = False
        episode_truncated = False

        while not done:
            action = agent.act(obs, training=False)  # Deterministic policy
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            ep_reward += reward
            ep_len += 1

            if done:
                episode_terminated = term
                episode_truncated = trunc

            if 'avg_waiting_time' in info:
                ep_waiting.append(info['avg_waiting_time'])
            if 'utilization_rates' in info:
                ep_utils.append(np.mean(info['utilization_rates']))

        eval_rewards.append(ep_reward)
        eval_lengths.append(ep_len)

        if episode_terminated:
            eval_terminated_count += 1
            crash_marker = " [CRASHED - System crash!]"
        elif episode_truncated:
            eval_truncated_count += 1
            crash_marker = " [COMPLETED]"
        else:
            crash_marker = ""

        if ep_waiting:
            eval_waiting_times.append(np.mean(ep_waiting))
        if ep_utils:
            eval_utilizations.append(np.mean(ep_utils))

        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}: {ep_reward:.2f} (length {ep_len}){crash_marker}")

    mean_reward = np.mean(eval_rewards)
    std_reward = np.std(eval_rewards)
    terminated_rate = eval_terminated_count / eval_episodes
    truncated_rate = eval_truncated_count / eval_episodes
    mean_waiting = np.mean(eval_waiting_times) if eval_waiting_times else 0
    mean_util = np.mean(eval_utilizations) if eval_utilizations else 0
    mean_length = np.mean(eval_lengths)

    print(f"\n{'='*80}")
    print(f"Evaluation Results:")
    print(f"  Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"  Best reward: {np.max(eval_rewards):.2f}")
    print(f"  Crash rate: {terminated_rate*100:.1f}% ({eval_terminated_count}/{eval_episodes})")
    print(f"  Completion rate: {truncated_rate*100:.1f}% ({eval_truncated_count}/{eval_episodes})")
    print(f"  Mean episode length: {mean_length:.1f}")
    print(f"  Mean waiting time: {mean_waiting:.2f} timesteps")
    print(f"  Mean utilization: {mean_util*100:.1f}%")
    print(f"  Training time: {training_time/60:.2f} minutes")
    print(f"{'='*80}")

    results = {
        'config_type': config_type,
        'algorithm': 'TD7',
        'layer_capacities': config.layer_capacities,
        'total_capacity': sum(config.layer_capacities),
        'arrival_weights': config.arrival_weights,
        'base_arrival_rate': float(config.base_arrival_rate),
        'high_load_multiplier': high_load_multiplier,
        'mean_reward': float(mean_reward),
        'std_reward': float(std_reward),
        'best_reward': float(np.max(eval_rewards)),
        'worst_reward': float(np.min(eval_rewards)),
        'crash_rate': float(terminated_rate),
        'completion_rate': float(truncated_rate),
        'terminated_count': eval_terminated_count,
        'truncated_count': eval_truncated_count,
        'mean_episode_length': float(mean_length),
        'mean_waiting_time': float(mean_waiting),
        'mean_utilization': float(mean_util),
        'training_time_minutes': float(training_time / 60),
        'eval_rewards': [float(r) for r in eval_rewards],
        'eval_lengths': [int(l) for l in eval_lengths],
        'timestamp': datetime.now().isoformat()
    }

    results_path = save_dir / 'TD7_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    env.close()
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, choices=["capacity_4x5", "capacity_6x5"])
    parser.add_argument('--timesteps', type=int, default=100000)
    parser.add_argument('--eval-episodes', type=int, default=50)
    parser.add_argument('--high-load-multiplier', type=float, default=10.0,
                       help='High load multiplier (relative to v3)')
    args = parser.parse_args()

    try:
        result = train_and_evaluate(args.config,
                                   args.timesteps, args.eval_episodes, args.high_load_multiplier)
        print(f"\nCompleted: {result['mean_reward']:.2f} +/- {result['std_reward']:.2f}")
        print(f"Crash rate: {result['crash_rate']*100:.1f}%")
    except Exception as e:
        print(f"\nFailed: {e}")
        import traceback
        traceback.print_exc()
