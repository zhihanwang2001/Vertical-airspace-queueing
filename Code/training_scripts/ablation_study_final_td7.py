"""
最终消融实验：CCF-B期刊版本 - TD7算法
Final Ablation Study: CCF-B Journal Version - TD7 Algorithm

关键参数：
1. 算法：TD7（自研advanced算法）
2. 评估轮次：50 episodes（更可靠的统计）
3. 高负载：10x arrival rate（系统接近饱和）
4. 固定流量模式：[0.3, 0.25, 0.2, 0.15, 0.1]（真实UAM）
5. 5种容量结构：倒金字塔、均匀、高容量、正金字塔、低容量

目标：
- 验证倒金字塔[8,6,4,3,2]在高负载下的优势
- 对比TD7与A2C/PPO的性能
- 提供solid的统计evidence支持CCF-B论文
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

# 导入TD7组件
from algorithms.advanced.td7.td7_agent import TD7_Agent
from algorithms.baselines.space_utils import SB3DictWrapper

# 导入环境
from env.config import VerticalQueueConfig
from env.configurable_env_wrapper import ConfigurableEnvWrapper


def create_config(config_type='inverted_pyramid', high_load_multiplier=10.0):
    """
    创建高负载配置

    high_load_multiplier: 负载倍数（相对v3）
    """
    config = VerticalQueueConfig()

    # 设置容量
    if config_type == 'inverted_pyramid':
        config.layer_capacities = [8, 6, 4, 3, 2]  # 总23
    elif config_type == 'uniform':
        config.layer_capacities = [5, 5, 5, 5, 5]  # 总25
    elif config_type == 'high_capacity':
        config.layer_capacities = [8, 8, 8, 8, 8]  # 总40
    elif config_type == 'reverse_pyramid':
        config.layer_capacities = [2, 3, 4, 6, 8]  # 总23
    elif config_type == 'low_capacity':
        config.layer_capacities = [2, 2, 2, 2, 2]  # 总10
    else:
        raise ValueError(f"Unknown config type: {config_type}")

    # 固定真实UAM流量模式
    config.arrival_weights = [0.3, 0.25, 0.2, 0.15, 0.1]

    # 关键：大幅提高到达率
    total_capacity = sum(config.layer_capacities)
    avg_service_rate = np.mean(config.layer_service_rates)

    # 基础到达率 × 高负载倍数
    base_rate_v3 = 0.75 * total_capacity * avg_service_rate / 5
    config.base_arrival_rate = base_rate_v3 * high_load_multiplier

    # 计算每层的理论负载
    layer_loads = []
    for i, (w, c) in enumerate(zip(config.arrival_weights, config.layer_capacities)):
        layer_arrival = config.base_arrival_rate * w
        actual_service_rate = config.layer_service_rates[i]
        layer_load = layer_arrival / (c * actual_service_rate)
        layer_loads.append(layer_load)

    print(f"\n{'='*80}")
    print(f"配置: {config_type}")
    print(f"容量: {config.layer_capacities} (总计: {total_capacity})")
    print(f"到达权重: {config.arrival_weights} (固定真实UAM模式)")
    print(f"总到达率: {config.base_arrival_rate:.2f} (v3的{high_load_multiplier:.1f}倍)")
    print(f"\n各层理论负载 (ρ = λ/(μ·c)):")
    for i, load in enumerate(layer_loads):
        mu = config.layer_service_rates[i]
        status = "🔴过载!" if load >= 1.0 else "🟡临界" if load > 0.8 else "🟢正常"
        print(f"  Layer {i} (容量{config.layer_capacities[i]}, μ={mu:.1f}): {load*100:.1f}% {status}")
    print(f"平均负载: {np.mean(layer_loads)*100:.1f}%")
    print(f"最高负载: {np.max(layer_loads)*100:.1f}% (Layer {np.argmax(layer_loads)})")

    # 预测崩溃
    if any(load >= 1.0 for load in layer_loads):
        print(f"⚠️  预警：预计系统崩溃（存在ρ>=1.0的层）")
    else:
        print(f"✅ 预计系统稳定（所有层ρ<1.0）")
    print(f"{'='*80}\n")

    return config


def create_wrapped_env(config):
    """创建包装后的环境"""
    base_env = ConfigurableEnvWrapper(config=config)
    # TD7需要SB3DictWrapper来处理observation
    wrapped_env = SB3DictWrapper(base_env)
    return wrapped_env


def train_and_evaluate(config_type='inverted_pyramid',
                       timesteps=100000, eval_episodes=50, high_load_multiplier=10.0):
    """训练和评估TD7"""

    print(f"\n{'='*80}")
    print(f"实验: TD7 + {config_type}")
    print(f"高负载倍数: {high_load_multiplier}x")
    print(f"评估轮次: {eval_episodes}")
    print(f"{'='*80}\n")

    config = create_config(config_type, high_load_multiplier)
    env = create_wrapped_env(config)

    save_dir = Path(project_root) / 'Results' / 'ablation_study_final' / config_type
    save_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # 创建TD7 agent（使用TD7推荐的配置）
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

    print(f"\n开始训练TD7...")

    # 训练循环
    state, info = env.reset()
    episode_reward = 0
    episode_length = 0
    episode_count = 0

    for step in range(1, timesteps + 1):
        # 选择动作
        if step < td7_config['learning_starts']:
            action = env.action_space.sample()
        else:
            action = agent.act(state, training=True)

        # 执行动作
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        episode_reward += reward
        episode_length += 1

        # 存储经验
        agent.store_transition(state, action, reward, next_state, done)

        state = next_state

        # 训练
        if step >= td7_config['learning_starts']:
            agent.train()

        # Episode结束
        if done:
            episode_count += 1
            if episode_count % 10 == 0:
                print(f"  Episode {episode_count}: Reward = {episode_reward:.2f}, Length = {episode_length}")

            state, info = env.reset()
            episode_reward = 0
            episode_length = 0

        # 进度显示
        if step % 10000 == 0:
            print(f"  Training progress: {step}/{timesteps} ({step/timesteps*100:.1f}%)")

    training_time = time.time() - start_time

    # 保存模型
    model_path = save_dir / 'TD7_model.pt'
    agent.save(str(model_path))

    # 评估
    print(f"\n评估 ({eval_episodes} 回合)...")
    eval_rewards = []
    eval_lengths = []
    eval_terminated_count = 0  # 真实崩溃
    eval_truncated_count = 0   # 正常截断
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
            action = agent.act(obs, training=False)  # 确定性策略
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
            crash_marker = " 🔴[CRASHED - 系统崩溃!]"
        elif episode_truncated:
            eval_truncated_count += 1
            crash_marker = " ✅[完成]"
        else:
            crash_marker = ""

        if ep_waiting:
            eval_waiting_times.append(np.mean(ep_waiting))
        if ep_utils:
            eval_utilizations.append(np.mean(ep_utils))

        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}: {ep_reward:.2f} (长度{ep_len}){crash_marker}")

    mean_reward = np.mean(eval_rewards)
    std_reward = np.std(eval_rewards)
    terminated_rate = eval_terminated_count / eval_episodes
    truncated_rate = eval_truncated_count / eval_episodes
    mean_waiting = np.mean(eval_waiting_times) if eval_waiting_times else 0
    mean_util = np.mean(eval_utilizations) if eval_utilizations else 0
    mean_length = np.mean(eval_lengths)

    print(f"\n{'='*80}")
    print(f"评估结果:")
    print(f"  平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"  最佳奖励: {np.max(eval_rewards):.2f}")
    print(f"  🔴 崩溃率: {terminated_rate*100:.1f}% ({eval_terminated_count}/{eval_episodes})")
    print(f"  ✅ 完成率: {truncated_rate*100:.1f}% ({eval_truncated_count}/{eval_episodes})")
    print(f"  平均回合长度: {mean_length:.1f}")
    print(f"  平均等待: {mean_waiting:.2f} 时间步")
    print(f"  平均利用率: {mean_util*100:.1f}%")
    print(f"  训练时间: {training_time/60:.2f}分钟")
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
    parser.add_argument('--config', required=True,
                       choices=['inverted_pyramid', 'uniform', 'high_capacity',
                               'reverse_pyramid', 'low_capacity'])
    parser.add_argument('--timesteps', type=int, default=100000)
    parser.add_argument('--eval-episodes', type=int, default=50)
    parser.add_argument('--high-load-multiplier', type=float, default=10.0,
                       help='高负载倍数（相对v3）')
    args = parser.parse_args()

    try:
        result = train_and_evaluate(args.config,
                                   args.timesteps, args.eval_episodes, args.high_load_multiplier)
        print(f"\n✅ 完成: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
        print(f"🔴 崩溃率: {result['crash_rate']*100:.1f}%")
    except Exception as e:
        print(f"\n❌ 失败: {e}")
        import traceback
        traceback.print_exc()
