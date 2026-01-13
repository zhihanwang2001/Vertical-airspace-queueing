"""
Major Revision Experiment 1.1: Extended Training for K=30
验证容量悖论是系统固有特性还是训练预算不足

关键问题：
- 论文声称 K=30 在 100K 步训练后崩溃（100%崩溃率）
- 评审质疑：可能只是训练不足，而非系统固有问题

实验设计：
1. K=30 (uniform [6,6,6,6,6]) 训练 1M 步（vs 原来的100K）
2. K=23 (inverted pyramid) 训练 1M 步作为对照
3. K=10 (low capacity) 训练 1M 步作为基准

算法：A2C, PPO（原论文主要算法）
每个配置：3 seeds
评估：每10K步评估一次，使用 T=200 统一协议

预期结果：
- Best case: K=30 仍然崩溃 → 容量悖论是真实的
- Worst case: K=30 成功收敛 → 容量悖论是训练不足
- Most likely: K=30 部分改善但仍差于K=10 → nuanced conclusion
"""

import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import gymnasium as gym
import numpy as np
import json
import time
from datetime import datetime
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from env.config import VerticalQueueConfig
from env.configurable_env_wrapper import ConfigurableEnvWrapper
from env.drl_wrapper_fixed import DictToBoxActionWrapperFixed, ObservationWrapperFixed


def create_config(capacity_type='k30_uniform', high_load_multiplier=10.0):
    """
    创建配置

    capacity_type:
    - 'k30_uniform': [6,6,6,6,6] 总30
    - 'k23_inverted': [8,6,4,3,2] 总23 (baseline)
    - 'k10_low': [2,2,2,2,2] 总10 (best performer in original)
    """
    config = VerticalQueueConfig()

    if capacity_type == 'k30_uniform':
        config.layer_capacities = [6, 6, 6, 6, 6]  # 总30
        name = "K=30 Uniform"
    elif capacity_type == 'k23_inverted':
        config.layer_capacities = [8, 6, 4, 3, 2]  # 总23
        name = "K=23 Inverted Pyramid"
    elif capacity_type == 'k10_low':
        config.layer_capacities = [2, 2, 2, 2, 2]  # 总10
        name = "K=10 Low Capacity"
    else:
        raise ValueError(f"Unknown capacity type: {capacity_type}")

    # 固定UAM流量模式（原论文设定）
    config.arrival_weights = [0.3, 0.25, 0.2, 0.15, 0.1]

    # 10× 高负载
    total_capacity = sum(config.layer_capacities)
    avg_service_rate = np.mean(config.layer_service_rates)
    base_rate_v3 = 0.75 * total_capacity * avg_service_rate / 5
    config.base_arrival_rate = base_rate_v3 * high_load_multiplier

    # 计算理论负载
    layer_loads = []
    for i, (w, c) in enumerate(zip(config.arrival_weights, config.layer_capacities)):
        layer_arrival = config.base_arrival_rate * w
        actual_service_rate = config.layer_service_rates[i]
        layer_load = layer_arrival / (c * actual_service_rate)
        layer_loads.append(layer_load)

    print(f"\n{'='*80}")
    print(f"配置: {name}")
    print(f"容量: {config.layer_capacities} (总计: {total_capacity})")
    print(f"到达权重: {config.arrival_weights}")
    print(f"总到达率: {config.base_arrival_rate:.2f}")
    print(f"\n各层理论负载:")
    for i, load in enumerate(layer_loads):
        status = "🔴" if load >= 1.0 else "🟡" if load > 0.8 else "🟢"
        print(f"  L{i}: {load*100:.1f}% {status}")
    print(f"平均负载: {np.mean(layer_loads)*100:.1f}%")
    print(f"最大负载: {np.max(layer_loads)*100:.1f}%")
    print(f"{'='*80}\n")

    return config, name


def create_env(config):
    """创建环境"""
    base_env = ConfigurableEnvWrapper(config)
    wrapped_env = DictToBoxActionWrapperFixed(base_env)
    env = ObservationWrapperFixed(wrapped_env)
    return env


def train_and_evaluate(
    algo_name='A2C',
    capacity_type='k30_uniform',
    seed=42,
    total_timesteps=1_000_000,  # 1M steps (vs original 100K)
    eval_freq=10_000,  # 每10K评估
    n_eval_episodes=10
):
    """
    训练并评估

    关键参数：
    - total_timesteps: 1M (10× original)
    - eval_freq: 10K (vs original 5K)
    - max_episode_steps: 200 (统一协议，与原论文A2C/PPO一致)
    """

    # 创建配置
    config, config_name = create_config(capacity_type)

    # 设置输出目录
    output_dir = Path(f"Results/major_revision_exp1/{capacity_type}/{algo_name}_seed{seed}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Training: {algo_name} on {config_name}")
    print(f"Seed: {seed}")
    print(f"Total steps: {total_timesteps:,}")
    print(f"Output: {output_dir}")
    print(f"{'='*80}\n")

    # 创建训练和评估环境
    train_env = create_env(config)
    eval_env = create_env(config)

    # 设置episode长度（统一协议）
    train_env.env.env._max_episode_steps = 1000  # 训练时较长
    eval_env.env.env._max_episode_steps = 200    # 评估时统一T=200

    # 创建算法
    if algo_name == 'A2C':
        # 使用原论文的staged learning rate
        # 但由于是1M步，调整transition point
        model = A2C(
            "MlpPolicy",
            train_env,
            learning_rate=7e-4,  # 初始高学习率
            n_steps=32,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=[512, 512, 256]),
            verbose=1,
            seed=seed,
            device='auto'
        )
    elif algo_name == 'PPO':
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            policy_kwargs=dict(net_arch=[256, 256]),
            verbose=1,
            seed=seed,
            device='auto'
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

    # 创建评估回调
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best_model"),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1
    )

    # 创建checkpoint回调（每50K保存）
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=str(output_dir / "checkpoints"),
        name_prefix=f"{algo_name}_checkpoint"
    )

    # 训练
    print(f"\n{'='*80}")
    print(f"开始训练...")
    print(f"{'='*80}\n")

    start_time = time.time()

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback],
            progress_bar=True
        )

        training_time = time.time() - start_time

        print(f"\n{'='*80}")
        print(f"训练完成！")
        print(f"耗时: {training_time/60:.1f} 分钟")
        print(f"{'='*80}\n")

        # 保存最终模型
        model.save(output_dir / "final_model")

        # 最终评估（T=200）
        print(f"\n{'='*80}")
        print(f"最终评估 (T=200, {n_eval_episodes} episodes)...")
        print(f"{'='*80}\n")

        eval_env.env.env._max_episode_steps = 200

        episode_rewards = []
        episode_lengths = []
        crash_count = 0

        for ep in range(n_eval_episodes):
            obs, _ = eval_env.reset()
            episode_reward = 0
            episode_length = 0
            done = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_length += 1

                if terminated and episode_length < 200:
                    crash_count += 1
                    break

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            print(f"  Episode {ep+1}: Reward={episode_reward:.1f}, Length={episode_length}")

        # 计算统计
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_length = np.mean(episode_lengths)
        crash_rate = crash_count / n_eval_episodes * 100
        completion_rate = 100 - crash_rate

        results = {
            'algorithm': algo_name,
            'capacity_type': capacity_type,
            'config_name': config_name,
            'seed': seed,
            'total_timesteps': total_timesteps,
            'training_time_minutes': training_time / 60,

            'final_eval': {
                'episode_steps': 200,
                'n_episodes': n_eval_episodes,
                'mean_reward': float(mean_reward),
                'std_reward': float(std_reward),
                'mean_episode_length': float(mean_length),
                'crash_rate': float(crash_rate),
                'completion_rate': float(completion_rate),
                'all_rewards': [float(r) for r in episode_rewards],
                'all_lengths': [int(l) for l in episode_lengths]
            },

            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'purpose': 'Major Revision Exp 1.1: Extended Training',
                'hypothesis_test': 'Capacity paradox: inherent vs training budget'
            }
        }

        # 保存结果
        results_file = output_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n{'='*80}")
        print(f"最终结果:")
        print(f"  平均奖励: {mean_reward:.1f} ± {std_reward:.1f}")
        print(f"  平均长度: {mean_length:.1f}")
        print(f"  崩溃率: {crash_rate:.1f}%")
        print(f"  完成率: {completion_rate:.1f}%")
        print(f"{'='*80}\n")

        print(f"结果已保存到: {results_file}")

        return results

    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        train_env.close()
        eval_env.close()


def main():
    """
    主函数：运行所有配置

    优先级：
    1. K=30 (关键) - 验证容量悖论
    2. K=23 (对照) - 确认扩展训练不破坏已知结果
    3. K=10 (基准) - 验证最优配置是否进一步改善
    """

    configurations = [
        # 最关键：K=30
        ('A2C', 'k30_uniform', 42),
        ('A2C', 'k30_uniform', 123),
        ('A2C', 'k30_uniform', 456),

        ('PPO', 'k30_uniform', 42),
        ('PPO', 'k30_uniform', 123),
        ('PPO', 'k30_uniform', 456),

        # 对照：K=23
        ('A2C', 'k23_inverted', 42),
        ('A2C', 'k23_inverted', 123),

        ('PPO', 'k23_inverted', 42),
        ('PPO', 'k23_inverted', 123),

        # 基准：K=10
        ('A2C', 'k10_low', 42),
        ('PPO', 'k10_low', 42),
    ]

    print(f"\n{'#'*80}")
    print(f"# Major Revision Experiment 1.1: Extended Training")
    print(f"# Total configurations: {len(configurations)}")
    print(f"# Estimated time: ~{len(configurations) * 2} hours (parallel可以减少)")
    print(f"{'#'*80}\n")

    all_results = []

    for i, (algo, capacity, seed) in enumerate(configurations):
        print(f"\n{'#'*80}")
        print(f"# Configuration {i+1}/{len(configurations)}")
        print(f"{'#'*80}\n")

        result = train_and_evaluate(
            algo_name=algo,
            capacity_type=capacity,
            seed=seed,
            total_timesteps=1_000_000,
            eval_freq=10_000,
            n_eval_episodes=10
        )

        if result:
            all_results.append(result)

    # 保存汇总结果
    summary_file = Path("Results/major_revision_exp1/summary.json")
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'#'*80}")
    print(f"# 所有实验完成！")
    print(f"# 汇总结果: {summary_file}")
    print(f"{'#'*80}\n")

    # 快速分析
    print("\n快速分析:")
    print("="*80)

    for capacity_type in ['k30_uniform', 'k23_inverted', 'k10_low']:
        relevant = [r for r in all_results if r['capacity_type'] == capacity_type]
        if not relevant:
            continue

        print(f"\n{relevant[0]['config_name']}:")

        for algo in ['A2C', 'PPO']:
            algo_results = [r for r in relevant if r['algorithm'] == algo]
            if not algo_results:
                continue

            rewards = [r['final_eval']['mean_reward'] for r in algo_results]
            crash_rates = [r['final_eval']['crash_rate'] for r in algo_results]

            print(f"  {algo}:")
            print(f"    Reward: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
            print(f"    Crash:  {np.mean(crash_rates):.1f}%")

    print("\n" + "="*80)
    print("分析完成！请查看详细结果进行论文修订。")


if __name__ == "__main__":
    main()
