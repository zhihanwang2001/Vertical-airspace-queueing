"""
实验A: 5× 负载结构对比实验 (CRITICAL)
Experiment A: Structural Comparison at 5× Load (CRITICAL)

问题背景：
- 补充实验(n=3) @ 10×负载全部崩溃 (crash_rate=100%)
- 原因: 10×负载过于极端，倒金字塔底层 ρ=345%，DRL无法学习有效策略
- 影响: 无法验证"倒金字塔优于正金字塔"的核心claim

解决方案：
- 降低负载至5×倍 (ρ≈172% - 具有挑战性但可学习)
- 只运行结构对比实验 (不含容量悖论)
- 保持其他参数与原实验一致

实验设计：
- Config 1: Inverted Pyramid [8,6,4,3,2] @ 5× load
- Config 2: Normal Pyramid [2,3,4,6,8] @ 5× load
- Algorithms: A2C, PPO
- Seeds: 42 (existing), 123, 456 (new)

默认总计: 12 training runs（可通过 --seeds / --n-seeds 扩展）
- 2 configs × 2 algorithms × N seeds

预期结果：
- 倒金字塔应显著优于正金字塔
- Crash rate < 50% (可接受的训练稳定性)
- 提供n=3统计显著性检验
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

from env.config import VerticalQueueConfig
from env.configurable_env_wrapper import ConfigurableEnvWrapper
from env.drl_wrapper_fixed import DictToBoxActionWrapperFixed, ObservationWrapperFixed


def create_config(config_type='inverted_pyramid', high_load_multiplier=5.0):
    """
    创建高负载配置

    config_type: 配置类型
    - inverted_pyramid: [8,6,4,3,2] 倒金字塔
    - normal_pyramid: [2,3,4,6,8] 正金字塔
    - low_capacity: [2,2,2,2,2] K=10
    - capacity_30: [6,6,6,6,6] K=30
    """
    config = VerticalQueueConfig()

    # 设置容量
    if config_type == 'inverted_pyramid':
        config.layer_capacities = [8, 6, 4, 3, 2]  # 总23
    elif config_type == 'normal_pyramid':
        config.layer_capacities = [2, 3, 4, 6, 8]  # 总23
    elif config_type == 'low_capacity':
        config.layer_capacities = [2, 2, 2, 2, 2]  # 总10 (K=10)
    elif config_type == 'capacity_30':
        config.layer_capacities = [6, 6, 6, 6, 6]  # 总30 (K=30)
    else:
        raise ValueError(f"Unknown config type: {config_type}")

    # 固定真实UAM流量模式
    config.arrival_weights = [0.3, 0.25, 0.2, 0.15, 0.1]

    # 高负载设置 (5x - 降低自10x以改善训练稳定性)
    total_capacity = sum(config.layer_capacities)
    avg_service_rate = np.mean(config.layer_service_rates)
    base_rate_v3 = 0.75 * total_capacity * avg_service_rate / 5
    config.base_arrival_rate = base_rate_v3 * high_load_multiplier  # 默认5.0×

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
    print(f"到达权重: {config.arrival_weights}")
    print(f"总到达率: {config.base_arrival_rate:.2f} ({high_load_multiplier:.1f}x高负载)")
    print(f"平均负载: {np.mean(layer_loads)*100:.1f}%")
    print(f"{'='*80}\n")

    return config


def create_wrapped_env(config):
    """创建包装后的环境"""
    base_env = ConfigurableEnvWrapper(config=config)
    wrapped_env = DictToBoxActionWrapperFixed(base_env)
    wrapped_env = ObservationWrapperFixed(wrapped_env)
    return wrapped_env


def train_and_evaluate(algorithm_name='A2C', config_type='inverted_pyramid',
                       timesteps=100000, eval_episodes=50, seed=42,
                       high_load_multiplier=5.0):
    """
    训练和评估单次实验 @ 5× 负载

    参数:
    - algorithm_name: 'A2C' or 'PPO'
    - config_type: 配置类型 (inverted_pyramid 或 normal_pyramid)
    - timesteps: 训练步数 (默认100K)
    - eval_episodes: 评估回合数 (默认50)
    - seed: 随机种子
    - high_load_multiplier: 高负载倍数 (默认5x - 降低自10x)
    """

    print(f"\n{'='*80}")
    print(f"实验: {algorithm_name} + {config_type}")
    print(f"Seed: {seed}")
    print(f"{'='*80}\n")

    config = create_config(config_type, high_load_multiplier)
    env = create_wrapped_env(config)

    # 保存路径: Data/ablation_studies/structural_5x_load/{config_type}/{algorithm}_{seed}_results.json
    save_dir = Path(project_root).parent / 'Data' / 'ablation_studies' / 'structural_5x_load' / config_type
    save_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    # 创建模型
    if algorithm_name == 'A2C':
        model = A2C('MlpPolicy', env, learning_rate=0.0007, n_steps=32,
                   gamma=0.99, gae_lambda=0.95, ent_coef=0.01, vf_coef=0.5,
                   max_grad_norm=0.5, normalize_advantage=True,
                   verbose=1, seed=seed, device='cuda')
    elif algorithm_name == 'PPO':
        model = PPO('MlpPolicy', env, learning_rate=0.0003, n_steps=2048,
                   batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                   clip_range=0.2, ent_coef=0.0, vf_coef=0.5,
                   max_grad_norm=0.5, verbose=1, seed=seed, device='cuda')
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")

    print(f"\n开始训练 ({timesteps} timesteps)...")
    model.learn(total_timesteps=timesteps)
    training_time = time.time() - start_time

    # 保存模型
    model_path = save_dir / f'{algorithm_name}_seed{seed}_model.zip'
    model.save(str(model_path))

    # 评估
    print(f"\n评估 ({eval_episodes} 回合)...")
    eval_rewards = []
    eval_lengths = []
    eval_terminated_count = 0  # 真实崩溃
    eval_truncated_count = 0   # 正常截断
    eval_waiting_times = []
    eval_utilizations = []
    # 稳定性代理指标（每个episode的均值）
    ep_means_lyapunov = []
    ep_means_lyapunov_drift = []
    ep_means_drift_l1 = []
    ep_safe_ratios = []
    ep_means_max_load = []

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
            action, _ = model.predict(obs, deterministic=True)
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
            # 收集稳定性代理指标
            if isinstance(info, dict):
                if 'lyapunov' in info:
                    ep_means_lyapunov.append(info['lyapunov'])
                if 'lyapunov_drift' in info:
                    ep_means_lyapunov_drift.append(info['lyapunov_drift'])
                if 'drift_l1' in info:
                    ep_means_drift_l1.append(info['drift_l1'])
                if 'is_safe' in info:
                    # 以布尔值平均作为安全比例贡献
                    ep_safe_ratios.append(1.0 if info['is_safe'] else 0.0)
                if 'max_load_rate' in info:
                    ep_means_max_load.append(info['max_load_rate'])

        eval_rewards.append(ep_reward)
        eval_lengths.append(ep_len)

        if episode_terminated:
            eval_terminated_count += 1
            crash_marker = " 🔴[CRASHED]"
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

    # 统计结果
    mean_reward = np.mean(eval_rewards)
    std_reward = np.std(eval_rewards)
    terminated_rate = eval_terminated_count / eval_episodes
    truncated_rate = eval_truncated_count / eval_episodes
    mean_waiting = np.mean(eval_waiting_times) if eval_waiting_times else 0
    mean_util = np.mean(eval_utilizations) if eval_utilizations else 0
    mean_length = np.mean(eval_lengths)

    # 计算稳定性代理的均值（若存在）
    def _safe_mean(arr):
        return float(np.mean(arr)) if len(arr) > 0 else 0.0
    stability_metrics = {
        'mean_lyapunov': _safe_mean(ep_means_lyapunov),
        'mean_lyapunov_drift': _safe_mean(ep_means_lyapunov_drift),
        'mean_drift_l1': _safe_mean(ep_means_drift_l1),
        'mean_safe_ratio': _safe_mean(ep_safe_ratios),
        'mean_max_load_rate': _safe_mean(ep_means_max_load)
    }

    print(f"\n{'='*80}")
    print(f"评估结果:")
    print(f"  平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"  最佳奖励: {np.max(eval_rewards):.2f}")
    print(f"  🔴 崩溃率: {terminated_rate*100:.1f}% ({eval_terminated_count}/{eval_episodes})")
    print(f"  ✅ 完成率: {truncated_rate*100:.1f}% ({eval_truncated_count}/{eval_episodes})")
    print(f"  平均回合长度: {mean_length:.1f}")
    print(f"  训练时间: {training_time/60:.2f}分钟")
    print(f"{'='*80}")

    # 保存结果
    results = {
        'config_type': config_type,
        'algorithm': algorithm_name,
        'seed': seed,
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
    results.update(stability_metrics)

    results_path = save_dir / f'{algorithm_name}_seed{seed}_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ 结果已保存至: {results_path}\n")

    env.close()
    return results


def _parse_seeds(seeds_arg: str = None, n_seeds: int = None) -> list:
    """Parse seeds from CLI: comma-separated list or generate range starting at 42."""
    if seeds_arg:
        try:
            return [int(s.strip()) for s in seeds_arg.split(',') if s.strip()]
        except Exception:
            print(f"⚠️ 无法解析 --seeds={seeds_arg}，使用默认 [42,123,456]")
            return [42, 123, 456]
    if n_seeds and n_seeds > 0:
        return list(range(42, 42 + n_seeds))
    return [42, 123, 456]


def run_all_supplementary_experiments(seeds: list = None,
                                      timesteps: int = 100000,
                                      eval_episodes: int = 50,
                                      high_load_multiplier: float = 5.0):
    """
    运行实验A: 5× 负载结构对比 (12 runs)

    目标: 修复10×负载下的100%崩溃问题

    实验矩阵:
    - Inverted Pyramid [8,6,4,3,2] vs Normal Pyramid [2,3,4,6,8]
    - Algorithms: A2C, PPO
    - Seeds: 42 (existing baseline), 123, 456 (new runs for n=3)
    - Load: 5× (降低自10×以改善训练稳定性)
    """

    # 定义实验矩阵 - 只含结构对比
    if seeds is None:
        seeds = [42, 123, 456]

    experiments = [
        {'config': 'inverted_pyramid', 'algo': 'A2C', 'seeds': seeds},
        {'config': 'inverted_pyramid', 'algo': 'PPO', 'seeds': seeds},
        {'config': 'normal_pyramid', 'algo': 'A2C', 'seeds': seeds},
        {'config': 'normal_pyramid', 'algo': 'PPO', 'seeds': seeds},
    ]

    total_experiments = sum(len(exp['seeds']) for exp in experiments)
    print(f"\n{'='*80}")
    print(f"实验A: 5× 负载结构对比")
    print(f"总计: {total_experiments} 次训练 (2 configs × 2 algos × 3 seeds)")
    print(f"负载倍数: 5× (降低自10×以改善训练稳定性)")
    print(f"{'='*80}")

    # 运行实验
    all_results = []
    completed = 0

    for exp_config in experiments:
        config_type = exp_config['config']
        algorithm = exp_config['algo']
        seeds = exp_config['seeds']

        for seed in seeds:
            completed += 1
            print(f"\n\n{'#'*80}")
            print(f"进度: [{completed}/{total_experiments}] {config_type} + {algorithm} (seed={seed})")
            print(f"{'#'*80}")

            try:
                result = train_and_evaluate(
                    algorithm_name=algorithm,
                    config_type=config_type,
                    timesteps=timesteps,
                    eval_episodes=eval_episodes,
                    seed=seed,
                    high_load_multiplier=high_load_multiplier
                )
                all_results.append(result)
                print(f"\n✅ [{completed}/{total_experiments}] 完成: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")

            except Exception as e:
                print(f"\n❌ [{completed}/{total_experiments}] 失败: {config_type} + {algorithm} (seed={seed})")
                print(f"错误: {e}")
                import traceback
                traceback.print_exc()

    # 保存总结
    summary = {
        'total_experiments': total_experiments,
        'completed': len(all_results),
        'failed': total_experiments - len(all_results),
        'timestamp': datetime.now().isoformat(),
        'experiments': all_results
    }

    summary_path = Path(project_root).parent / 'Data' / 'ablation_studies' / 'structural_5x_load' / 'EXPERIMENT_SUMMARY.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n\n{'='*80}")
    print(f"实验A完成! (5× 负载结构对比)")
    print(f"成功: {len(all_results)}/{total_experiments}")
    print(f"总结已保存至: {summary_path}")
    print(f"{'='*80}\n")

    # 显示关键结果对比
    if len(all_results) > 0:
        print("\n关键结果预览:")
        print("="*80)
        for r in all_results:
            crash_indicator = "🔴" if r['crash_rate'] > 0.5 else "✅"
            print(f"{crash_indicator} {r['config_type']:<20} {r['algorithm']:<5} seed={r['seed']:<4} "
                  f"reward={r['mean_reward']:>8.1f} crash={r['crash_rate']*100:>5.1f}%")
        print("="*80)

    return all_results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='实验A: 5× 负载结构对比')
    parser.add_argument('--mode', choices=['single', 'all'], default='all',
                       help='运行模式: single (单次实验) 或 all (全部12次)')
    parser.add_argument('--algorithm', choices=['A2C', 'PPO'],
                       help='算法 (仅single模式)')
    parser.add_argument('--config',
                       choices=['inverted_pyramid', 'normal_pyramid', 'low_capacity', 'capacity_30'],
                       help='配置类型 (仅single模式)')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子 (仅single模式, 默认42)')
    parser.add_argument('--timesteps', type=int, default=100000,
                       help='训练步数')
    parser.add_argument('--eval-episodes', type=int, default=50,
                       help='评估回合数')
    parser.add_argument('--load-multiplier', type=float, default=5.0,
                       help='负载倍数 (默认5.0)')
    parser.add_argument('--seeds', type=str, default=None,
                       help='以逗号分隔的随机种子列表，如 42,123,456')
    parser.add_argument('--n-seeds', type=int, default=None,
                       help='自动生成的种子数量（从42开始递增）')

    args = parser.parse_args()

    if args.mode == 'all':
        seeds = _parse_seeds(args.seeds, args.n_seeds)
        print(f"\n🚀 开始运行实验A: 5× 负载结构对比 ({len(seeds)*4} 次训练)...\n")
        run_all_supplementary_experiments(
            seeds=seeds,
            timesteps=args.timesteps,
            eval_episodes=args.eval_episodes,
            high_load_multiplier=args.load_multiplier
        )

    elif args.mode == 'single':
        if not args.algorithm or not args.config:
            print("❌ 错误: single模式需要指定 --algorithm 和 --config")
            parser.print_help()
        else:
            print(f"\n🚀 运行单次实验: {args.algorithm} + {args.config} (seed={args.seed}) @ {args.load_multiplier}× load\n")
            result = train_and_evaluate(
                algorithm_name=args.algorithm,
                config_type=args.config,
                timesteps=args.timesteps,
                eval_episodes=args.eval_episodes,
                seed=args.seed,
                high_load_multiplier=args.load_multiplier
            )
            print(f"\n✅ 完成: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}, crash={result['crash_rate']*100:.1f}%")
