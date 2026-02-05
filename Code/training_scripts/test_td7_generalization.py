"""
TD7 Cross-Region Generalization Test Script

Core Objective: Verify TD7 model's generalization ability across different heterogeneous regions
Important: This is not a mock test - it uses real environment execution and model inference!

Test Logic:
1. Load trained TD7 model (./models/td7/td7_model_500000.pt)
2. Test in 5 different heterogeneous regions
3. Run 10 episodes per region to obtain real performance data
4. Record detailed performance metrics and environment configurations
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'rpTransition'))

import numpy as np
import json
from pathlib import Path
from typing import Dict, List
import time

# 导入基线算法
from algorithms.advanced.td7.td7_baseline import TD7Baseline

# 导入环境和配置
from env.configurable_env_wrapper import ConfigurableEnvWrapper
from algorithms.baselines.space_utils import SB3DictWrapper

# 导入异质性配置生成器
import importlib.util
spec = importlib.util.spec_from_file_location(
    "heterogeneous_configs",
    os.path.join(os.path.dirname(__file__), '..', 'rpTransition', 'heterogeneous_configs.py')
)
heterogeneous_configs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(heterogeneous_configs)

HeterogeneousRegionConfigs = heterogeneous_configs.HeterogeneousRegionConfigs


def test_td7_in_region(td7_baseline, config, region_name: str, n_episodes: int = 10, verbose: bool = True):
    """
    Test TD7 model in specified region

    Args:
        td7_baseline: TD7Baseline instance with loaded model
        config: VerticalQueueConfig configuration
        region_name: Region name
        n_episodes: Number of test episodes
        verbose: Whether to print detailed information

    Returns:
        dict: Test results
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"Testing region: {region_name}")
        print(f"{'='*80}")

    # Create environment for this region
    base_env = ConfigurableEnvWrapper(config)
    eval_env = SB3DictWrapper(base_env)

    # Record results
    episode_rewards = []
    episode_lengths = []
    episode_details = []

    # Run n_episodes episodes
    for episode in range(n_episodes):
        obs, info = eval_env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        # Run a complete episode
        while not done:
            # Use TD7 model to predict action (deterministic policy)
            action = td7_baseline.agent.act(obs, training=False)

            # Execute action
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

            # Prevent infinite loop
            if episode_length >= 1000:
                if verbose:
                    print(f"  Warning: Episode {episode+1} reached maximum step limit (1000)")
                break

        # Record results
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_details.append({
            'episode': episode + 1,
            'reward': float(episode_reward),
            'length': int(episode_length)
        })

        if verbose:
            print(f"  Episode {episode+1}/{n_episodes}: Reward = {episode_reward:.2f}, Length = {episode_length}")

    # Calculate statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)

    results = {
        'region_name': region_name,
        'n_episodes': n_episodes,
        'mean_reward': float(mean_reward),
        'std_reward': float(std_reward),
        'mean_length': float(mean_length),
        'episode_rewards': [float(r) for r in episode_rewards],
        'episode_lengths': [int(l) for l in episode_lengths],
        'episode_details': episode_details,
        'config_summary': base_env.get_config_summary()
    }

    if verbose:
        print(f"\nTest results for {region_name}:")
        print(f"   Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"   Mean length: {mean_length:.1f}")
        print(f"   Config summary: arrival_rate={results['config_summary']['base_arrival_rate']:.3f}, "
              f"capacity={results['config_summary']['total_capacity']}")

    # Clean up environment
    eval_env.close()

    return results


def main():
    """Main function: Test TD7 generalization performance across all heterogeneous regions"""

    print("\n" + "="*80)
    print("TD7 Cross-Region Generalization Test")
    print("="*80 + "\n")

    # ========== Step 1: Load trained TD7 model ==========
    print("Step 1: Load trained TD7 model")
    print("-"*80)

    model_path = "../../Models/td7/td7_model_500000.pt"

    if not os.path.exists(model_path):
        print(f"❌ 错误：找不到TD7模型文件 {model_path}")
        print("   请先训练TD7模型！")
        return

    print(f"📂 加载模型: {model_path}")
    print(f"   文件大小: {os.path.getsize(model_path) / (1024*1024):.1f} MB")

    # 创建TD7 baseline并加载模型
    td7 = TD7Baseline()
    td7.load(model_path)

    print("✅ TD7模型加载成功！")

    # ========== 第2步：创建异质性区域配置 ==========
    print("\n第2步：创建异质性区域配置")
    print("-"*80)

    config_generator = HeterogeneousRegionConfigs()
    all_configs = config_generator.get_all_configs()

    print(f"✅ 已创建 {len(all_configs)} 个区域配置:")
    for region_name in all_configs.keys():
        print(f"   - {region_name}")

    # ========== 第3步：在每个区域运行测试 ==========
    print("\n第3步：在每个区域运行泛化测试")
    print("-"*80)
    print("⚠️  这是真实测试，不是mock数据！每个区域将运行10个完整episode")

    all_results = {}
    n_episodes_per_region = 10

    start_time = time.time()

    for region_name, config in all_configs.items():
        results = test_td7_in_region(
            td7_baseline=td7,
            config=config,
            region_name=region_name,
            n_episodes=n_episodes_per_region,
            verbose=True
        )
        all_results[region_name] = results

    total_time = time.time() - start_time

    # ========== 第4步：汇总结果 ==========
    print("\n" + "="*80)
    print("测试完成！汇总结果")
    print("="*80 + "\n")

    print(f"{'区域':<30} {'平均奖励':<20} {'标准差':<15} {'平均长度':<15}")
    print("-"*80)

    baseline_reward = None
    for region_name, results in all_results.items():
        mean_reward = results['mean_reward']
        std_reward = results['std_reward']
        mean_length = results['mean_length']

        # 记录baseline（Region A）的性能
        if 'Standard' in region_name:
            baseline_reward = mean_reward

        # 计算与baseline的差异百分比
        if baseline_reward is not None and 'Standard' not in region_name:
            diff_pct = ((mean_reward - baseline_reward) / baseline_reward) * 100
            diff_str = f"({diff_pct:+.1f}%)"
        else:
            diff_str = "(baseline)"

        print(f"{region_name:<30} {mean_reward:<20.2f} {std_reward:<15.2f} {mean_length:<15.1f} {diff_str}")

    print("\n" + "-"*80)
    print(f"总测试时间: {total_time:.1f}秒")
    print(f"总episode数: {len(all_configs) * n_episodes_per_region}")

    # ========== 第5步：保存结果 ==========
    print("\n第5步：保存测试结果")
    print("-"*80)

    # 创建保存目录
    save_dir = Path("../../Results/generalization")
    save_dir.mkdir(exist_ok=True)

    # 保存详细结果
    results_file = save_dir / "td7_generalization_results.json"

    full_results = {
        'model_path': model_path,
        'test_date': time.strftime("%Y-%m-%d %H:%M:%S"),
        'n_episodes_per_region': n_episodes_per_region,
        'total_time_seconds': total_time,
        'regions': all_results
    }

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)

    print(f"✅ 详细结果已保存到: {results_file}")

    # 保存汇总表格（CSV格式）
    summary_file = save_dir / "td7_generalization_summary.csv"
    import csv

    with open(summary_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Region', 'Mean Reward', 'Std Reward', 'Mean Length', 'Diff from Baseline (%)'])

        for region_name, results in all_results.items():
            if baseline_reward is not None and 'Standard' not in region_name:
                diff_pct = ((results['mean_reward'] - baseline_reward) / baseline_reward) * 100
            else:
                diff_pct = 0.0

            writer.writerow([
                region_name,
                f"{results['mean_reward']:.2f}",
                f"{results['std_reward']:.2f}",
                f"{results['mean_length']:.1f}",
                f"{diff_pct:+.1f}%"
            ])

    print(f"✅ 汇总表格已保存到: {summary_file}")

    print("\n" + "="*80)
    print("✅ TD7 泛化性测试全部完成！")
    print("="*80 + "\n")

    print("📌 关键发现:")
    print(f"   1. Region A (Baseline): {all_results['Region_A_Standard']['mean_reward']:.2f} ± {all_results['Region_A_Standard']['std_reward']:.2f}")

    if baseline_reward:
        for region_name, results in all_results.items():
            if 'Standard' not in region_name:
                diff_pct = ((results['mean_reward'] - baseline_reward) / baseline_reward) * 100
                print(f"   - {region_name}: {results['mean_reward']:.2f} ({diff_pct:+.1f}%)")

    print("\n💡 下一步：")
    print("   1. 查看详细结果: cat generalization_results/td7_generalization_results.json")
    print("   2. 对比A2C和PPO的泛化性能")
    print("   3. 绘制可视化图表")


if __name__ == "__main__":
    main()
