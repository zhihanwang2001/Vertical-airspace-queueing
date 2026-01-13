"""
Top 3模型跨区域泛化性测试脚本 V2 - 增强版
Top 3 Models Cross-Region Generalization Test Script V2 - Enhanced

🎯 核心改进：
1. 不仅看累积reward，还评估多维度系统性能指标
2. 提取队列饱和度、负载率、稳定性、吞吐量等关键指标
3. 更准确地反映模型在异质性环境下的真实表现

评估指标：
- 累积奖励 (Cumulative Reward)
- 平均队列利用率 (Average Queue Utilization)
- 平均负载率 (Average Load Rate)
- 系统吞吐量 (System Throughput)
- 稳定性得分 (Stability Score)
- 拥堵度 (Congestion Level)
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
from algorithms.baselines.sb3_a2c_baseline import SB3A2CBaseline
from algorithms.baselines.sb3_ppo_baseline import SB3PPOBaseline
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


def test_model_in_region(model, model_type: str, config, region_name: str,
                         n_episodes: int = 10, verbose: bool = True):
    """
    在指定区域测试模型 - 增强版（提取多维度指标）

    Args:
        model: 已加载模型的baseline实例
        model_type: 模型类型 ('A2C', 'PPO', 'TD7')
        config: VerticalQueueConfig配置
        region_name: 区域名称
        n_episodes: 测试episode数量
        verbose: 是否打印详细信息

    Returns:
        dict: 测试结果（包含多维度指标）
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"测试: {model_type} @ {region_name}")
        print(f"{'='*80}")

    # 创建该区域的环境
    base_env = ConfigurableEnvWrapper(config)
    eval_env = SB3DictWrapper(base_env)

    # 记录结果
    episode_rewards = []
    episode_lengths = []

    # 新增：系统性能指标
    episode_avg_utilizations = []  # 平均队列利用率
    episode_avg_load_rates = []     # 平均负载率
    episode_throughputs = []        # 吞吐量
    episode_stability_scores = []   # 稳定性得分
    episode_max_utilizations = []   # 最大队列利用率（拥堵度）

    episode_details = []

    # 运行n_episodes个episode
    for episode in range(n_episodes):
        obs, info = eval_env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        # 收集episode内的系统指标
        step_utilizations = []
        step_load_rates = []
        step_stability_scores = []

        # 运行一个完整的episode
        while not done:
            # 根据模型类型选择预测方法
            if model_type == 'TD7':
                action = model.agent.act(obs, training=False)
            else:  # A2C or PPO
                action, _ = model.model.predict(obs, deterministic=True)

            # 执行动作
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

            # 提取系统指标（从info中）
            if 'utilization_rates' in info:
                step_utilizations.append(np.mean(info['utilization_rates']))
            if 'load_rates' in info:
                step_load_rates.append(np.mean(info['load_rates']))
            if 'stability_score' in info:
                step_stability_scores.append(info['stability_score'])

            # 防止无限循环
            if episode_length >= 1000:
                if verbose:
                    print(f"  ⚠️  Episode {episode+1} 达到最大步数限制 (1000)")
                break

        # 计算episode级别的系统指标
        avg_utilization = np.mean(step_utilizations) if step_utilizations else 0.0
        avg_load_rate = np.mean(step_load_rates) if step_load_rates else 0.0
        avg_stability = np.mean(step_stability_scores) if step_stability_scores else 0.0
        max_utilization = np.max(step_utilizations) if step_utilizations else 0.0
        throughput = info.get('throughput', 0.0) if info else 0.0

        # 记录结果
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_avg_utilizations.append(avg_utilization)
        episode_avg_load_rates.append(avg_load_rate)
        episode_throughputs.append(throughput)
        episode_stability_scores.append(avg_stability)
        episode_max_utilizations.append(max_utilization)

        episode_details.append({
            'episode': episode + 1,
            'reward': float(episode_reward),
            'length': int(episode_length),
            'avg_utilization': float(avg_utilization),
            'avg_load_rate': float(avg_load_rate),
            'throughput': float(throughput),
            'stability_score': float(avg_stability),
            'max_utilization': float(max_utilization)
        })

        if verbose:
            print(f"  Episode {episode+1}/{n_episodes}:")
            print(f"    Reward={episode_reward:.2f}, Length={episode_length}")
            print(f"    Utilization={avg_utilization:.3f}, LoadRate={avg_load_rate:.3f}, Throughput={throughput:.2f}")

    # 计算统计结果
    results = {
        'model_type': model_type,
        'region_name': region_name,
        'n_episodes': n_episodes,

        # 原有指标
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'mean_length': float(np.mean(episode_lengths)),

        # 新增：系统性能指标
        'mean_utilization': float(np.mean(episode_avg_utilizations)),
        'std_utilization': float(np.std(episode_avg_utilizations)),
        'mean_load_rate': float(np.mean(episode_avg_load_rates)),
        'std_load_rate': float(np.std(episode_avg_load_rates)),
        'mean_throughput': float(np.mean(episode_throughputs)),
        'std_throughput': float(np.std(episode_throughputs)),
        'mean_stability': float(np.mean(episode_stability_scores)),
        'std_stability': float(np.std(episode_stability_scores)),
        'mean_max_congestion': float(np.mean(episode_max_utilizations)),
        'std_max_congestion': float(np.std(episode_max_utilizations)),

        # 详细数据
        'episode_rewards': [float(r) for r in episode_rewards],
        'episode_lengths': [int(l) for l in episode_lengths],
        'episode_details': episode_details,
        'config_summary': base_env.get_config_summary()
    }

    if verbose:
        print(f"\n📊 {model_type} @ {region_name} 测试结果:")
        print(f"   累积奖励: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"   队列利用率: {results['mean_utilization']:.3f} ± {results['std_utilization']:.3f}")
        print(f"   负载率: {results['mean_load_rate']:.3f} ± {results['std_load_rate']:.3f}")
        print(f"   吞吐量: {results['mean_throughput']:.2f} ± {results['std_throughput']:.2f}")
        print(f"   稳定性: {results['mean_stability']:.3f} ± {results['std_stability']:.3f}")
        print(f"   最大拥堵度: {results['mean_max_congestion']:.3f} ± {results['std_max_congestion']:.3f}")

    # 清理环境
    eval_env.close()

    return results


def main():
    """主函数：测试所有3个模型在所有异质性区域的泛化性能 - 增强版"""

    print("\n" + "="*80)
    print("Top 3 模型跨区域泛化性测试 V2 - 增强版")
    print("Cross-Region Generalization Test V2 - Enhanced with Multi-Dimensional Metrics")
    print("="*80 + "\n")

    # ========== 第1步：加载训练好的3个模型 ==========
    print("第1步：加载训练好的3个模型")
    print("-"*80)

    models = {}
    model_paths = {
        'A2C': '../../Models/a2c/a2c_model_500000',
        'PPO': '../../Models/ppo/ppo_model_500000',
        'TD7': '../../Models/td7/td7_model_500000.pt'
    }

    # 加载A2C
    print("\n1.1 加载A2C模型...")
    if not os.path.exists(model_paths['A2C'] + '.pth'):
        print(f"❌ 错误：找不到A2C模型文件 {model_paths['A2C']}.pth")
        return

    a2c = SB3A2CBaseline()
    a2c.load(model_paths['A2C'])
    models['A2C'] = a2c
    print("✅ A2C模型加载成功！")

    # 加载PPO
    print("\n1.2 加载PPO模型...")
    if not os.path.exists(model_paths['PPO'] + '.pth'):
        print(f"❌ 错误：找不到PPO模型文件 {model_paths['PPO']}.pth")
        return

    ppo = SB3PPOBaseline()
    ppo.load(model_paths['PPO'])
    models['PPO'] = ppo
    print("✅ PPO模型加载成功！")

    # 加载TD7
    print("\n1.3 加载TD7模型...")
    if not os.path.exists(model_paths['TD7']):
        print(f"❌ 错误：找不到TD7模型文件 {model_paths['TD7']}")
        return

    print(f"📂 模型文件大小: {os.path.getsize(model_paths['TD7']) / (1024*1024):.1f} MB")
    td7 = TD7Baseline()
    td7.load(model_paths['TD7'])
    models['TD7'] = td7
    print("✅ TD7模型加载成功！")

    print("\n✅ 所有3个模型加载完成！")

    # ========== 第2步：创建异质性区域配置 ==========
    print("\n第2步：创建异质性区域配置")
    print("-"*80)

    config_generator = HeterogeneousRegionConfigs()
    all_configs = config_generator.get_all_configs()

    print(f"✅ 已创建 {len(all_configs)} 个区域配置:")
    for region_name in all_configs.keys():
        print(f"   - {region_name}")

    # ========== 第3步：在每个区域运行测试 ==========
    print("\n第3步：在每个区域运行泛化测试（增强版 - 提取多维度指标）")
    print("-"*80)
    print("⚠️  这是真实测试，不是mock数据！")
    print(f"   总测试数: {len(models)} 模型 × {len(all_configs)} 区域 × 10 episodes = {len(models) * len(all_configs) * 10} episodes")

    all_results = {
        'A2C': {},
        'PPO': {},
        'TD7': {}
    }

    n_episodes_per_region = 10
    start_time = time.time()

    # 对每个模型和每个区域运行测试
    for model_name in ['A2C', 'PPO', 'TD7']:
        print(f"\n{'='*80}")
        print(f"开始测试 {model_name} 模型")
        print(f"{'='*80}")

        model = models[model_name]

        for region_name, config in all_configs.items():
            results = test_model_in_region(
                model=model,
                model_type=model_name,
                config=config,
                region_name=region_name,
                n_episodes=n_episodes_per_region,
                verbose=True
            )
            all_results[model_name][region_name] = results

    total_time = time.time() - start_time

    # ========== 第4步：汇总结果（多维度） ==========
    print("\n" + "="*80)
    print("测试完成！汇总结果（多维度指标）")
    print("="*80 + "\n")

    # 表1: 累积奖励对比
    print("【表1】累积奖励对比 (Cumulative Reward)")
    print("-"*90)
    print(f"{'区域':<30} {'A2C':<20} {'PPO':<20} {'TD7':<20}")
    print("-"*90)

    baseline_rewards = {}

    for region_name in all_configs.keys():
        a2c_reward = all_results['A2C'][region_name]['mean_reward']
        ppo_reward = all_results['PPO'][region_name]['mean_reward']
        td7_reward = all_results['TD7'][region_name]['mean_reward']

        if 'Standard' in region_name:
            baseline_rewards['A2C'] = a2c_reward
            baseline_rewards['PPO'] = ppo_reward
            baseline_rewards['TD7'] = td7_reward

        print(f"{region_name:<30} {a2c_reward:<20.2f} {ppo_reward:<20.2f} {td7_reward:<20.2f}")

    # 表2: 队列利用率对比
    print("\n【表2】平均队列利用率 (Average Queue Utilization)")
    print("-"*90)
    print(f"{'区域':<30} {'A2C':<20} {'PPO':<20} {'TD7':<20}")
    print("-"*90)

    for region_name in all_configs.keys():
        a2c_util = all_results['A2C'][region_name]['mean_utilization']
        ppo_util = all_results['PPO'][region_name]['mean_utilization']
        td7_util = all_results['TD7'][region_name]['mean_utilization']

        print(f"{region_name:<30} {a2c_util:<20.3f} {ppo_util:<20.3f} {td7_util:<20.3f}")

    # 表3: 负载率对比
    print("\n【表3】平均负载率 (Average Load Rate - 越接近1越好)")
    print("-"*90)
    print(f"{'区域':<30} {'A2C':<20} {'PPO':<20} {'TD7':<20}")
    print("-"*90)

    for region_name in all_configs.keys():
        a2c_load = all_results['A2C'][region_name]['mean_load_rate']
        ppo_load = all_results['PPO'][region_name]['mean_load_rate']
        td7_load = all_results['TD7'][region_name]['mean_load_rate']

        print(f"{region_name:<30} {a2c_load:<20.3f} {ppo_load:<20.3f} {td7_load:<20.3f}")

    # 表4: 系统吞吐量对比
    print("\n【表4】系统吞吐量 (System Throughput - orders/step)")
    print("-"*90)
    print(f"{'区域':<30} {'A2C':<20} {'PPO':<20} {'TD7':<20}")
    print("-"*90)

    for region_name in all_configs.keys():
        a2c_thru = all_results['A2C'][region_name]['mean_throughput']
        ppo_thru = all_results['PPO'][region_name]['mean_throughput']
        td7_thru = all_results['TD7'][region_name]['mean_throughput']

        print(f"{region_name:<30} {a2c_thru:<20.2f} {ppo_thru:<20.2f} {td7_thru:<20.2f}")

    # 表5: 稳定性得分对比
    print("\n【表5】稳定性得分 (Stability Score - 越高越好)")
    print("-"*90)
    print(f"{'区域':<30} {'A2C':<20} {'PPO':<20} {'TD7':<20}")
    print("-"*90)

    for region_name in all_configs.keys():
        a2c_stab = all_results['A2C'][region_name]['mean_stability']
        ppo_stab = all_results['PPO'][region_name]['mean_stability']
        td7_stab = all_results['TD7'][region_name]['mean_stability']

        print(f"{region_name:<30} {a2c_stab:<20.3f} {ppo_stab:<20.3f} {td7_stab:<20.3f}")

    print("\n" + "-"*80)
    print(f"总测试时间: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
    print(f"总episode数: {len(models) * len(all_configs) * n_episodes_per_region}")

    # ========== 第5步：保存结果 ==========
    print("\n第5步：保存测试结果")
    print("-"*80)

    # 创建保存目录
    save_dir = Path("../../Results/generalization")
    save_dir.mkdir(exist_ok=True)

    # 保存详细结果
    results_file = save_dir / "all_models_generalization_results_v2.json"

    full_results = {
        'test_info': {
            'version': 'v2_enhanced',
            'test_date': time.strftime("%Y-%m-%d %H:%M:%S"),
            'n_episodes_per_region': n_episodes_per_region,
            'total_time_seconds': total_time,
            'models_tested': ['A2C', 'PPO', 'TD7'],
            'regions_tested': list(all_configs.keys()),
            'metrics_evaluated': [
                'cumulative_reward', 'queue_utilization', 'load_rate',
                'throughput', 'stability_score', 'max_congestion'
            ]
        },
        'model_paths': model_paths,
        'baseline_performance': {
            'A2C': f"{baseline_rewards['A2C']:.2f}",
            'PPO': f"{baseline_rewards['PPO']:.2f}",
            'TD7': f"{baseline_rewards['TD7']:.2f}"
        },
        'results': all_results
    }

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)

    print(f"✅ 详细结果已保存到: {results_file}")

    # 保存汇总表格（CSV格式 - 增强版）
    summary_file = save_dir / "all_models_generalization_summary_v2.csv"
    import csv

    with open(summary_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Region', 'Model',
            'Mean Reward', 'Std Reward',
            'Mean Utilization', 'Std Utilization',
            'Mean Load Rate', 'Std Load Rate',
            'Mean Throughput', 'Std Throughput',
            'Mean Stability', 'Std Stability',
            'Mean Max Congestion', 'Std Max Congestion'
        ])

        for region_name in all_configs.keys():
            for model_name in ['A2C', 'PPO', 'TD7']:
                res = all_results[model_name][region_name]
                writer.writerow([
                    region_name, model_name,
                    f"{res['mean_reward']:.2f}", f"{res['std_reward']:.2f}",
                    f"{res['mean_utilization']:.4f}", f"{res['std_utilization']:.4f}",
                    f"{res['mean_load_rate']:.4f}", f"{res['std_load_rate']:.4f}",
                    f"{res['mean_throughput']:.2f}", f"{res['std_throughput']:.2f}",
                    f"{res['mean_stability']:.4f}", f"{res['std_stability']:.4f}",
                    f"{res['mean_max_congestion']:.4f}", f"{res['std_max_congestion']:.4f}"
                ])

    print(f"✅ 汇总表格已保存到: {summary_file}")

    print("\n" + "="*80)
    print("✅ 所有模型泛化性测试全部完成（增强版）！")
    print("="*80 + "\n")

    print("📌 关键发现（多维度评估）:")
    print(f"   Baseline性能 (Region A):")
    print(f"     - A2C: {baseline_rewards['A2C']:.2f}")
    print(f"     - PPO: {baseline_rewards['PPO']:.2f}")
    print(f"     - TD7: {baseline_rewards['TD7']:.2f}")

    print("\n💡 下一步：")
    print("   1. 查看详细结果: cat generalization_results/all_models_generalization_results_v2.json")
    print("   2. 查看汇总表格: cat generalization_results/all_models_generalization_summary_v2.csv")
    print("   3. 分析多维度指标，撰写论文")


if __name__ == "__main__":
    main()
