"""
Top 3模型跨区域泛化性测试脚本
Top 3 Models Cross-Region Generalization Test Script

🎯 核心目标：验证Top 3模型（A2C, PPO, TD7）在不同异质性区域的泛化能力
⚠️  重要：这不是mock测试，使用真实的环境运行和模型推理！

测试逻辑：
1. 加载已训练的3个模型
   - A2C: ./models/a2c/a2c_model_500000.pth (4392.86 ± 145.42)
   - PPO: ./models/ppo/ppo_model_500000.pth (4419.98 ± 135.71)
   - TD7: ./models/td7/td7_model_500000.pt  (4351.84 from RP1)
2. 在5个不同的heterogeneous region中测试
3. 每个region运行10个episode获取真实性能数据
4. 记录详细的性能指标和环境配置
5. 对比3个模型的泛化能力
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
    在指定区域测试模型

    Args:
        model: 已加载模型的baseline实例
        model_type: 模型类型 ('A2C', 'PPO', 'TD7')
        config: VerticalQueueConfig配置
        region_name: 区域名称
        n_episodes: 测试episode数量
        verbose: 是否打印详细信息

    Returns:
        dict: 测试结果
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
    episode_details = []

    # 运行n_episodes个episode
    for episode in range(n_episodes):
        obs, info = eval_env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

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

            # 防止无限循环
            if episode_length >= 1000:
                if verbose:
                    print(f"  ⚠️  Episode {episode+1} 达到最大步数限制 (1000)")
                break

        # 记录结果
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_details.append({
            'episode': episode + 1,
            'reward': float(episode_reward),
            'length': int(episode_length)
        })

        if verbose:
            print(f"  Episode {episode+1}/{n_episodes}: Reward = {episode_reward:.2f}, Length = {episode_length}")

    # 计算统计结果
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)

    results = {
        'model_type': model_type,
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
        print(f"\n📊 {model_type} @ {region_name} 测试结果:")
        print(f"   平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"   平均长度: {mean_length:.1f}")

    # 清理环境
    eval_env.close()

    return results


def main():
    """主函数：测试所有3个模型在所有异质性区域的泛化性能"""

    print("\n" + "="*80)
    print("Top 3 模型跨区域泛化性测试")
    print("Cross-Region Generalization Test for Top 3 Models (A2C, PPO, TD7)")
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
    print("\n第3步：在每个区域运行泛化测试")
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

    # ========== 第4步：汇总结果 ==========
    print("\n" + "="*80)
    print("测试完成！汇总结果")
    print("="*80 + "\n")

    # 打印各模型在各区域的性能对比
    print(f"{'区域':<30} {'A2C':<20} {'PPO':<20} {'TD7':<20}")
    print("-"*90)

    baseline_rewards = {}  # 记录各模型在baseline region的性能

    for region_name in all_configs.keys():
        a2c_reward = all_results['A2C'][region_name]['mean_reward']
        ppo_reward = all_results['PPO'][region_name]['mean_reward']
        td7_reward = all_results['TD7'][region_name]['mean_reward']

        # 记录baseline性能
        if 'Standard' in region_name:
            baseline_rewards['A2C'] = a2c_reward
            baseline_rewards['PPO'] = ppo_reward
            baseline_rewards['TD7'] = td7_reward

        print(f"{region_name:<30} {a2c_reward:<20.2f} {ppo_reward:<20.2f} {td7_reward:<20.2f}")

    # 打印性能下降百分比
    print("\n" + "="*80)
    print("性能下降百分比 (相对于Region A Baseline)")
    print("="*80 + "\n")

    print(f"{'区域':<30} {'A2C':<20} {'PPO':<20} {'TD7':<20}")
    print("-"*90)

    for region_name in all_configs.keys():
        if 'Standard' in region_name:
            print(f"{region_name:<30} {'0.0%':<20} {'0.0%':<20} {'0.0%':<20}")
        else:
            a2c_diff = ((all_results['A2C'][region_name]['mean_reward'] - baseline_rewards['A2C'])
                       / baseline_rewards['A2C'] * 100)
            ppo_diff = ((all_results['PPO'][region_name]['mean_reward'] - baseline_rewards['PPO'])
                       / baseline_rewards['PPO'] * 100)
            td7_diff = ((all_results['TD7'][region_name]['mean_reward'] - baseline_rewards['TD7'])
                       / baseline_rewards['TD7'] * 100)

            print(f"{region_name:<30} {a2c_diff:+.1f}%{' ':<15} {ppo_diff:+.1f}%{' ':<15} {td7_diff:+.1f}%")

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
    results_file = save_dir / "all_models_generalization_results.json"

    full_results = {
        'test_info': {
            'test_date': time.strftime("%Y-%m-%d %H:%M:%S"),
            'n_episodes_per_region': n_episodes_per_region,
            'total_time_seconds': total_time,
            'models_tested': ['A2C', 'PPO', 'TD7'],
            'regions_tested': list(all_configs.keys())
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

    # 保存汇总表格（CSV格式）
    summary_file = save_dir / "all_models_generalization_summary.csv"
    import csv

    with open(summary_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Region', 'A2C Mean', 'A2C Std', 'PPO Mean', 'PPO Std',
                        'TD7 Mean', 'TD7 Std', 'Best Model'])

        for region_name in all_configs.keys():
            a2c_res = all_results['A2C'][region_name]
            ppo_res = all_results['PPO'][region_name]
            td7_res = all_results['TD7'][region_name]

            # 找出最佳模型
            best_reward = max(a2c_res['mean_reward'], ppo_res['mean_reward'], td7_res['mean_reward'])
            if a2c_res['mean_reward'] == best_reward:
                best_model = 'A2C'
            elif ppo_res['mean_reward'] == best_reward:
                best_model = 'PPO'
            else:
                best_model = 'TD7'

            writer.writerow([
                region_name,
                f"{a2c_res['mean_reward']:.2f}",
                f"{a2c_res['std_reward']:.2f}",
                f"{ppo_res['mean_reward']:.2f}",
                f"{ppo_res['std_reward']:.2f}",
                f"{td7_res['mean_reward']:.2f}",
                f"{td7_res['std_reward']:.2f}",
                best_model
            ])

    print(f"✅ 汇总表格已保存到: {summary_file}")

    # 保存各模型的泛化性评分
    generalization_file = save_dir / "generalization_ranking.txt"

    with open(generalization_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("跨区域泛化性能排名\n")
        f.write("Cross-Region Generalization Performance Ranking\n")
        f.write("="*80 + "\n\n")

        # 计算平均性能下降
        avg_drop = {}
        for model_name in ['A2C', 'PPO', 'TD7']:
            drops = []
            for region_name in all_configs.keys():
                if 'Standard' not in region_name:
                    reward = all_results[model_name][region_name]['mean_reward']
                    baseline = baseline_rewards[model_name]
                    drop_pct = ((reward - baseline) / baseline) * 100
                    drops.append(drop_pct)
            avg_drop[model_name] = np.mean(drops)

        # 排序（下降越小越好）
        ranking = sorted(avg_drop.items(), key=lambda x: x[1], reverse=True)

        f.write("平均性能下降 (越小越好，表示泛化能力越强):\n")
        f.write("-"*80 + "\n")
        for rank, (model_name, drop) in enumerate(ranking, 1):
            f.write(f"{rank}. {model_name}: {drop:+.2f}%\n")

        f.write("\n" + "="*80 + "\n")
        f.write("各区域最佳模型:\n")
        f.write("-"*80 + "\n")

        for region_name in all_configs.keys():
            rewards = {
                'A2C': all_results['A2C'][region_name]['mean_reward'],
                'PPO': all_results['PPO'][region_name]['mean_reward'],
                'TD7': all_results['TD7'][region_name]['mean_reward']
            }
            best = max(rewards.items(), key=lambda x: x[1])
            f.write(f"{region_name}: {best[0]} ({best[1]:.2f})\n")

    print(f"✅ 泛化性排名已保存到: {generalization_file}")

    print("\n" + "="*80)
    print("✅ 所有模型泛化性测试全部完成！")
    print("="*80 + "\n")

    print("📌 关键发现:")
    print(f"   Baseline性能 (Region A):")
    print(f"     - A2C: {baseline_rewards['A2C']:.2f}")
    print(f"     - PPO: {baseline_rewards['PPO']:.2f}")
    print(f"     - TD7: {baseline_rewards['TD7']:.2f}")

    print(f"\n   平均性能下降:")
    for rank, (model_name, drop) in enumerate(ranking, 1):
        print(f"     {rank}. {model_name}: {drop:+.2f}% {'(泛化能力最强)' if rank == 1 else ''}")

    print("\n💡 下一步：")
    print("   1. 查看详细结果: cat generalization_results/all_models_generalization_results.json")
    print("   2. 查看汇总表格: cat generalization_results/all_models_generalization_summary.csv")
    print("   3. 查看泛化排名: cat generalization_results/generalization_ranking.txt")
    print("   4. 绘制可视化图表")


if __name__ == "__main__":
    main()
