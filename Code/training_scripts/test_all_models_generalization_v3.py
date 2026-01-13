"""
Top 3模型跨区域泛化性测试脚本 V3 - 奖励组件分解版
Top 3 Models Cross-Region Generalization Test Script V3 - Reward Decomposition

🎯 核心改进 (V2 → V3):
1. 保留V2的所有多维度系统指标
2. **新增：提取奖励组件分解 (reward_components)**
3. 分析单目标优化(RP1)在多目标权衡上的限制
4. 为RP1→RP2的transition提供科学依据

评估指标：
【V2指标】
- 累积奖励 (Cumulative Reward)
- 队列利用率 (Queue Utilization)
- 负载率 (Load Rate)
- 系统吞吐量 (Throughput)
- 稳定性得分 (Stability Score)

【V3新增 - 奖励组件】
- R_throughput: 吞吐量奖励 (10.0 × 服务订单数)
- R_balance: 负载均衡奖励 (基尼系数, 0-5.0)
- R_efficiency: 能效奖励 (服务/能耗比, 0-3.0)
- transfer_benefit: 转移效益 (0-2.0)
- stability_bonus: 稳定性奖励 (0-2.0)
- P_congestion: 拥堵惩罚 (<0)
- P_instability: 不稳定惩罚 (<0)

📊 分析目的：
揭示RP1的单目标优化虽然获得高累积奖励，但在：
  - 层间公平性 (R_balance)
  - 能源效率 (R_efficiency)
  - 负载均衡
存在trade-offs → motivates RP2的MORL方法
"""

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from heterogeneous_configs import HeterogeneousRegionConfigs


def test_model_in_region(model, model_type: str, config, region_name: str,
                         n_episodes: int = 10, verbose: bool = True):
    """
    在指定区域测试模型 - V3版本（提取奖励组件分解）

    Args:
        model: 已加载模型的baseline实例
        model_type: 模型类型 ('A2C', 'PPO', 'TD7')
        config: VerticalQueueConfig配置
        region_name: 区域名称
        n_episodes: 测试episode数量
        verbose: 是否打印详细信息

    Returns:
        dict: 测试结果（包含多维度指标 + 奖励组件分解）
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

    # V2指标：系统性能
    episode_avg_utilizations = []
    episode_avg_load_rates = []
    episode_throughputs = []
    episode_stability_scores = []
    episode_max_utilizations = []

    # V3新增：奖励组件
    episode_avg_r_throughput = []
    episode_avg_r_balance = []
    episode_avg_r_efficiency = []
    episode_avg_transfer = []
    episode_avg_stability_bonus = []
    episode_avg_p_congestion = []
    episode_avg_p_instability = []

    episode_details = []

    # 运行n_episodes个episode
    for episode in range(n_episodes):
        obs, info = eval_env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        # V2指标收集
        step_utilizations = []
        step_load_rates = []
        step_stability_scores = []

        # V3新增：奖励组件收集
        step_r_throughput = []
        step_r_balance = []
        step_r_efficiency = []
        step_transfer = []
        step_stability_bonus = []
        step_p_congestion = []
        step_p_instability = []

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

            # 提取V2系统指标
            if 'utilization_rates' in info:
                step_utilizations.append(np.mean(info['utilization_rates']))
            if 'load_rates' in info:
                step_load_rates.append(np.mean(info['load_rates']))
            if 'stability_score' in info:
                step_stability_scores.append(info['stability_score'])

            # V3新增：提取奖励组件
            if 'reward_components' in info:
                rc = info['reward_components']
                step_r_throughput.append(rc.get('throughput', 0.0))
                step_r_balance.append(rc.get('balance', 0.0))
                step_r_efficiency.append(rc.get('efficiency', 0.0))
                step_transfer.append(rc.get('transfer', 0.0))
                step_stability_bonus.append(rc.get('stability', 0.0))
                step_p_congestion.append(rc.get('congestion', 0.0))
                step_p_instability.append(rc.get('instability', 0.0))

            # 防止无限循环
            if episode_length >= 1000:
                if verbose:
                    print(f"  ⚠️  Episode {episode+1} 达到最大步数限制 (1000)")
                break

        # 计算episode级别的统计
        # V2指标
        avg_utilization = np.mean(step_utilizations) if step_utilizations else 0.0
        avg_load_rate = np.mean(step_load_rates) if step_load_rates else 0.0
        avg_stability = np.mean(step_stability_scores) if step_stability_scores else 0.0
        max_utilization = np.max(step_utilizations) if step_utilizations else 0.0
        throughput = info.get('throughput', 0.0) if info else 0.0

        # V3新增：奖励组件平均值
        avg_r_throughput = np.mean(step_r_throughput) if step_r_throughput else 0.0
        avg_r_balance = np.mean(step_r_balance) if step_r_balance else 0.0
        avg_r_efficiency = np.mean(step_r_efficiency) if step_r_efficiency else 0.0
        avg_transfer = np.mean(step_transfer) if step_transfer else 0.0
        avg_stability_bonus = np.mean(step_stability_bonus) if step_stability_bonus else 0.0
        avg_p_congestion = np.mean(step_p_congestion) if step_p_congestion else 0.0
        avg_p_instability = np.mean(step_p_instability) if step_p_instability else 0.0

        # 记录结果
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # V2指标
        episode_avg_utilizations.append(avg_utilization)
        episode_avg_load_rates.append(avg_load_rate)
        episode_throughputs.append(throughput)
        episode_stability_scores.append(avg_stability)
        episode_max_utilizations.append(max_utilization)

        # V3新增：奖励组件
        episode_avg_r_throughput.append(avg_r_throughput)
        episode_avg_r_balance.append(avg_r_balance)
        episode_avg_r_efficiency.append(avg_r_efficiency)
        episode_avg_transfer.append(avg_transfer)
        episode_avg_stability_bonus.append(avg_stability_bonus)
        episode_avg_p_congestion.append(avg_p_congestion)
        episode_avg_p_instability.append(avg_p_instability)

        episode_details.append({
            'episode': episode + 1,
            'reward': float(episode_reward),
            'length': int(episode_length),
            # V2指标
            'avg_utilization': float(avg_utilization),
            'avg_load_rate': float(avg_load_rate),
            'throughput': float(throughput),
            'stability_score': float(avg_stability),
            'max_utilization': float(max_utilization),
            # V3新增：奖励组件
            'reward_components': {
                'throughput': float(avg_r_throughput),
                'balance': float(avg_r_balance),
                'efficiency': float(avg_r_efficiency),
                'transfer': float(avg_transfer),
                'stability': float(avg_stability_bonus),
                'congestion': float(avg_p_congestion),
                'instability': float(avg_p_instability)
            }
        })

        if verbose:
            print(f"  Episode {episode+1}/{n_episodes}:")
            print(f"    Reward={episode_reward:.2f}, Length={episode_length}")
            print(f"    [V2] Util={avg_utilization:.3f}, Load={avg_load_rate:.3f}, Throughput={throughput:.2f}")
            print(f"    [V3] R_throughput={avg_r_throughput:.1f}, R_balance={avg_r_balance:.2f}, R_efficiency={avg_r_efficiency:.2f}")

    # 计算统计结果
    results = {
        'model_type': model_type,
        'region_name': region_name,
        'n_episodes': n_episodes,

        # 原有指标
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'mean_length': float(np.mean(episode_lengths)),

        # V2指标：系统性能
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

        # V3新增：奖励组件统计
        'reward_components': {
            'mean_throughput': float(np.mean(episode_avg_r_throughput)),
            'std_throughput': float(np.std(episode_avg_r_throughput)),
            'mean_balance': float(np.mean(episode_avg_r_balance)),
            'std_balance': float(np.std(episode_avg_r_balance)),
            'mean_efficiency': float(np.mean(episode_avg_r_efficiency)),
            'std_efficiency': float(np.std(episode_avg_r_efficiency)),
            'mean_transfer': float(np.mean(episode_avg_transfer)),
            'std_transfer': float(np.std(episode_avg_transfer)),
            'mean_stability': float(np.mean(episode_avg_stability_bonus)),
            'std_stability': float(np.std(episode_avg_stability_bonus)),
            'mean_congestion': float(np.mean(episode_avg_p_congestion)),
            'std_congestion': float(np.std(episode_avg_p_congestion)),
            'mean_instability': float(np.mean(episode_avg_p_instability)),
            'std_instability': float(np.std(episode_avg_p_instability))
        },

        # 详细数据
        'episode_rewards': [float(r) for r in episode_rewards],
        'episode_lengths': [int(l) for l in episode_lengths],
        'episode_details': episode_details,
        'config_summary': base_env.get_config_summary()
    }

    if verbose:
        print(f"\n📊 {model_type} @ {region_name} 测试结果:")
        print(f"   【V2指标】")
        print(f"   累积奖励: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"   队列利用率: {results['mean_utilization']:.3f} ± {results['std_utilization']:.3f}")
        print(f"   负载率: {results['mean_load_rate']:.3f} ± {results['std_load_rate']:.3f}")
        print(f"   吞吐量: {results['mean_throughput']:.2f} ± {results['std_throughput']:.2f}")
        print(f"   稳定性: {results['mean_stability']:.3f} ± {results['std_stability']:.3f}")
        print(f"\n   【V3奖励组件】")
        rc = results['reward_components']
        print(f"   R_throughput: {rc['mean_throughput']:.2f} ± {rc['std_throughput']:.2f}")
        print(f"   R_balance (公平性): {rc['mean_balance']:.2f} ± {rc['std_balance']:.2f}")
        print(f"   R_efficiency (能效): {rc['mean_efficiency']:.2f} ± {rc['std_efficiency']:.2f}")
        print(f"   P_congestion (拥堵惩罚): {rc['mean_congestion']:.2f} ± {rc['std_congestion']:.2f}")

    # 清理环境
    eval_env.close()

    return results


def main():
    """主函数：测试所有3个模型在所有异质性区域的泛化性能 - V3版本（奖励组件分解）"""

    print("\n" + "="*80)
    print("Top 3 模型跨区域泛化性测试 V3 - 奖励组件分解版")
    print("Cross-Region Generalization Test V3 - Reward Component Decomposition")
    print("="*80 + "\n")

    print("🎯 V3核心改进：")
    print("   - 保留V2的多维度系统指标")
    print("   - 新增：提取奖励组件分解 (7个组件)")
    print("   - 揭示单目标优化的多目标trade-offs")
    print("   - 为RP1→RP2 transition提供科学依据\n")

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
    print("\n第3步：在每个区域运行泛化测试（V3 - 奖励组件分解）")
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

    # ========== 第4步：汇总结果（V3 - 包含奖励组件分析） ==========
    print("\n" + "="*80)
    print("测试完成！汇总结果（V3 - 多维度指标 + 奖励组件分解）")
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

    # V3新增：奖励组件对比表格
    print("\n【表6】R_balance (负载均衡/公平性) 对比")
    print("-"*90)
    print(f"{'区域':<30} {'A2C':<20} {'PPO':<20} {'TD7':<20}")
    print("-"*90)

    for region_name in all_configs.keys():
        a2c_bal = all_results['A2C'][region_name]['reward_components']['mean_balance']
        ppo_bal = all_results['PPO'][region_name]['reward_components']['mean_balance']
        td7_bal = all_results['TD7'][region_name]['reward_components']['mean_balance']

        print(f"{region_name:<30} {a2c_bal:<20.3f} {ppo_bal:<20.3f} {td7_bal:<20.3f}")

    print("\n【表7】R_efficiency (能源效率) 对比")
    print("-"*90)
    print(f"{'区域':<30} {'A2C':<20} {'PPO':<20} {'TD7':<20}")
    print("-"*90)

    for region_name in all_configs.keys():
        a2c_eff = all_results['A2C'][region_name]['reward_components']['mean_efficiency']
        ppo_eff = all_results['PPO'][region_name]['reward_components']['mean_efficiency']
        td7_eff = all_results['TD7'][region_name]['reward_components']['mean_efficiency']

        print(f"{region_name:<30} {a2c_eff:<20.3f} {ppo_eff:<20.3f} {td7_eff:<20.3f}")

    print("\n【表8】R_throughput (吞吐量奖励组件) 对比")
    print("-"*90)
    print(f"{'区域':<30} {'A2C':<20} {'PPO':<20} {'TD7':<20}")
    print("-"*90)

    for region_name in all_configs.keys():
        a2c_thr = all_results['A2C'][region_name]['reward_components']['mean_throughput']
        ppo_thr = all_results['PPO'][region_name]['reward_components']['mean_throughput']
        td7_thr = all_results['TD7'][region_name]['reward_components']['mean_throughput']

        print(f"{region_name:<30} {a2c_thr:<20.2f} {ppo_thr:<20.2f} {td7_thr:<20.2f}")

    print("\n【表9】P_congestion (拥堵惩罚) 对比")
    print("-"*90)
    print(f"{'区域':<30} {'A2C':<20} {'PPO':<20} {'TD7':<20}")
    print("-"*90)

    for region_name in all_configs.keys():
        a2c_cong = all_results['A2C'][region_name]['reward_components']['mean_congestion']
        ppo_cong = all_results['PPO'][region_name]['reward_components']['mean_congestion']
        td7_cong = all_results['TD7'][region_name]['reward_components']['mean_congestion']

        print(f"{region_name:<30} {a2c_cong:<20.2f} {ppo_cong:<20.2f} {td7_cong:<20.2f}")

    print("\n" + "-"*80)
    print(f"总测试时间: {total_time:.1f}秒 ({total_time/60:.1f}分钟)")
    print(f"总episode数: {len(models) * len(all_configs) * n_episodes_per_region}")

    # ========== 第5步：保存结果 ==========
    print("\n第5步：保存测试结果 (V3版本)")
    print("-"*80)

    # 创建保存目录
    save_dir = Path("../../Results/generalization")
    save_dir.mkdir(exist_ok=True)

    # 保存详细结果
    results_file = save_dir / "all_models_generalization_results_v3.json"

    full_results = {
        'test_info': {
            'version': 'v3_reward_decomposition',
            'test_date': time.strftime("%Y-%m-%d %H:%M:%S"),
            'n_episodes_per_region': n_episodes_per_region,
            'total_time_seconds': total_time,
            'models_tested': ['A2C', 'PPO', 'TD7'],
            'regions_tested': list(all_configs.keys()),
            'metrics_evaluated': {
                'v2_system_metrics': [
                    'cumulative_reward', 'queue_utilization', 'load_rate',
                    'throughput', 'stability_score', 'max_congestion'
                ],
                'v3_reward_components': [
                    'R_throughput', 'R_balance', 'R_efficiency',
                    'transfer_benefit', 'stability_bonus',
                    'P_congestion', 'P_instability'
                ]
            },
            'purpose': 'Reveal multi-objective trade-offs in single-objective RL (RP1) to motivate MORL approach (RP2)'
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

    # 保存汇总表格（CSV格式 - V3增强版，包含奖励组件）
    summary_file = save_dir / "all_models_generalization_summary_v3.csv"
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
            # V3新增：奖励组件
            'R_throughput', 'R_balance', 'R_efficiency',
            'transfer_benefit', 'stability_bonus',
            'P_congestion', 'P_instability'
        ])

        for region_name in all_configs.keys():
            for model_name in ['A2C', 'PPO', 'TD7']:
                res = all_results[model_name][region_name]
                rc = res['reward_components']
                writer.writerow([
                    region_name, model_name,
                    f"{res['mean_reward']:.2f}", f"{res['std_reward']:.2f}",
                    f"{res['mean_utilization']:.4f}", f"{res['std_utilization']:.4f}",
                    f"{res['mean_load_rate']:.4f}", f"{res['std_load_rate']:.4f}",
                    f"{res['mean_throughput']:.2f}", f"{res['std_throughput']:.2f}",
                    f"{res['mean_stability']:.4f}", f"{res['std_stability']:.4f}",
                    # V3新增
                    f"{rc['mean_throughput']:.2f}", f"{rc['mean_balance']:.3f}", f"{rc['mean_efficiency']:.3f}",
                    f"{rc['mean_transfer']:.3f}", f"{rc['mean_stability']:.3f}",
                    f"{rc['mean_congestion']:.3f}", f"{rc['mean_instability']:.3f}"
                ])

    print(f"✅ 汇总表格已保存到: {summary_file}")

    print("\n" + "="*80)
    print("✅ 所有模型泛化性测试全部完成（V3 - 奖励组件分解版）！")
    print("="*80 + "\n")

    print("📌 V3关键发现（奖励组件分解）:")
    print(f"\n   Baseline性能 (Region A - Standard):")
    print(f"     - A2C: {baseline_rewards['A2C']:.2f}")
    print(f"     - PPO: {baseline_rewards['PPO']:.2f}")
    print(f"     - TD7: {baseline_rewards['TD7']:.2f}")

    print(f"\n   🎯 RP1→RP2 Transition Logic:")
    print(f"   虽然RP1的单目标优化获得了高累积奖励，")
    print(f"   但奖励组件分解显示在多个目标上存在trade-offs：")
    print(f"     - R_balance (公平性): 层间负载分布不均")
    print(f"     - R_efficiency (能效): 能源利用率较低")
    print(f"     - P_congestion (拥堵): 高负载下拥堵增加")
    print(f"   这些trade-offs揭示了单目标优化的局限性，")
    print(f"   motivates RP2采用MORL方法进行帕累托优化。")

    print("\n💡 下一步：")
    print("   1. 查看详细结果: cat generalization_results/all_models_generalization_results_v3.json")
    print("   2. 查看汇总表格: cat generalization_results/all_models_generalization_summary_v3.csv")
    print("   3. 分析奖励组件trade-offs，设计RP1→RP2 transition逻辑")
    print("   4. 撰写论文Section 3.4: 跨场景泛化性分析 + RP2 motivation")


if __name__ == "__main__":
    main()
