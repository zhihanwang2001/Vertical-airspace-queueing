"""
训练Top 3模型用于跨区域泛化性测试
Train Top 3 Models for Cross-Region Generalization Testing

训练并保存:
- A2C (排名第1): 4437.86
- PPO (排名第2): 4419.98
- TD7 (排名第3): 4351.84 (已存在)

用于rpTransition项目的泛化性实验
"""

import sys
import os
import time
import numpy as np

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from algorithms.baselines.sb3_a2c_baseline import SB3A2CBaseline
from algorithms.baselines.sb3_ppo_baseline import SB3PPOBaseline


def train_a2c(timesteps=500000, save_path="../../Models/a2c/a2c_model_500000"):
    """
    训练A2C模型（RP1排名第1）

    参数:
        timesteps: 训练步数（默认500k）
        save_path: 模型保存路径

    返回:
        评估结果字典
    """
    print("\n" + "="*80)
    print("训练A2C模型（RP1排名第1，平均奖励4437.86）")
    print("="*80 + "\n")

    # 创建保存目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 创建A2C baseline
    a2c = SB3A2CBaseline()

    # 训练
    print(f"开始训练 A2C，总步数: {timesteps:,}")
    print(f"配置: 延迟余弦退火学习率（前300k固定7e-4，后200k退火至1e-5）")
    print(f"网络: [512, 512, 256]")
    print()

    start_time = time.time()
    a2c.train(total_timesteps=timesteps)
    training_time = time.time() - start_time

    print(f"\n训练完成！用时: {training_time:.1f}秒 ({training_time/60:.1f}分钟)")

    # 评估
    print("\n" + "-"*80)
    print("评估A2C模型性能...")
    print("-"*80)

    eval_results = a2c.evaluate(n_episodes=20, deterministic=True, verbose=True)

    print("\n" + "="*80)
    print(f"A2C评估结果:")
    print(f"  平均奖励: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"  平均回合长度: {eval_results['mean_length']:.1f}")
    print(f"  期望值: 4437.86 (RP1记录)")
    print(f"  差异: {eval_results['mean_reward'] - 4437.86:.2f}")
    print("="*80 + "\n")

    # 保存模型
    print(f"保存A2C模型到: {save_path}")
    a2c.save(save_path)

    # 返回结果
    return {
        'algorithm': 'A2C',
        'training_time': training_time,
        'mean_reward': eval_results['mean_reward'],
        'std_reward': eval_results['std_reward'],
        'mean_length': eval_results['mean_length'],
        'model_path': save_path
    }


def train_ppo(timesteps=500000, save_path="../../Models/ppo/ppo_model_500000"):
    """
    训练PPO模型（RP1排名第2）

    参数:
        timesteps: 训练步数（默认500k）
        save_path: 模型保存路径

    返回:
        评估结果字典
    """
    print("\n" + "="*80)
    print("训练PPO模型（RP1排名第2，平均奖励4419.98）")
    print("="*80 + "\n")

    # 创建保存目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 创建PPO baseline
    ppo = SB3PPOBaseline()

    # 训练
    print(f"开始训练 PPO，总步数: {timesteps:,}")
    print(f"配置: 余弦退火学习率（3e-4 → 1e-6）")
    print(f"n_steps: 2048, batch_size: 64, n_epochs: 10")
    print()

    start_time = time.time()
    ppo.train(total_timesteps=timesteps)
    training_time = time.time() - start_time

    print(f"\n训练完成！用时: {training_time:.1f}秒 ({training_time/60:.1f}分钟)")

    # 评估
    print("\n" + "-"*80)
    print("评估PPO模型性能...")
    print("-"*80)

    eval_results = ppo.evaluate(n_episodes=20, deterministic=True, verbose=True)

    print("\n" + "="*80)
    print(f"PPO评估结果:")
    print(f"  平均奖励: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"  平均回合长度: {eval_results['mean_length']:.1f}")
    print(f"  期望值: 4419.98 (RP1记录)")
    print(f"  差异: {eval_results['mean_reward'] - 4419.98:.2f}")
    print("="*80 + "\n")

    # 保存模型
    print(f"保存PPO模型到: {save_path}")
    ppo.save(save_path)

    # 返回结果
    return {
        'algorithm': 'PPO',
        'training_time': training_time,
        'mean_reward': eval_results['mean_reward'],
        'std_reward': eval_results['std_reward'],
        'mean_length': eval_results['mean_length'],
        'model_path': save_path
    }


def main():
    """主函数：训练Top 3模型"""
    print("\n" + "🎯"*40)
    print("训练Top 3模型用于跨区域泛化性测试")
    print("Train Top 3 Models for Cross-Region Generalization")
    print("🎯"*40 + "\n")

    print("RP1算法排名（500k steps）:")
    print("  🥇 A2C v3:  4437.86 ± 128.41 (延迟余弦退火)")
    print("  🥈 PPO:     4419.98 ± 135.71 (余弦退火)")
    print("  🥉 TD7:     4351.84 ± 51.07  (已存在模型)")
    print()

    # 询问用户要训练哪些模型
    print("选择要训练的模型:")
    print("  1. 只训练A2C")
    print("  2. 只训练PPO")
    print("  3. 训练A2C和PPO")
    print("  4. 全部跳过（使用现有模型）")

    choice = input("\n请输入选择 (1-4，默认3): ").strip()
    if not choice:
        choice = "3"

    results = []

    # 训练A2C
    if choice in ["1", "3"]:
        try:
            a2c_result = train_a2c(timesteps=500000)
            results.append(a2c_result)
        except Exception as e:
            print(f"\n❌ A2C训练失败: {e}")

    # 训练PPO
    if choice in ["2", "3"]:
        try:
            ppo_result = train_ppo(timesteps=500000)
            results.append(ppo_result)
        except Exception as e:
            print(f"\n❌ PPO训练失败: {e}")

    # 总结
    if results:
        print("\n" + "="*80)
        print("训练总结")
        print("="*80)

        for result in results:
            print(f"\n{result['algorithm']}:")
            print(f"  训练时间: {result['training_time']:.1f}秒 ({result['training_time']/60:.1f}分钟)")
            print(f"  平均奖励: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
            print(f"  模型路径: {result['model_path']}")

        print("\n" + "="*80)
        print("✅ 所有模型训练完成！")
        print("="*80)

        # 保存总结
        import json
        summary_path = "../../Models/top3_training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n训练总结已保存到: {summary_path}")

    else:
        print("\n⏭️  跳过训练，使用现有模型")

    print("\n下一步:")
    print("  1. 检查模型文件:")
    print("     - ./models/a2c/a2c_model_500000.zip")
    print("     - ./models/ppo/ppo_model_500000.zip")
    print("     - ./models/td7/td7_model_500000.pt")
    print()
    print("  2. 运行泛化性测试:")
    print("     cd ../rpTransition")
    print("     python cross_region_generalization_test_top3.py")
    print()


if __name__ == "__main__":
    main()
