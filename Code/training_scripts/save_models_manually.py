"""
手动保存已训练的A2C和PPO模型
使用PyTorch的state_dict直接保存，绕过pickle问题
"""
import sys
import os
import torch
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from algorithms.baselines.sb3_a2c_baseline import SB3A2CBaseline
from algorithms.baselines.sb3_ppo_baseline import SB3PPOBaseline

def save_model_state_dict(model, save_path, algorithm_name):
    """
    保存模型的PyTorch state dict和必要的配置
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 保存模型参数
    model_state = {
        'policy_state_dict': model.policy.state_dict(),
        'observation_space': model.observation_space,
        'action_space': model.action_space,
        'n_envs': model.n_envs,
        '_n_updates': model._n_updates,
    }

    # 算法特定参数
    if hasattr(model, 'learning_rate'):
        model_state['learning_rate'] = model.learning_rate
    if hasattr(model, 'gamma'):
        model_state['gamma'] = model.gamma
    if hasattr(model, 'n_steps'):
        model_state['n_steps'] = model.n_steps

    torch.save(model_state, save_path + '.pth')
    print(f"✅ {algorithm_name} model saved to: {save_path}.pth")

    # 验证文件大小
    size = os.path.getsize(save_path + '.pth')
    print(f"   文件大小: {size / 1024:.2f} KB")

    return size > 0

def main():
    print("\n" + "="*80)
    print("手动保存A2C和PPO模型（绕过pickle问题）")
    print("="*80 + "\n")

    # 训练并保存A2C
    print("1. 训练A2C模型...")
    a2c = SB3A2CBaseline()
    a2c.train(total_timesteps=500000)

    print("\n评估A2C模型...")
    eval_results = a2c.evaluate(n_episodes=20, deterministic=True, verbose=True)
    print(f"A2C评估结果: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")

    print("\n保存A2C模型state dict...")
    a2c_path = "../../Models/a2c/a2c_model_500000"
    save_model_state_dict(a2c.model, a2c_path, "A2C")

    # 训练并保存PPO
    print("\n" + "-"*80)
    print("2. 训练PPO模型...")
    ppo = SB3PPOBaseline()
    ppo.train(total_timesteps=500000)

    print("\n评估PPO模型...")
    eval_results = ppo.evaluate(n_episodes=20, deterministic=True, verbose=True)
    print(f"PPO评估结果: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")

    print("\n保存PPO模型state dict...")
    ppo_path = "../../Models/ppo/ppo_model_500000"
    save_model_state_dict(ppo.model, ppo_path, "PPO")

    print("\n" + "="*80)
    print("✅ 所有模型已成功保存！")
    print("="*80)

    print("\n模型文件:")
    print(f"  A2C: {a2c_path}.pth")
    print(f"  PPO: {ppo_path}.pth")

    print("\n注意: 这些是PyTorch state dict文件，加载时需要重新创建环境和模型结构")

if __name__ == "__main__":
    main()
