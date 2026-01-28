"""
Manually save trained A2C and PPO models
Use PyTorch's state_dict to save directly, bypassing pickle issues
"""
import sys
import os
import torch
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from algorithms.baselines.sb3_a2c_baseline import SB3A2CBaseline
from algorithms.baselines.sb3_ppo_baseline import SB3PPOBaseline

def save_model_state_dict(model, save_path, algorithm_name):
    """
    Save model's PyTorch state dict and necessary configuration
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save model parameters
    model_state = {
        'policy_state_dict': model.policy.state_dict(),
        'observation_space': model.observation_space,
        'action_space': model.action_space,
        'n_envs': model.n_envs,
        '_n_updates': model._n_updates,
    }

    # Algorithm-specific parameters
    if hasattr(model, 'learning_rate'):
        model_state['learning_rate'] = model.learning_rate
    if hasattr(model, 'gamma'):
        model_state['gamma'] = model.gamma
    if hasattr(model, 'n_steps'):
        model_state['n_steps'] = model.n_steps

    torch.save(model_state, save_path + '.pth')
    print(f"{algorithm_name} model saved to: {save_path}.pth")

    # Verify file size
    size = os.path.getsize(save_path + '.pth')
    print(f"   File size: {size / 1024:.2f} KB")

    return size > 0

def main():
    print("\n" + "="*80)
    print("Manually save A2C and PPO models (bypassing pickle issues)")
    print("="*80 + "\n")

    # Train and save A2C
    print("1. Training A2C model...")
    a2c = SB3A2CBaseline()
    a2c.train(total_timesteps=500000)

    print("\nEvaluating A2C model...")
    eval_results = a2c.evaluate(n_episodes=20, deterministic=True, verbose=True)
    print(f"A2C evaluation results: {eval_results['mean_reward']:.2f} +/- {eval_results['std_reward']:.2f}")

    print("\nSaving A2C model state dict...")
    a2c_path = "../../Models/a2c/a2c_model_500000"
    save_model_state_dict(a2c.model, a2c_path, "A2C")

    # Train and save PPO
    print("\n" + "-"*80)
    print("2. Training PPO model...")
    ppo = SB3PPOBaseline()
    ppo.train(total_timesteps=500000)

    print("\nEvaluating PPO model...")
    eval_results = ppo.evaluate(n_episodes=20, deterministic=True, verbose=True)
    print(f"PPO evaluation results: {eval_results['mean_reward']:.2f} +/- {eval_results['std_reward']:.2f}")

    print("\nSaving PPO model state dict...")
    ppo_path = "../../Models/ppo/ppo_model_500000"
    save_model_state_dict(ppo.model, ppo_path, "PPO")

    print("\n" + "="*80)
    print("All models saved successfully!")
    print("="*80)

    print("\nModel files:")
    print(f"  A2C: {a2c_path}.pth")
    print(f"  PPO: {ppo_path}.pth")

    print("\nNote: These are PyTorch state dict files, loading requires recreating environment and model structure")

if __name__ == "__main__":
    main()
