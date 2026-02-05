"""
Train Top 3 Models for Cross-Region Generalization Testing

Train and save:
- A2C (Rank 1): 4437.86
- PPO (Rank 2): 4419.98
- TD7 (Rank 3): 4351.84 (already exists)

For generalization experiments in rpTransition project
"""

import sys
import os
import time
import numpy as np

# Add path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from algorithms.baselines.sb3_a2c_baseline import SB3A2CBaseline
from algorithms.baselines.sb3_ppo_baseline import SB3PPOBaseline


def train_a2c(timesteps=500000, save_path="../../Models/a2c/a2c_model_500000"):
    """
    Train A2C model (RP1 Rank 1)

    Parameters:
        timesteps: Training steps (default 500k)
        save_path: Model save path

    Returns:
        Evaluation results dictionary
    """
    print("\n" + "="*80)
    print("Training A2C model (RP1 Rank 1, mean reward 4437.86)")
    print("="*80 + "\n")

    # Create save directory
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Create A2C baseline
    a2c = SB3A2CBaseline()

    # Train
    print(f"Starting A2C training, total steps: {timesteps:,}")
    print(f"Configuration: Delayed cosine annealing learning rate (first 300k fixed at 7e-4, last 200k annealed to 1e-5)")
    print(f"Network: [512, 512, 256]")
    print()

    start_time = time.time()
    a2c.train(total_timesteps=timesteps)
    training_time = time.time() - start_time

    print(f"\nTraining completed! Time: {training_time:.1f}s ({training_time/60:.1f} minutes)")

    # Evaluation
    print("\n" + "-"*80)
    print("Evaluating A2C model performance...")
    print("-"*80)

    eval_results = a2c.evaluate(n_episodes=20, deterministic=True, verbose=True)

    print("\n" + "="*80)
    print(f"A2C Evaluation Results:")
    print(f"  Mean reward: {eval_results['mean_reward']:.2f} +/- {eval_results['std_reward']:.2f}")
    print(f"  Mean episode length: {eval_results['mean_length']:.1f}")
    print(f"  Expected value: 4437.86 (RP1 record)")
    print(f"  Difference: {eval_results['mean_reward'] - 4437.86:.2f}")
    print("="*80 + "\n")

    # Save model
    print(f"Saving A2C model to: {save_path}")
    a2c.save(save_path)

    # Return results
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
    Train PPO model (RP1 Rank 2)

    Parameters:
        timesteps: Training steps (default 500k)
        save_path: Model save path

    Returns:
        Evaluation results dictionary
    """
    print("\n" + "="*80)
    print("Training PPO model (RP1 Rank 2, mean reward 4419.98)")
    print("="*80 + "\n")

    # Create save directory
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Create PPO baseline
    ppo = SB3PPOBaseline()

    # Train
    print(f"Starting PPO training, total steps: {timesteps:,}")
    print(f"Configuration: Cosine annealing learning rate (3e-4 → 1e-6)")
    print(f"n_steps: 2048, batch_size: 64, n_epochs: 10")
    print()

    start_time = time.time()
    ppo.train(total_timesteps=timesteps)
    training_time = time.time() - start_time

    print(f"\nTraining completed! Time: {training_time:.1f}s ({training_time/60:.1f} minutes)")

    # Evaluation
    print("\n" + "-"*80)
    print("Evaluating PPO model performance...")
    print("-"*80)

    eval_results = ppo.evaluate(n_episodes=20, deterministic=True, verbose=True)

    print("\n" + "="*80)
    print(f"PPO Evaluation Results:")
    print(f"  Mean reward: {eval_results['mean_reward']:.2f} +/- {eval_results['std_reward']:.2f}")
    print(f"  Mean episode length: {eval_results['mean_length']:.1f}")
    print(f"  Expected value: 4419.98 (RP1 record)")
    print(f"  Difference: {eval_results['mean_reward'] - 4419.98:.2f}")
    print("="*80 + "\n")

    # Save model
    print(f"Saving PPO model to: {save_path}")
    ppo.save(save_path)

    # Return results
    return {
        'algorithm': 'PPO',
        'training_time': training_time,
        'mean_reward': eval_results['mean_reward'],
        'std_reward': eval_results['std_reward'],
        'mean_length': eval_results['mean_length'],
        'model_path': save_path
    }


def main():
    """Main function: Train Top 3 models"""
    print("\n" + "="*80)
    print("Train Top 3 Models for Cross-Region Generalization Testing")
    print("="*80 + "\n")

    print("RP1 Algorithm Rankings (500k steps):")
    print("  1st A2C v3:  4437.86 +/- 128.41 (delayed cosine annealing)")
    print("  2nd PPO:     4419.98 +/- 135.71 (cosine annealing)")
    print("  3rd TD7:     4351.84 +/- 51.07  (existing model)")
    print()

    # Ask user which models to train
    print("Select models to train:")
    print("  1. Train A2C only")
    print("  2. Train PPO only")
    print("  3. Train both A2C and PPO")
    print("  4. Skip all (use existing models)")

    choice = input("\nEnter choice (1-4, default 3): ").strip()
    if not choice:
        choice = "3"

    results = []

    # Train A2C
    if choice in ["1", "3"]:
        try:
            a2c_result = train_a2c(timesteps=500000)
            results.append(a2c_result)
        except Exception as e:
            print(f"\nA2C training failed: {e}")

    # Train PPO
    if choice in ["2", "3"]:
        try:
            ppo_result = train_ppo(timesteps=500000)
            results.append(ppo_result)
        except Exception as e:
            print(f"\nPPO training failed: {e}")

    # Summary
    if results:
        print("\n" + "="*80)
        print("Training Summary")
        print("="*80)

        for result in results:
            print(f"\n{result['algorithm']}:")
            print(f"  Training time: {result['training_time']:.1f}s ({result['training_time']/60:.1f} minutes)")
            print(f"  Mean reward: {result['mean_reward']:.2f} +/- {result['std_reward']:.2f}")
            print(f"  Model path: {result['model_path']}")

        print("\n" + "="*80)
        print("All models trained successfully!")
        print("="*80)

        # Save summary
        import json
        summary_path = "../../Models/top3_training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nTraining summary saved to: {summary_path}")

    else:
        print("\nSkipped training, using existing models")

    print("\nNext steps:")
    print("  1. Check model files:")
    print("     - ./models/a2c/a2c_model_500000.zip")
    print("     - ./models/ppo/ppo_model_500000.zip")
    print("     - ./models/td7/td7_model_500000.pt")
    print()
    print("  2. Run generalization test:")
    print("     cd ../rpTransition")
    print("     python cross_region_generalization_test_top3.py")
    print()


if __name__ == "__main__":
    main()
