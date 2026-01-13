"""
简化版：只运行K=30 A2C seed 42（最关键实验）
"""
import sys
sys.path.insert(0, '../..')
from Code.training_scripts.major_revision_exp1_extended_training import train_and_evaluate

if __name__ == "__main__":
    print("="*80)
    print("Running CRITICAL EXPERIMENT: K=30 A2C seed 42")
    print("="*80)

    result = train_and_evaluate(
        algo_name='A2C',
        capacity_type='k30_uniform',
        seed=42,
        total_timesteps=1_000_000,
        eval_freq=10_000,
        n_eval_episodes=10
    )

    if result:
        print("\n" + "="*80)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print(f"Final reward: {result['final_eval']['mean_reward']:.1f}")
        print(f"Crash rate: {result['final_eval']['crash_rate']:.1f}%")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("EXPERIMENT FAILED!")
        print("="*80)
