"""
Run baseline algorithm comparison experiments
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env.drl_optimized_env_fixed import DRLOptimizedQueueEnvFixed
from algorithms.baselines.comparison_runner import ComparisonRunner
import argparse


def main():
    parser = argparse.ArgumentParser(description='Run baseline algorithm comparison')
    parser.add_argument('--algorithms', nargs='+',
                       default=['SB3_PPO', 'SB3_SAC', 'SB3_TD3', 'A2C', 'Heuristic'],
                       help='Algorithms to compare')
    parser.add_argument('--timesteps', type=int, default=1000000,
                       help='Training timesteps per algorithm')
    parser.add_argument('--eval-episodes', type=int, default=20,
                       help='Number of evaluation episodes')
    parser.add_argument('--runs', type=int, default=1,
                       help='Number of runs per algorithm')
    parser.add_argument('--save-dir', type=str, default='../../Results/comparison',
                       help='Directory to save results')
    parser.add_argument('--eval-freq', type=int, default=100000,
                       help='Evaluation frequency for long training')

    args = parser.parse_args()

    print("Initializing environment...")

    # Create environment
    env = DRLOptimizedQueueEnvFixed()

    print(f"Environment initialized")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Create comparison runner
    runner = ComparisonRunner(env, save_dir=args.save_dir)

    # Run comparison experiments
    print(f"\nStarting comparison with algorithms: {args.algorithms}")
    print(f"Using optimized configurations for better convergence...")

    results = runner.run_comparison(
        algorithms=args.algorithms,
        total_timesteps=args.timesteps,
        n_eval_episodes=args.eval_episodes,
        n_runs=args.runs,
        eval_freq=args.eval_freq
    )

    # Save data
    runner.save_comparison_data()

    print(f"\nComparison completed! Results saved to: {args.save_dir}")
    print(f"Check the following files:")
    print(f"  - comparison_report.txt: Detailed text report")
    print(f"  - training_curves.png: Training progress visualization")
    print(f"  - performance_boxplot.png: Performance distribution")
    print(f"  - training_time_comparison.png: Training efficiency")
    print(f"  - performance_radar.png: Multi-metric comparison")
    print(f"  - comparison_data.json: Raw comparison data")


if __name__ == "__main__":
    main()