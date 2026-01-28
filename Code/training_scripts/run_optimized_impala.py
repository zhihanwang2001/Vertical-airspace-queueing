"""
Run Optimized IMPALA for Queue System
Run optimized IMPALA algorithm for queue system

Usage:
python run_optimized_impala.py [--timesteps TIMESTEPS] [--eval] [--load PATH]
"""

import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from algorithms.advanced.impala.impala_optimized import OptimizedIMPALABaseline


def main():
    parser = argparse.ArgumentParser(description='Run Optimized IMPALA')
    parser.add_argument('--timesteps', type=int, default=500000,
                        help='Total training timesteps (default: 500000)')
    parser.add_argument('--eval', action='store_true',
                        help='Only evaluate pre-trained model')
    parser.add_argument('--load', type=str,
                        help='Path to load pre-trained model')
    parser.add_argument('--save_freq', type=int, default=50000,
                        help='Model save frequency (default: 50000)')
    parser.add_argument('--eval_freq', type=int, default=10000,
                        help='Evaluation frequency (default: 10000)')

    args = parser.parse_args()

    print("Running Optimized IMPALA for Vertical Stratified Queue System")
    print("=" * 80)

    # Create optimized IMPALA baseline
    config = {
        # Conservative hyperparameter settings
        'learning_rate': 5e-5,
        'rho_bar': 0.8,
        'c_bar': 0.8,
        'buffer_size': 50000,
        'sequence_length': 32,
        'batch_size': 32,
        'hidden_dim': 512,
        'num_layers': 3,
        'entropy_coeff': 0.02,
        'gradient_clip': 10.0,
        'learning_starts': 2000,
        'train_freq': 2
    }

    baseline = OptimizedIMPALABaseline(config=config)

    if args.load:
        print(f"Loading pre-trained model from: {args.load}")
        baseline.load(args.load)

    if args.eval:
        # Evaluation mode only
        print("Evaluation mode...")
        if not args.load:
            print("Error: Please specify --load path for evaluation mode")
            return

        eval_results = baseline.evaluate(n_episodes=20, deterministic=True, verbose=True)

        print("\nFinal Evaluation Results:")
        print(f"   Mean reward: {eval_results['mean_reward']:.2f} +/- {eval_results['std_reward']:.2f}")
        print(f"   Mean length: {eval_results['mean_length']:.1f}")
        print(f"   Best episode: {max(eval_results['episode_rewards']):.2f}")
        print(f"   Worst episode: {min(eval_results['episode_rewards']):.2f}")

    else:
        # Training mode
        print(f"Training for {args.timesteps:,} timesteps...")
        print(f"Optimizations applied:")
        print(f"   - Mixed action space support (continuous + discrete)")
        print(f"   - Queue-specific network architecture")
        print(f"   - Conservative V-trace parameters (rho_bar={config['rho_bar']}, c_bar={config['c_bar']})")
        print(f"   - Lower learning rate ({config['learning_rate']})")
        print(f"   - Larger buffer size ({config['buffer_size']:,})")
        print(f"   - Longer sequences ({config['sequence_length']})")
        print(f"   - Learning rate scheduling")
        print(f"   - Enhanced gradient clipping ({config['gradient_clip']})")

        # Start training
        results = baseline.train(
            total_timesteps=args.timesteps,
            eval_freq=args.eval_freq,
            save_freq=args.save_freq
        )

        print(f"\nTraining completed!")
        print(f"   Episodes: {results['episodes']}")
        print(f"   Final reward: {results['final_reward']:.2f}")
        print(f"   Training time: {results['training_time']:.2f}s")

        # Save results
        baseline.save_results("../../Results/excel/IMPALA_Optimized")

        # Final evaluation
        print(f"\nFinal evaluation...")
        eval_results = baseline.evaluate(n_episodes=10, deterministic=True, verbose=False)

        print(f"Final Performance:")
        print(f"   Mean reward: {eval_results['mean_reward']:.2f} +/- {eval_results['std_reward']:.2f}")
        print(f"   Mean length: {eval_results['mean_length']:.1f}")

        # Compare with previous IMPALA results
        print(f"\nComparison with Original IMPALA:")
        print(f"   Original IMPALA: 1705.13 +/- 25.24 (from result.md)")
        print(f"   Optimized IMPALA: {eval_results['mean_reward']:.2f} +/- {eval_results['std_reward']:.2f}")

        improvement = eval_results['mean_reward'] - 1705.13
        print(f"   Improvement: {improvement:+.2f} ({improvement/1705.13*100:+.1f}%)")

        if improvement > 0:
            print("   Optimization successful!")
        else:
            print("   Need further optimization")


if __name__ == "__main__":
    main()
