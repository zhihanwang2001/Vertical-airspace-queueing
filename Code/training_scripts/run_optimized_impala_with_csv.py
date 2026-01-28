"""
Run Optimized IMPALA with CSV Logging (Compatible with result_excel format)
Run optimized IMPALA algorithm with CSV logging, compatible with result_excel file format

This script is specifically created for compatibility with existing result formats,
generating CSV files in the same format as other algorithms
"""

import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import time
import csv
from algorithms.advanced.impala.impala_optimized import OptimizedIMPALABaseline


class CSVTrainingLogger:
    """CSV training logger, compatible with result_excel format"""

    def __init__(self, save_path: str):
        self.save_path = save_path
        self.csv_file = None
        self.csv_writer = None
        self.start_time = time.time()

    def __enter__(self):
        self.csv_file = open(self.save_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        # Write standard format header (consistent with other algorithms)
        self.csv_writer.writerow(['Wall time', 'Step', 'Value'])
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.csv_file:
            self.csv_file.close()

    def log(self, timestep: int, reward: float):
        """Log training data point"""
        wall_time = time.time()
        self.csv_writer.writerow([wall_time, timestep, reward])
        self.csv_file.flush()  # Write immediately


def run_optimized_impala_with_csv():
    """Run optimized IMPALA and generate CSV log"""
    parser = argparse.ArgumentParser(description='Run Optimized IMPALA with CSV logging')
    parser.add_argument('--timesteps', type=int, default=500000,
                        help='Total training timesteps (default: 500000)')
    parser.add_argument('--eval-episodes', type=int, default=10,
                        help='Number of evaluation episodes (default: 10)')
    parser.add_argument('--log-freq', type=int, default=50000,
                        help='Logging frequency in timesteps (default: 50000)')
    parser.add_argument('--csv-file', type=str, default=None,
                        help='CSV output file path (auto-generated if not specified)')

    args = parser.parse_args()

    # Generate CSV filename (consistent with other algorithm formats)
    if args.csv_file is None:
        timestamp = int(time.time())
        csv_filename = f"../../Results/excel/IMPALA_Optimized_{timestamp}.csv"
    else:
        csv_filename = args.csv_file

    # Ensure directory exists
    os.makedirs(os.path.dirname(csv_filename), exist_ok=True)

    print("Running Optimized IMPALA with CSV Logging")
    print("=" * 80)
    print(f"Configuration:")
    print(f"   Total timesteps: {args.timesteps:,}")
    print(f"   Evaluation episodes: {args.eval_episodes}")
    print(f"   Log frequency: {args.log_freq:,}")
    print(f"   CSV output: {csv_filename}")

    # Optimized IMPALA configuration
    config = {
        # Conservative hyperparameter settings (avoid training collapse)
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
        'train_freq': 2,
        'verbose': 1  # Enable progress output
    }

    print(f"\nIMPALA Optimizations:")
    print(f"   Mixed action space support")
    print(f"   Queue-specific network architecture")
    print(f"   Conservative V-trace: rho_bar={config['rho_bar']}, c_bar={config['c_bar']}")
    print(f"   Lower learning rate: {config['learning_rate']}")
    print(f"   Enhanced stability mechanisms")

    # Create baseline
    baseline = OptimizedIMPALABaseline(config=config)

    # Custom training loop with CSV logging
    print(f"\nStarting training...")

    with CSVTrainingLogger(csv_filename) as logger:

        # Setup environment and agent
        baseline.setup_env()
        baseline.create_agent()

        # Training variables
        episode = 0
        timestep = 0
        episode_reward = 0.0
        episode_length = 0
        recent_rewards = []

        # Reset environment
        state, _ = baseline.env.reset()
        start_time = time.time()

        while timestep < args.timesteps:
            # Select action
            action = baseline.agent.act(state, training=True)

            # Execute action
            try:
                step_result = baseline.env.step(action)
                if len(step_result) == 5:
                    next_state, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    next_state, reward, done, info = step_result
            except Exception as e:
                print(f"Environment step error: {e}")
                break

            # Store experience
            baseline.agent.store_transition(state, action, reward, next_state, done)

            # Update statistics
            episode_reward += reward
            episode_length += 1
            timestep += 1

            # Train agent
            if timestep >= config['learning_starts']:
                train_info = baseline.agent.train()

            # Episode end handling
            if done:
                # Record episode information
                baseline.training_history['episode_rewards'].append(episode_reward)
                baseline.training_history['episode_lengths'].append(episode_length)
                recent_rewards.append(episode_reward)

                # Keep last 100 rewards
                if len(recent_rewards) > 100:
                    recent_rewards.pop(0)

                # Print progress
                if episode % 20 == 0:
                    elapsed_time = time.time() - start_time
                    avg_recent = np.mean(recent_rewards) if recent_rewards else 0

                    print(f"Episode {episode:5d} | "
                          f"Timestep {timestep:8d} | "
                          f"Reward: {episode_reward:8.2f} | "
                          f"Avg(recent): {avg_recent:8.2f} | "
                          f"Length: {episode_length:4d} | "
                          f"Time: {elapsed_time:.1f}s")

                # Reset episode
                episode += 1
                episode_reward = 0.0
                episode_length = 0
                state, _ = baseline.env.reset()
            else:
                state = next_state

            # CSV logging (by frequency)
            if timestep % args.log_freq == 0 and timestep > 0:
                # Run quick evaluation to get current performance
                eval_reward = 0.0
                if recent_rewards:
                    eval_reward = np.mean(recent_rewards)
                else:
                    # If no recent_rewards, do a quick evaluation
                    eval_state, _ = baseline.env.reset()
                    eval_episode_reward = 0.0
                    eval_done = False
                    eval_steps = 0

                    while not eval_done and eval_steps < 200:
                        eval_action = baseline.agent.act(eval_state, training=False)
                        eval_step_result = baseline.env.step(eval_action)

                        if len(eval_step_result) == 5:
                            eval_state, eval_r, eval_terminated, eval_truncated, eval_info = eval_step_result
                            eval_done = eval_terminated or eval_truncated
                        else:
                            eval_state, eval_r, eval_done, eval_info = eval_step_result

                        eval_episode_reward += eval_r
                        eval_steps += 1

                    eval_reward = eval_episode_reward

                # Log to CSV
                logger.log(timestep, eval_reward)

                print(f"Logged timestep {timestep:,}: reward = {eval_reward:.2f}")

    # Training completed
    total_time = time.time() - start_time

    print(f"\nTraining completed!")
    print(f"   Total episodes: {episode}")
    print(f"   Total time: {total_time:.2f}s ({total_time/60:.1f}min)")
    print(f"   CSV file saved: {csv_filename}")

    # Final evaluation
    print(f"\nFinal evaluation...")
    eval_results = baseline.evaluate(n_episodes=args.eval_episodes, deterministic=True, verbose=False)

    print(f"Final Performance:")
    print(f"   Mean reward: {eval_results['mean_reward']:.2f} +/- {eval_results['std_reward']:.2f}")
    print(f"   Mean length: {eval_results['mean_length']:.1f}")

    # Compare with previous IMPALA results
    original_impala_score = 1705.13  # From result_excel/prepare_fix.md
    improvement = eval_results['mean_reward'] - original_impala_score
    improvement_pct = (improvement / original_impala_score) * 100

    print(f"\nComparison with Original IMPALA:")
    print(f"   Original IMPALA: {original_impala_score:.2f} (with training collapse)")
    print(f"   Optimized IMPALA: {eval_results['mean_reward']:.2f}")
    print(f"   Improvement: {improvement:+.2f} ({improvement_pct:+.1f}%)")

    if improvement > 0:
        print(f"   Optimization successful!")
        if improvement > 500:
            print(f"   Significant improvement achieved!")
    else:
        print(f"   Performance did not improve as expected")

    # Save final model
    final_model_path = "../../Models/impala_optimized_final.pt"
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    baseline.save(final_model_path)
    print(f"Final model saved: {final_model_path}")

    return {
        'final_reward': eval_results['mean_reward'],
        'csv_file': csv_filename,
        'improvement': improvement,
        'training_time': total_time
    }


if __name__ == "__main__":
    try:
        results = run_optimized_impala_with_csv()
        print(f"\nSummary:")
        print(f"   Final performance: {results['final_reward']:.2f}")
        print(f"   Training time: {results['training_time']:.1f}s")
        print(f"   Improvement: {results['improvement']:+.2f}")
        print(f"   CSV data saved: {results['csv_file']}")

    except KeyboardInterrupt:
        print(f"\nTraining interrupted by user")
    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()
