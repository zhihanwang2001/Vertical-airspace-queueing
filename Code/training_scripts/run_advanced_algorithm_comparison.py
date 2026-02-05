"""
Advanced DRL Algorithm Comparison Experiments

Run comparison of latest algorithms like Rainbow DQN, IMPALA, R2D2, SAC v2, TD7 with existing baselines
For large-scale experimental validation in CCF B journal paper
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import time
import json
from datetime import datetime
from typing import Dict, List, Any
import numpy as np

from env.drl_optimized_env_fixed import DRLOptimizedQueueEnvFixed
from algorithms.baselines.comparison_runner import ComparisonRunner
from advanced_algorithms import (
    get_available_algorithms, 
    create_algorithm_baseline,
    print_algorithms_status
)


class AdvancedAlgorithmComparisonRunner:
    """Advanced algorithm comparison experiment runner"""

    def __init__(self, save_dir: str = "../../Results/comparison/"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Available advanced algorithms
        self.advanced_algorithms = get_available_algorithms()

        # Existing baseline algorithms for comparison
        self.baseline_algorithms = [
            'SB3_PPO',   # 4399
            'SB3_TD3',   # 4255
            'SB3_A2C',   # 1721
            'SB3_SAC',   # baseline
            'SB3_DDPG'   # baseline
        ]

        print(f"Advanced Algorithm Comparison Runner initialized")
        print(f"   Save directory: {save_dir}")
        
    def run_advanced_algorithms_comparison(self,
                                         algorithms: List[str] = None,
                                         total_timesteps: int = 1000000,
                                         n_eval_episodes: int = 30,
                                         n_runs: int = 1) -> Dict:
        """
        Run advanced algorithm comparison experiment

        Args:
            algorithms: List of algorithms to test
            total_timesteps: Training steps per algorithm
            n_eval_episodes: Number of evaluation episodes
            n_runs: Number of runs per algorithm
        """
        if algorithms is None:
            # Test only implemented algorithms
            algorithms = [name for name, info in self.advanced_algorithms.items()
                         if info['status'] == 'implemented']
        
        print(f"\nStarting Advanced DRL Algorithm Comparison")
        print(f"   Algorithms to test: {algorithms}")
        print(f"   Training timesteps: {total_timesteps:,}")
        print(f"   Evaluation episodes: {n_eval_episodes}")
        print(f"   Runs per algorithm: {n_runs}")
        print("=" * 70)
        
        results = {}
        start_time = time.time()
        
        for algorithm_name in algorithms:
            print(f"\nTesting {algorithm_name.upper()}...")
            
            algorithm_results = []
            
            for run in range(n_runs):
                print(f"   Run {run + 1}/{n_runs}")
                
                try:
                    # Create algorithm baseline
                    baseline = create_algorithm_baseline(algorithm_name)

                    # Train
                    train_start = time.time()
                    train_results = baseline.train(
                        total_timesteps=total_timesteps,
                        eval_freq=total_timesteps // 10,  # 10 evaluation checkpoints
                        save_freq=total_timesteps // 4    # 4 save checkpoints
                    )
                    train_time = time.time() - train_start

                    # Evaluate
                    eval_results = baseline.evaluate(
                        n_episodes=n_eval_episodes,
                        deterministic=True,
                        verbose=False
                    )
                    
                    # Save results
                    run_result = {
                        'algorithm': algorithm_name,
                        'run': run,
                        'train_results': train_results,
                        'eval_results': eval_results,
                        'training_time': train_time,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    algorithm_results.append(run_result)
                    
                    print(f"     Run {run + 1} completed - "
                          f"Mean reward: {eval_results['mean_reward']:.2f} +/- {eval_results['std_reward']:.2f}")
                    
                    # Save model
                    model_path = f"../../Models/{algorithm_name}_run_{run}.pt"
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    baseline.save(model_path)
                    
                except Exception as e:
                    print(f"     Run {run + 1} failed: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            results[algorithm_name] = algorithm_results
            
            # Calculate average results
            if algorithm_results:
                mean_rewards = [r['eval_results']['mean_reward'] for r in algorithm_results]
                mean_training_times = [r['training_time'] for r in algorithm_results]
                
                print(f"   {algorithm_name} Summary:")
                print(f"      Average reward: {np.mean(mean_rewards):.2f} +/- {np.std(mean_rewards):.2f}")
                print(f"      Average training time: {np.mean(mean_training_times):.1f}s")
        
        total_time = time.time() - start_time
        
        # Save complete results
        self._save_comparison_results(results, total_time, {
            'algorithms': algorithms,
            'total_timesteps': total_timesteps,
            'n_eval_episodes': n_eval_episodes,
            'n_runs': n_runs
        })
        
        # Generate comparison report
        self._generate_comparison_report(results)
        
        print(f"\nAdvanced algorithm comparison completed!")
        print(f"   Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
        print(f"   Results saved to: {self.save_dir}")
        
        return results
    
    def run_comprehensive_comparison(self,
                                   total_timesteps: int = 1000000,
                                   n_eval_episodes: int = 50,
                                   include_baselines: bool = True) -> Dict:
        """
        Run comprehensive comparison experiment including baseline algorithms

        Args:
            total_timesteps: Training steps
            n_eval_episodes: Number of evaluation episodes
            include_baselines: Whether to include existing baseline algorithms
        """
        print(f"\nRunning Comprehensive Algorithm Comparison")

        # Run advanced algorithm experiments
        advanced_results = self.run_advanced_algorithms_comparison(
            algorithms=None,  # All implemented algorithms
            total_timesteps=total_timesteps,
            n_eval_episodes=n_eval_episodes,
            n_runs=1
        )
        
        comprehensive_results = {
            'advanced_algorithms': advanced_results,
            'baseline_algorithms': {}
        }
        
        # Include baseline algorithms if requested
        if include_baselines:
            print(f"\nRunning baseline algorithms for comparison...")

            # Create environment
            env = DRLOptimizedQueueEnvFixed()
            baseline_runner = ComparisonRunner(env, save_dir=f"{self.save_dir}/baselines/")

            try:
                baseline_results = baseline_runner.run_comparison(
                    algorithms=self.baseline_algorithms,
                    total_timesteps=total_timesteps,
                    n_eval_episodes=n_eval_episodes,
                    n_runs=1
                )
                comprehensive_results['baseline_algorithms'] = baseline_results

            except Exception as e:
                print(f"Warning: Baseline comparison failed: {str(e)}")
                comprehensive_results['baseline_algorithms'] = {}
        
        # Generate comprehensive comparison report
        self._generate_comprehensive_report(comprehensive_results)
        
        return comprehensive_results
    
    def _save_comparison_results(self, results: Dict, total_time: float, config: Dict):
        """Save comparison results to JSON file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Prepare JSON results
        result_data = {
            'results': results,
            'total_time': total_time,
            'config': config,
            'timestamp': timestamp
        }

        json_path = f"{self.save_dir}/advanced_comparison_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, default=str, ensure_ascii=False)

        print(f"Results saved to: {json_path}")
    
    def _generate_comparison_report(self, results: Dict):
        """Generate comparison report in markdown format"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = f"{self.save_dir}/advanced_comparison_report_{timestamp}.md"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Advanced DRL Algorithm Comparison Report\n\n")
            
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Number of algorithms tested**: {len(results)}\n\n")
            
            # Performance comparison table
            f.write("## Algorithm Performance Comparison\n\n")
            f.write("| Algorithm | Mean Reward | Std Dev | Training Time(s) | Status |\n")
            f.write("|-----------|-------------|---------|------------------|--------|\n")

            # Collect results for sorting
            algorithm_summaries = []
            
            for algorithm_name, algorithm_results in results.items():
                if algorithm_results:  # If there are successful runs
                    mean_rewards = [r['eval_results']['mean_reward'] for r in algorithm_results]
                    training_times = [r['training_time'] for r in algorithm_results]
                    
                    avg_reward = np.mean(mean_rewards)
                    std_reward = np.std(mean_rewards)
                    avg_time = np.mean(training_times)
                    
                    algorithm_summaries.append({
                        'name': algorithm_name,
                        'avg_reward': avg_reward,
                        'std_reward': std_reward,
                        'avg_time': avg_time,
                        'status': 'Success'
                    })
                else:
                    algorithm_summaries.append({
                        'name': algorithm_name,
                        'avg_reward': 0,
                        'std_reward': 0,
                        'avg_time': 0,
                        'status': 'Failed'
                    })
            
            # Sort by average reward
            algorithm_summaries.sort(key=lambda x: x['avg_reward'], reverse=True)
            
            for summary in algorithm_summaries:
                f.write(f"| {summary['name']} | "
                       f"{summary['avg_reward']:.2f} | "
                       f"{summary['std_reward']:.2f} | "
                       f"{summary['avg_time']:.1f} | "
                       f"{summary['status']} |\n")
            
            # Detailed analysis
            f.write("\n## Detailed Analysis\n\n")
            
            if algorithm_summaries:
                best_algorithm = algorithm_summaries[0]
                f.write(f"### Best Algorithm: {best_algorithm['name']}\n")
                f.write(f"- Mean reward: {best_algorithm['avg_reward']:.2f} +/- {best_algorithm['std_reward']:.2f}\n")
                f.write(f"- Training time: {best_algorithm['avg_time']:.1f}s\n\n")
            
            # Algorithm characteristics analysis
            f.write("### Algorithm Characteristics\n\n")
            for name, info in self.advanced_algorithms.items():
                if name in results:
                    f.write(f"**{info['name']}**: {info['description']}\n")
                    f.write(f"- Type: {info['type']}\n")
                    f.write(f"- Paper: {info['paper']}\n\n")
        
        print(f"Report generated: {report_path}")
    
    def _generate_comprehensive_report(self, results: Dict):
        """Generate comprehensive comparison report including baseline algorithms"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = f"{self.save_dir}/comprehensive_report_{timestamp}.md"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Advanced vs Baseline Algorithms Comprehensive Report\n\n")
            
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Collect performance data for all algorithms
            all_algorithms = []

            # Process advanced algorithms
            for name, algorithm_results in results.get('advanced_algorithms', {}).items():
                if algorithm_results:
                    mean_rewards = [r['eval_results']['mean_reward'] for r in algorithm_results]
                    all_algorithms.append({
                        'name': name,
                        'type': 'Advanced',
                        'reward': np.mean(mean_rewards),
                        'std': np.std(mean_rewards)
                    })

            # Baseline algorithm performance from previous experiments
            baseline_performance = {
                'SB3_PPO': 4399,
                'SB3_TD3': 4255,
                'SB3_A2C': 1721
            }
            
            for name, reward in baseline_performance.items():
                all_algorithms.append({
                    'name': name,
                    'type': 'Baseline',
                    'reward': reward,
                    'std': 0  # Single run, no standard deviation
                })
            
            # Sort
            all_algorithms.sort(key=lambda x: x['reward'], reverse=True)
            
            # Generate comprehensive comparison table
            f.write("## Algorithm Performance Comprehensive Ranking\n\n")
            f.write("| Rank | Algorithm | Type | Mean Reward | Std Dev |\n")
            f.write("|------|-----------|------|-------------|----------|\n")
            
            for i, algo in enumerate(all_algorithms, 1):
                f.write(f"| {i} | {algo['name']} | {algo['type']} | "
                       f"{algo['reward']:.2f} | {algo['std']:.2f} |\n")
            
            # Analysis
            f.write("\n## Key Findings\n\n")
            
            if all_algorithms:
                top3 = all_algorithms[:3]
                f.write("### Top 3 Algorithms\n")
                for i, algo in enumerate(top3, 1):
                    f.write(f"{i}. **{algo['name']}** ({algo['type']}): {algo['reward']:.2f}\n")
            
            f.write("\n### Advanced vs Baseline Algorithms\n")
            advanced_best = max([a for a in all_algorithms if a['type'] == 'Advanced'], 
                              key=lambda x: x['reward'], default={'reward': 0})
            baseline_best = max([a for a in all_algorithms if a['type'] == 'Baseline'],
                              key=lambda x: x['reward'], default={'reward': 0})
            
            f.write(f"- Best advanced algorithm: {advanced_best.get('name', 'N/A')} ({advanced_best.get('reward', 0):.2f})\n")
            f.write(f"- Best baseline algorithm: {baseline_best.get('name', 'N/A')} ({baseline_best.get('reward', 0):.2f})\n")
            
            if advanced_best.get('reward', 0) > baseline_best.get('reward', 0):
                improvement = (advanced_best['reward'] - baseline_best['reward']) / baseline_best['reward'] * 100
                f.write(f"- **Advanced algorithm performance improvement**: {improvement:.1f}%\n")
            else:
                f.write("- Baseline algorithm performed better in this experiment\n")
        
        print(f"Comprehensive report generated: {report_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Advanced DRL Algorithm Comparison")
    parser.add_argument('--algorithms', nargs='+',
                       help='Specific algorithms to test')
    parser.add_argument('--timesteps', type=int, default=1000000,
                       help='Training timesteps per algorithm')
    parser.add_argument('--eval-episodes', type=int, default=30,
                       help='Number of evaluation episodes')
    parser.add_argument('--runs', type=int, default=1,
                       help='Number of runs per algorithm')
    parser.add_argument('--comprehensive', action='store_true',
                       help='Run comprehensive comparison including baselines')
    parser.add_argument('--quick-test', action='store_true',
                       help='Quick test with reduced parameters')
    parser.add_argument('--list-algorithms', action='store_true',
                       help='List available algorithms and exit')
    parser.add_argument('--use-optimized-impala', action='store_true',
                       help='Use optimized IMPALA instead of original (recommended)')
    parser.add_argument('--compare-impala-versions', action='store_true',
                       help='Compare original vs optimized IMPALA')
    
    args = parser.parse_args()
    
    if args.list_algorithms:
        print_algorithms_status()
        return
    
    # Create experiment runner
    runner = AdvancedAlgorithmComparisonRunner()

    # Handle IMPALA optimization options
    algorithms_to_run = args.algorithms

    if args.compare_impala_versions:
        # Compare two versions of IMPALA
        algorithms_to_run = ['impala', 'impala_optimized']
        print("Comparing Original vs Optimized IMPALA")

    elif args.use_optimized_impala:
        # Only switch when explicitly specified with --use-optimized-impala
        if algorithms_to_run and 'impala' in algorithms_to_run:
            algorithms_to_run = [algo if algo != 'impala' else 'impala_optimized' for algo in algorithms_to_run]
            print("Using impala_optimized (explicitly requested)")
        else:
            algorithms_to_run = ['impala_optimized']
            print("Using impala_optimized")

    # Adjust parameters (quick test)
    if args.quick_test:
        args.timesteps = 50000
        args.eval_episodes = 10
        print("Quick test mode enabled")

    print(f"\nAdvanced DRL Algorithm Comparison Experiment")
    print(f"   Algorithms: {algorithms_to_run if algorithms_to_run else 'All available'}")
    print(f"   Timesteps: {args.timesteps:,}")
    print(f"   Evaluation episodes: {args.eval_episodes}")
    print(f"   Runs per algorithm: {args.runs}")

    # Display IMPALA optimization information
    if algorithms_to_run and 'impala_optimized' in algorithms_to_run:
        print(f"\nIMPALA Optimizations Applied:")
        print(f"   Mixed action space support (continuous + discrete)")
        print(f"   Queue-specific network architecture")
        print(f"   Conservative V-trace parameters (avoid training collapse)")
        print(f"   Lower learning rate with scheduling")
        print(f"   Enhanced stability mechanisms")

    try:
        if args.comprehensive:
            # Comprehensive comparison experiment
            results = runner.run_comprehensive_comparison(
                total_timesteps=args.timesteps,
                n_eval_episodes=args.eval_episodes,
                include_baselines=True
            )
        else:
            # Advanced algorithms comparison only
            results = runner.run_advanced_algorithms_comparison(
                algorithms=algorithms_to_run,
                total_timesteps=args.timesteps,
                n_eval_episodes=args.eval_episodes,
                n_runs=args.runs
            )
        
        print("\nExperiment completed successfully!")
        
    except KeyboardInterrupt:
        print("\n\nExperiment interrupted by user")
    except Exception as e:
        print(f"\n\nExperiment failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
