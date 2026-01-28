"""
Comparison Runner
Experimental framework for comparing performance of different baseline algorithms
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
import os
import json
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

from .sb3_td3_baseline import SB3TD3Baseline
from .sb3_ddpg_baseline import SB3DDPGBaseline
from .sb3_sac_baseline import SB3SACBaseline
from .sb3_ppo_baseline import SB3PPOBaseline
from .sb3_a2c_baseline import SB3A2CBaseline
from .random_baseline import RandomBaseline
from .heuristic_baseline import HeuristicBaseline
from .traditional_baselines_fixed import FCFSBaseline, SJFBaseline, PriorityBaseline


class ComparisonRunner:
    """Baseline algorithm comparison runner"""
    
    def __init__(self, env, save_dir: str = "./comparison_results"):
        """
        Args:
            env: Environment instance
            save_dir: Results save directory
        """
        self.env = env
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # SB3 algorithm configurations
        self.algorithm_configs = {
            'SB3_TD3': {
                'learning_rate': 1e-4,
                'min_lr': 1e-6,  # New: minimum learning rate
                'use_cosine_schedule': True,  # New: enable cosine learning rate schedule
                'buffer_size': 1000000,
                'learning_starts': 1000,  # Increase initial learning steps
                'batch_size': 256,
                'tau': 0.005,
                'gamma': 0.99,
                'train_freq': 1,
                'gradient_steps': 1,
                'policy_delay': 2,
                'target_policy_noise': 0.2,
                'target_noise_clip': 0.5,
                'tensorboard_log': "./tensorboard_logs/",
                'verbose': 1,  # Restore output for TensorBoard logging
                'seed': 42
            },
            'SB3_DDPG': {
                'learning_rate': 1e-4,
                'buffer_size': 1000000,
                'learning_starts': 100,
                'batch_size': 256,
                'tau': 0.005,
                'gamma': 0.99,
                'train_freq': 1,
                'gradient_steps': 1,
                'action_noise_type': 'normal',
                'action_noise_sigma': 0.1,
                'tensorboard_log': "./tensorboard_logs/",
                'verbose': 1,
                'seed': 42
            },
            'SB3_SAC': {
                'learning_rate': 3e-4,
                'buffer_size': 1000000,
                'learning_starts': 1000,  # Increase initial learning steps
                'batch_size': 256,
                'tau': 0.005,
                'gamma': 0.99,
                'train_freq': 1,
                'gradient_steps': 1,
                'ent_coef': 'auto',
                'target_update_interval': 1,
                'target_entropy': 'auto',
                'tensorboard_log': "./tensorboard_logs/",
                'verbose': 1,  # Restore output for TensorBoard logging
                'seed': 42
            },
            'SB3_PPO': {
                'learning_rate': 3e-4,
                'min_lr': 1e-6,  # Minimum learning rate for cosine annealing
                'n_steps': 1024,  # Reduce steps for more frequent logging
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.0,
                'vf_coef': 0.5,
                'max_grad_norm': 0.5,
                'tensorboard_log': "./tensorboard_logs/",
                'verbose': 1,  # Restore output for TensorBoard logging
                'seed': 42
            },
            'SB3_A2C': {
                'learning_rate': 7e-4,
                'n_steps': 5,
                'gamma': 0.99,
                'gae_lambda': 1.0,
                'ent_coef': 0.0,
                'vf_coef': 0.5,
                'max_grad_norm': 0.5,
                'tensorboard_log': "./tensorboard_logs/",
                'verbose': 1,
                'seed': 42
            }
        }
        
        # Experiment parameters
        self.comparison_results = {}
        
        print(f"Comparison Runner initialized")
        print(f"Save directory: {save_dir}")
    
    def run_single_algorithm(self, 
                           algorithm_name: str,
                           total_timesteps: int = 50000,
                           n_eval_episodes: int = 10,
                           save_models: bool = True,
                           eval_freq: int = 10000) -> Dict:
        """Run single algorithm"""
        print(f"\n{'='*50}")
        print(f"Running {algorithm_name}")
        print(f"{'='*50}")
        
        start_time = time.time()
        
        # Create algorithm instance
        algorithm = self._create_algorithm(algorithm_name)
        
        # Train
        if hasattr(algorithm, 'train') and algorithm_name.startswith('SB3_'):
            # SB3 algorithms support eval_freq parameter
            train_results = algorithm.train(total_timesteps, eval_freq=eval_freq)
        else:
            # Self-implemented algorithms
            train_results = algorithm.train(total_timesteps)
        
        # Evaluate
        print(f"Evaluating {algorithm_name}...")
        eval_results = algorithm.evaluate(n_episodes=n_eval_episodes, verbose=True)
        
        # Save model
        if save_models:
            if algorithm_name.startswith('SB3_'):
                # SB3 models use dedicated model directory structure and baseline's save method (with fallback)
                algo_lower = algorithm_name.lower().replace('sb3_', '')  # sb3_a2c -> a2c
                model_dir = os.path.join('../../../Models', algo_lower)
                os.makedirs(model_dir, exist_ok=True)
                model_path = os.path.join(model_dir, f"{algo_lower}_model_500000")
                print(f"Saving {algorithm_name} model to: {model_path}")
                # Use baseline's save method, which has fallback mechanism
                algorithm.save(model_path)
            else:
                # Non-SB3 models
                model_path = os.path.join(self.save_dir, f"{algorithm_name}_model")
                algorithm.save(model_path)
        
        # Save results
        algorithm.save_results(os.path.join(self.save_dir, algorithm_name))
        
        total_time = time.time() - start_time
        
        # Organize results
        results = {
            'algorithm': algorithm_name,
            'train_results': train_results,
            'eval_results': eval_results,
            'training_time': total_time,
            'training_history': algorithm.training_history,
            'config': algorithm.config
        }
        
        print(f"{algorithm_name} completed in {total_time:.2f} seconds")
        print(f"Final evaluation: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
        
        return results
    
    def run_comparison(self, 
                      algorithms: List[str] = None,
                      total_timesteps: int = 50000,
                      n_eval_episodes: int = 10,
                      n_runs: int = 1,
                      eval_freq: int = 10000) -> Dict:
        """Run comparison experiment"""
        if algorithms is None:
            algorithms = ['SB3_TD3', 'SB3_A2C', 'SB3_SAC', 'SB3_PPO', 'SB3_DDPG']
        
        print(f"\nStarting comparison experiment")
        print(f"Algorithms: {algorithms}")
        print(f"Total timesteps per run: {total_timesteps}")
        print(f"Number of runs: {n_runs}")
        print(f"Evaluation episodes: {n_eval_episodes}")
        
        all_results = {}
        
        for algorithm in algorithms:
            print(f"\nRunning {algorithm}...")
            
            algorithm_results = []
            for run in range(n_runs):
                if n_runs > 1:
                    print(f"  Run {run + 1}/{n_runs}")
                
                # Reset environment seed (if supported)
                if hasattr(self.env, 'seed'):
                    self.env.seed(42 + run)
                
                result = self.run_single_algorithm(
                    algorithm, total_timesteps, n_eval_episodes, 
                    save_models=(run == 0),  # Only save model from first run
                    eval_freq=eval_freq
                )
                algorithm_results.append(result)
            
            all_results[algorithm] = algorithm_results
        
        self.comparison_results = all_results
        
        # Generate comparison report
        self._generate_comparison_report()
        
        # Plot comparison results
        self._plot_comparison_results()
        
        return all_results
    
    def _create_algorithm(self, algorithm_name: str):
        """Create algorithm instance"""
        config = self.algorithm_configs.get(algorithm_name, {})
        
        if algorithm_name == 'SB3_TD3':
            return SB3TD3Baseline(config=config)
        elif algorithm_name == 'SB3_SAC':
            return SB3SACBaseline(config=config)
        elif algorithm_name == 'SB3_PPO':
            return SB3PPOBaseline(config=config)
        elif algorithm_name == 'SB3_A2C':
            return SB3A2CBaseline(config=config)
        elif algorithm_name == 'SB3_DDPG':
            return SB3DDPGBaseline(config=config)
        elif algorithm_name == 'Random':
            return RandomBaseline(self.env, config=config)
        elif algorithm_name == 'Heuristic':
            return HeuristicBaseline(self.env, config=config)
        elif algorithm_name == 'FCFS':
            return FCFSBaseline(self.env, config=config)
        elif algorithm_name == 'SJF':
            return SJFBaseline(self.env, config=config)
        elif algorithm_name == 'Priority':
            return PriorityBaseline(self.env, config=config)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
    
    def _generate_comparison_report(self):
        """Generate comparison report"""
        report_path = os.path.join(self.save_dir, "comparison_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Baseline Algorithm Comparison Report\n")
            f.write("="*50 + "\n\n")
            
            # Summary statistics
            summary_data = []
            for algorithm, results_list in self.comparison_results.items():
                rewards = [r['eval_results']['mean_reward'] for r in results_list]
                stds = [r['eval_results']['std_reward'] for r in results_list]
                times = [r['training_time'] for r in results_list]
                
                summary_data.append({
                    'Algorithm': algorithm,
                    'Mean Reward': f"{np.mean(rewards):.2f} ± {np.std(rewards):.2f}",
                    'Best Reward': f"{np.max(rewards):.2f}",
                    'Reward Std': f"{np.mean(stds):.2f}",
                    'Training Time': f"{np.mean(times):.2f}s ± {np.std(times):.2f}s"
                })
            
            # Sort by average reward
            summary_data.sort(key=lambda x: float(x['Mean Reward'].split()[0]), reverse=True)
            
            f.write("Algorithm Performance Ranking:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Rank':<4} {'Algorithm':<12} {'Mean Reward':<15} {'Best':<10} {'Std':<10} {'Time':<15}\n")
            f.write("-" * 80 + "\n")
            
            for i, data in enumerate(summary_data):
                f.write(f"{i+1:<4} {data['Algorithm']:<12} {data['Mean Reward']:<15} "
                       f"{data['Best Reward']:<10} {data['Reward Std']:<10} {data['Training Time']:<15}\n")
            
            f.write("\n\nDetailed Results:\n")
            f.write("="*50 + "\n")
            
            for algorithm, results_list in self.comparison_results.items():
                f.write(f"\n{algorithm}:\n")
                f.write("-" * 30 + "\n")
                
                for i, result in enumerate(results_list):
                    eval_res = result['eval_results']
                    f.write(f"  Run {i+1}: {eval_res['mean_reward']:.2f} ± {eval_res['std_reward']:.2f} "
                           f"(Training time: {result['training_time']:.2f}s)\n")
                    
                    # System metrics
                    if eval_res['system_metrics']:
                        avg_throughput = np.mean([m.get('throughput', 0) for m in eval_res['system_metrics']])
                        avg_stability = np.mean([m.get('stability_score', 0) for m in eval_res['system_metrics']])
                        f.write(f"    Throughput: {avg_throughput:.2f}, Stability: {avg_stability:.2f}\n")
        
        print(f"Comparison report saved to: {report_path}")
    
    def _plot_comparison_results(self):
        """Plot comparison results"""
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
        # 1. Training curves comparison
        self._plot_training_curves()
        
        # 2. Performance boxplot
        self._plot_performance_boxplot()
        
        # 3. Training time comparison
        self._plot_training_time_comparison()
        
        # 4. Detailed performance radar chart
        self._plot_radar_chart()
    
    def _plot_training_curves(self):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Training Curves Comparison', fontsize=16)
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.comparison_results)))
        
        for idx, (algorithm, results_list) in enumerate(self.comparison_results.items()):
            color = colors[idx]
            
            # Use first run results for plotting
            result = results_list[0]
            history = result['training_history']
            
            # Episode Rewards
            if history['episode_rewards']:
                axes[0, 0].plot(history['episode_rewards'], label=algorithm, color=color, alpha=0.7)
            
            # Average Rewards
            if history['avg_rewards']:
                axes[0, 1].plot(history['avg_rewards'], label=algorithm, color=color, alpha=0.7)
            
            # Episode Lengths
            if history['episode_lengths']:
                axes[1, 0].plot(history['episode_lengths'], label=algorithm, color=color, alpha=0.7)
            
            # Loss Values (if available)
            if history.get('loss_values'):
                axes[1, 1].plot(history['loss_values'], label=algorithm, color=color, alpha=0.7)
        
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].set_title('Average Rewards (100 episodes)')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Average Reward')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        axes[1, 0].set_title('Episode Lengths')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Length')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        axes[1, 1].set_title('Loss Values')
        axes[1, 1].set_xlabel('Training Step')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_curves.png'), dpi=300)
        plt.close()
    
    def _plot_performance_boxplot(self):
        """Plot performance boxplot"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Prepare data
        algorithms = []
        rewards = []
        training_times = []
        
        for algorithm, results_list in self.comparison_results.items():
            for result in results_list:
                algorithms.append(algorithm)
                rewards.append(result['eval_results']['mean_reward'])
                training_times.append(result['training_time'])
        
        df = pd.DataFrame({
            'Algorithm': algorithms,
            'Reward': rewards,
            'Training_Time': training_times
        })
        
        # Reward boxplot
        sns.boxplot(data=df, x='Algorithm', y='Reward', ax=ax1)
        ax1.set_title('Evaluation Rewards Distribution')
        ax1.set_ylabel('Mean Reward')
        ax1.tick_params(axis='x', rotation=45)
        
        # Training time boxplot
        sns.boxplot(data=df, x='Algorithm', y='Training_Time', ax=ax2)
        ax2.set_title('Training Time Distribution')
        ax2.set_ylabel('Training Time (seconds)')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'performance_boxplot.png'), dpi=300)
        plt.close()
    
    def _plot_training_time_comparison(self):
        """Plot training time comparison"""
        algorithms = list(self.comparison_results.keys())
        mean_times = []
        std_times = []
        
        for algorithm in algorithms:
            times = [r['training_time'] for r in self.comparison_results[algorithm]]
            mean_times.append(np.mean(times))
            std_times.append(np.std(times))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(algorithms, mean_times, yerr=std_times, capsize=5)
        
        # Add value labels
        for i, (mean_time, std_time) in enumerate(zip(mean_times, std_times)):
            ax.text(i, mean_time + std_time + max(mean_times) * 0.01, 
                   f'{mean_time:.1f}s', ha='center', va='bottom')
        
        ax.set_title('Training Time Comparison')
        ax.set_ylabel('Training Time (seconds)')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_time_comparison.png'), dpi=300)
        plt.close()
    
    def _plot_radar_chart(self):
        """Plot performance radar chart"""
        # Calculate standardized performance metrics for each algorithm
        metrics = ['Reward', 'Stability', 'Efficiency', 'Throughput']
        algorithms = list(self.comparison_results.keys())
        
        # Prepare data matrix
        data_matrix = []
        
        for algorithm in algorithms:
            results = self.comparison_results[algorithm]
            
            # Calculate metrics
            rewards = [r['eval_results']['mean_reward'] for r in results]
            mean_reward = np.mean(rewards)
            
            # Simplified metric calculation
            stability = 1.0 / (1.0 + np.std(rewards))  # Stability
            efficiency = 1.0 / np.mean([r['training_time'] for r in results]) * 1000  # Efficiency
            
            # Throughput (if system metrics available)
            throughputs = []
            for result in results:
                if result['eval_results']['system_metrics']:
                    throughputs.extend([m.get('throughput', 0) for m in result['eval_results']['system_metrics']])
            throughput = np.mean(throughputs) if throughputs else abs(mean_reward) * 0.1
            
            data_matrix.append([mean_reward, stability, efficiency, throughput])
        
        # Normalize to 0-1 range
        data_matrix = np.array(data_matrix)
        for i in range(data_matrix.shape[1]):
            col = data_matrix[:, i]
            data_matrix[:, i] = (col - col.min()) / (col.max() - col.min() + 1e-6)
        
        # Plot radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Close the plot
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(algorithms)))
        
        for i, algorithm in enumerate(algorithms):
            values = data_matrix[i].tolist()
            values += values[:1]  # Close the plot
            
            ax.plot(angles, values, 'o-', linewidth=2, label=algorithm, color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title('Algorithm Performance Radar Chart', size=16, y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'performance_radar.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_comparison_data(self, filename: str = "comparison_data.json"):
        """Save comparison data"""
        save_path = os.path.join(self.save_dir, filename)
        
        # Convert to serializable format
        serializable_results = {}
        for algorithm, results_list in self.comparison_results.items():
            serializable_results[algorithm] = []
            for result in results_list:
                serializable_result = {
                    'algorithm': result['algorithm'],
                    'eval_mean_reward': result['eval_results']['mean_reward'],
                    'eval_std_reward': result['eval_results']['std_reward'],
                    'training_time': result['training_time'],
                    'total_episodes': result['train_results'].get('episodes', 0),
                    'config': result['config']
                }
                serializable_results[algorithm].append(serializable_result)

        try:
            # Try to delete old file (if exists and read-only)
            if os.path.exists(save_path):
                os.chmod(save_path, 0o644)

            with open(save_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)

            print(f"Comparison data saved to: {save_path}")
        except PermissionError as e:
            print(f"⚠️  Warning: Could not save comparison data to {save_path}: {e}")
            # Try to save to backup location
            backup_path = save_path.replace('.json', '_backup.json')
            try:
                with open(backup_path, 'w') as f:
                    json.dump(serializable_results, f, indent=2)
                print(f"Comparison data saved to backup location: {backup_path}")
            except Exception as e2:
                print(f"⚠️  Could not save to backup either: {e2}")
                print(f"   Training and evaluation results are still saved individually.")
    
    def load_and_analyze_results(self, filename: str = "comparison_data.json"):
        """Load and analyze saved results"""
        load_path = os.path.join(self.save_dir, filename)
        
        with open(load_path, 'r') as f:
            self.comparison_results = json.load(f)
        
        self._generate_comparison_report()
        self._plot_comparison_results()
        
        print(f"Results loaded and analyzed from: {load_path}")
