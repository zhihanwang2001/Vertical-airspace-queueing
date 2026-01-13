"""
Comparison Runner
用于比较不同基线算法性能的实验框架
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
    """基线算法比较运行器"""
    
    def __init__(self, env, save_dir: str = "./comparison_results"):
        """
        Args:
            env: 环境实例
            save_dir: 结果保存目录
        """
        self.env = env
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # SB3算法配置
        self.algorithm_configs = {
            'SB3_TD3': {
                'learning_rate': 1e-4,
                'min_lr': 1e-6,  # 新增：最小学习率
                'use_cosine_schedule': True,  # 新增：启用余弦学习率调度
                'buffer_size': 1000000,
                'learning_starts': 1000,  # 增加初始学习步数
                'batch_size': 256,
                'tau': 0.005,
                'gamma': 0.99,
                'train_freq': 1,
                'gradient_steps': 1,
                'policy_delay': 2,
                'target_policy_noise': 0.2,
                'target_noise_clip': 0.5,
                'tensorboard_log': "./tensorboard_logs/",
                'verbose': 1,  # 恢复输出以便TensorBoard记录
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
                'learning_starts': 1000,  # 增加初始学习步数
                'batch_size': 256,
                'tau': 0.005,
                'gamma': 0.99,
                'train_freq': 1,
                'gradient_steps': 1,
                'ent_coef': 'auto',
                'target_update_interval': 1,
                'target_entropy': 'auto',
                'tensorboard_log': "./tensorboard_logs/",
                'verbose': 1,  # 恢复输出以便TensorBoard记录
                'seed': 42
            },
            'SB3_PPO': {
                'learning_rate': 3e-4,
                'min_lr': 1e-6,  # 余弦退火的最小学习率
                'n_steps': 1024,  # 减少steps，更频繁记录
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.0,
                'vf_coef': 0.5,
                'max_grad_norm': 0.5,
                'tensorboard_log': "./tensorboard_logs/",
                'verbose': 1,  # 恢复输出以便TensorBoard记录
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
            }
        }
        
        # 实验参数
        self.comparison_results = {}
        
        print(f"Comparison Runner initialized")
        print(f"Save directory: {save_dir}")
    
    def run_single_algorithm(self, 
                           algorithm_name: str,
                           total_timesteps: int = 50000,
                           n_eval_episodes: int = 10,
                           save_models: bool = True,
                           eval_freq: int = 10000) -> Dict:
        """运行单个算法"""
        print(f"\n{'='*50}")
        print(f"Running {algorithm_name}")
        print(f"{'='*50}")
        
        start_time = time.time()
        
        # 创建算法实例
        algorithm = self._create_algorithm(algorithm_name)
        
        # 训练
        if hasattr(algorithm, 'train') and algorithm_name.startswith('SB3_'):
            # SB3算法支持eval_freq参数
            train_results = algorithm.train(total_timesteps, eval_freq=eval_freq)
        else:
            # 自实现算法
            train_results = algorithm.train(total_timesteps)
        
        # 评估
        print(f"Evaluating {algorithm_name}...")
        eval_results = algorithm.evaluate(n_episodes=n_eval_episodes, verbose=True)
        
        # 保存模型
        if save_models:
            if algorithm_name.startswith('SB3_'):
                # SB3模型使用专用的模型目录结构，并使用baseline的save方法（带fallback）
                algo_lower = algorithm_name.lower().replace('sb3_', '')  # sb3_a2c -> a2c
                model_dir = os.path.join('../../../Models', algo_lower)
                os.makedirs(model_dir, exist_ok=True)
                model_path = os.path.join(model_dir, f"{algo_lower}_model_500000")
                print(f"Saving {algorithm_name} model to: {model_path}")
                # 使用baseline的save方法，它有fallback机制
                algorithm.save(model_path)
            else:
                # 非SB3模型
                model_path = os.path.join(self.save_dir, f"{algorithm_name}_model")
                algorithm.save(model_path)
        
        # 保存结果
        algorithm.save_results(os.path.join(self.save_dir, algorithm_name))
        
        total_time = time.time() - start_time
        
        # 整理结果
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
        """运行比较实验"""
        if algorithms is None:
            algorithms = ['SB3_TD3', 'SB3_A2C', 'SB3_SAC', 'SB3_PPO', 'SB3_DDPG']
        
        print(f"\\nStarting comparison experiment")
        print(f"Algorithms: {algorithms}")
        print(f"Total timesteps per run: {total_timesteps}")
        print(f"Number of runs: {n_runs}")
        print(f"Evaluation episodes: {n_eval_episodes}")
        
        all_results = {}
        
        for algorithm in algorithms:
            print(f"\\nRunning {algorithm}...")
            
            algorithm_results = []
            for run in range(n_runs):
                if n_runs > 1:
                    print(f"  Run {run + 1}/{n_runs}")
                
                # 重置环境种子（如果支持）
                if hasattr(self.env, 'seed'):
                    self.env.seed(42 + run)
                
                result = self.run_single_algorithm(
                    algorithm, total_timesteps, n_eval_episodes, 
                    save_models=(run == 0),  # 只保存第一次运行的模型
                    eval_freq=eval_freq
                )
                algorithm_results.append(result)
            
            all_results[algorithm] = algorithm_results
        
        self.comparison_results = all_results
        
        # 生成比较报告
        self._generate_comparison_report()
        
        # 绘制比较图表
        self._plot_comparison_results()
        
        return all_results
    
    def _create_algorithm(self, algorithm_name: str):
        """创建算法实例"""
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
        """生成比较报告"""
        report_path = os.path.join(self.save_dir, "comparison_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("基线算法比较报告\\n")
            f.write("="*50 + "\\n\\n")
            
            # 汇总统计
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
            
            # 按平均奖励排序
            summary_data.sort(key=lambda x: float(x['Mean Reward'].split()[0]), reverse=True)
            
            f.write("算法性能排名:\\n")
            f.write("-" * 80 + "\\n")
            f.write(f"{'Rank':<4} {'Algorithm':<12} {'Mean Reward':<15} {'Best':<10} {'Std':<10} {'Time':<15}\\n")
            f.write("-" * 80 + "\\n")
            
            for i, data in enumerate(summary_data):
                f.write(f"{i+1:<4} {data['Algorithm']:<12} {data['Mean Reward']:<15} "
                       f"{data['Best Reward']:<10} {data['Reward Std']:<10} {data['Training Time']:<15}\\n")
            
            f.write("\\n\\n详细结果:\\n")
            f.write("="*50 + "\\n")
            
            for algorithm, results_list in self.comparison_results.items():
                f.write(f"\\n{algorithm}:\\n")
                f.write("-" * 30 + "\\n")
                
                for i, result in enumerate(results_list):
                    eval_res = result['eval_results']
                    f.write(f"  Run {i+1}: {eval_res['mean_reward']:.2f} ± {eval_res['std_reward']:.2f} "
                           f"(训练时间: {result['training_time']:.2f}s)\\n")
                    
                    # 系统指标
                    if eval_res['system_metrics']:
                        avg_throughput = np.mean([m.get('throughput', 0) for m in eval_res['system_metrics']])
                        avg_stability = np.mean([m.get('stability_score', 0) for m in eval_res['system_metrics']])
                        f.write(f"    吞吐量: {avg_throughput:.2f}, 稳定性: {avg_stability:.2f}\\n")
        
        print(f"Comparison report saved to: {report_path}")
    
    def _plot_comparison_results(self):
        """绘制比较结果图表"""
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
        # 1. 训练曲线比较
        self._plot_training_curves()
        
        # 2. 性能箱型图
        self._plot_performance_boxplot()
        
        # 3. 训练时间比较
        self._plot_training_time_comparison()
        
        # 4. 详细性能雷达图
        self._plot_radar_chart()
    
    def _plot_training_curves(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Training Curves Comparison', fontsize=16)
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(self.comparison_results)))
        
        for idx, (algorithm, results_list) in enumerate(self.comparison_results.items()):
            color = colors[idx]
            
            # 取第一次运行的结果用于绘图
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
            
            # Loss Values (如果有)
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
        """绘制性能箱型图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 准备数据
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
        
        # 奖励箱型图
        sns.boxplot(data=df, x='Algorithm', y='Reward', ax=ax1)
        ax1.set_title('Evaluation Rewards Distribution')
        ax1.set_ylabel('Mean Reward')
        ax1.tick_params(axis='x', rotation=45)
        
        # 训练时间箱型图
        sns.boxplot(data=df, x='Algorithm', y='Training_Time', ax=ax2)
        ax2.set_title('Training Time Distribution')
        ax2.set_ylabel('Training Time (seconds)')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'performance_boxplot.png'), dpi=300)
        plt.close()
    
    def _plot_training_time_comparison(self):
        """绘制训练时间比较"""
        algorithms = list(self.comparison_results.keys())
        mean_times = []
        std_times = []
        
        for algorithm in algorithms:
            times = [r['training_time'] for r in self.comparison_results[algorithm]]
            mean_times.append(np.mean(times))
            std_times.append(np.std(times))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(algorithms, mean_times, yerr=std_times, capsize=5)
        
        # 添加数值标签
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
        """绘制性能雷达图"""
        # 计算各算法的标准化性能指标
        metrics = ['Reward', 'Stability', 'Efficiency', 'Throughput']
        algorithms = list(self.comparison_results.keys())
        
        # 准备数据矩阵
        data_matrix = []
        
        for algorithm in algorithms:
            results = self.comparison_results[algorithm]
            
            # 计算指标
            rewards = [r['eval_results']['mean_reward'] for r in results]
            mean_reward = np.mean(rewards)
            
            # 简化指标计算
            stability = 1.0 / (1.0 + np.std(rewards))  # 稳定性
            efficiency = 1.0 / np.mean([r['training_time'] for r in results]) * 1000  # 效率
            
            # 吞吐量（如果有系统指标）
            throughputs = []
            for result in results:
                if result['eval_results']['system_metrics']:
                    throughputs.extend([m.get('throughput', 0) for m in result['eval_results']['system_metrics']])
            throughput = np.mean(throughputs) if throughputs else abs(mean_reward) * 0.1
            
            data_matrix.append([mean_reward, stability, efficiency, throughput])
        
        # 标准化到0-1范围
        data_matrix = np.array(data_matrix)
        for i in range(data_matrix.shape[1]):
            col = data_matrix[:, i]
            data_matrix[:, i] = (col - col.min()) / (col.max() - col.min() + 1e-6)
        
        # 绘制雷达图
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(algorithms)))
        
        for i, algorithm in enumerate(algorithms):
            values = data_matrix[i].tolist()
            values += values[:1]  # 闭合图形
            
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
        """保存比较数据"""
        save_path = os.path.join(self.save_dir, filename)
        
        # 转换为可序列化格式
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
            # 尝试删除旧文件（如果存在且只读）
            if os.path.exists(save_path):
                os.chmod(save_path, 0o644)

            with open(save_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)

            print(f"Comparison data saved to: {save_path}")
        except PermissionError as e:
            print(f"⚠️  Warning: Could not save comparison data to {save_path}: {e}")
            # 尝试保存到备份位置
            backup_path = save_path.replace('.json', '_backup.json')
            try:
                with open(backup_path, 'w') as f:
                    json.dump(serializable_results, f, indent=2)
                print(f"Comparison data saved to backup location: {backup_path}")
            except Exception as e2:
                print(f"⚠️  Could not save to backup either: {e2}")
                print(f"   Training and evaluation results are still saved individually.")
    
    def load_and_analyze_results(self, filename: str = "comparison_data.json"):
        """加载并分析已保存的结果"""
        load_path = os.path.join(self.save_dir, filename)
        
        with open(load_path, 'r') as f:
            self.comparison_results = json.load(f)
        
        self._generate_comparison_report()
        self._plot_comparison_results()
        
        print(f"Results loaded and analyzed from: {load_path}")