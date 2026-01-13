"""
高级DRL算法对比实验
Advanced DRL Algorithm Comparison Experiments

运行Rainbow DQN, IMPALA, R2D2, SAC v2, TD7等最新算法与现有基线的对比
用于CCF B区论文的大规模实验验证
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
    """高级算法对比实验运行器"""
    
    def __init__(self, save_dir: str = "../../Results/comparison/"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # 可用的高级算法
        self.advanced_algorithms = get_available_algorithms()
        
        # 现有基线算法（用于对比）
        self.baseline_algorithms = [
            'SB3_PPO',   # 4399
            'SB3_TD3',   # 4255  
            'SB3_A2C',   # 1721
            'SB3_SAC',   # 基线
            'SB3_DDPG'   # 基线
        ]
        
        print(f"🚀 Advanced Algorithm Comparison Runner initialized")
        print(f"   Save directory: {save_dir}")
        
    def run_advanced_algorithms_comparison(self,
                                         algorithms: List[str] = None,
                                         total_timesteps: int = 1000000,
                                         n_eval_episodes: int = 30,
                                         n_runs: int = 1) -> Dict:
        """
        运行高级算法对比实验
        
        Args:
            algorithms: 要测试的算法列表
            total_timesteps: 每个算法的训练步数
            n_eval_episodes: 评估回合数
            n_runs: 每个算法运行次数
        """
        if algorithms is None:
            # 只测试已实现的算法
            algorithms = [name for name, info in self.advanced_algorithms.items() 
                         if info['status'] == 'implemented']
        
        print(f"\n🧪 Starting Advanced DRL Algorithm Comparison")
        print(f"   Algorithms to test: {algorithms}")
        print(f"   Training timesteps: {total_timesteps:,}")
        print(f"   Evaluation episodes: {n_eval_episodes}")
        print(f"   Runs per algorithm: {n_runs}")
        print("=" * 70)
        
        results = {}
        start_time = time.time()
        
        for algorithm_name in algorithms:
            print(f"\n🎯 Testing {algorithm_name.upper()}...")
            
            algorithm_results = []
            
            for run in range(n_runs):
                print(f"   Run {run + 1}/{n_runs}")
                
                try:
                    # 创建算法基线
                    baseline = create_algorithm_baseline(algorithm_name)
                    
                    # 训练
                    train_start = time.time()
                    train_results = baseline.train(
                        total_timesteps=total_timesteps,
                        eval_freq=total_timesteps // 10,  # 10次评估
                        save_freq=total_timesteps // 4    # 4次保存
                    )
                    train_time = time.time() - train_start
                    
                    # 评估
                    eval_results = baseline.evaluate(
                        n_episodes=n_eval_episodes,
                        deterministic=True,
                        verbose=False
                    )
                    
                    # 保存结果
                    run_result = {
                        'algorithm': algorithm_name,
                        'run': run,
                        'train_results': train_results,
                        'eval_results': eval_results,
                        'training_time': train_time,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    algorithm_results.append(run_result)
                    
                    print(f"     ✅ Run {run + 1} completed - "
                          f"Mean reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
                    
                    # 保存模型
                    model_path = f"../../Models/{algorithm_name}_run_{run}.pt"
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)
                    baseline.save(model_path)
                    
                except Exception as e:
                    print(f"     ❌ Run {run + 1} failed: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            results[algorithm_name] = algorithm_results
            
            # 计算平均结果
            if algorithm_results:
                mean_rewards = [r['eval_results']['mean_reward'] for r in algorithm_results]
                mean_training_times = [r['training_time'] for r in algorithm_results]
                
                print(f"   📊 {algorithm_name} Summary:")
                print(f"      Average reward: {np.mean(mean_rewards):.2f} ± {np.std(mean_rewards):.2f}")
                print(f"      Average training time: {np.mean(mean_training_times):.1f}s")
        
        total_time = time.time() - start_time
        
        # 保存完整结果
        self._save_comparison_results(results, total_time, {
            'algorithms': algorithms,
            'total_timesteps': total_timesteps,
            'n_eval_episodes': n_eval_episodes,
            'n_runs': n_runs
        })
        
        # 生成对比报告
        self._generate_comparison_report(results)
        
        print(f"\n🎉 Advanced algorithm comparison completed!")
        print(f"   Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
        print(f"   Results saved to: {self.save_dir}")
        
        return results
    
    def run_comprehensive_comparison(self,
                                   total_timesteps: int = 1000000,
                                   n_eval_episodes: int = 50,
                                   include_baselines: bool = True) -> Dict:
        """
        运行包含基线算法的综合对比实验
        
        Args:
            total_timesteps: 训练步数
            n_eval_episodes: 评估回合数  
            include_baselines: 是否包含现有基线算法
        """
        print(f"\n🔬 Running Comprehensive Algorithm Comparison")
        
        # 高级算法结果
        advanced_results = self.run_advanced_algorithms_comparison(
            algorithms=None,  # 所有已实现的算法
            total_timesteps=total_timesteps,
            n_eval_episodes=n_eval_episodes,
            n_runs=1
        )
        
        comprehensive_results = {
            'advanced_algorithms': advanced_results,
            'baseline_algorithms': {}
        }
        
        # 如果包含基线算法，运行现有的对比实验
        if include_baselines:
            print(f"\n🔄 Running baseline algorithms for comparison...")
            
            # 创建环境
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
                print(f"⚠️  Baseline comparison failed: {str(e)}")
                comprehensive_results['baseline_algorithms'] = {}
        
        # 生成综合对比报告
        self._generate_comprehensive_report(comprehensive_results)
        
        return comprehensive_results
    
    def _save_comparison_results(self, results: Dict, total_time: float, config: Dict):
        """保存对比结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # JSON结果
        result_data = {
            'results': results,
            'total_time': total_time,
            'config': config,
            'timestamp': timestamp
        }
        
        json_path = f"{self.save_dir}/advanced_comparison_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, default=str, ensure_ascii=False)
        
        print(f"💾 Results saved to: {json_path}")
    
    def _generate_comparison_report(self, results: Dict):
        """生成对比报告"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = f"{self.save_dir}/advanced_comparison_report_{timestamp}.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 高级DRL算法对比实验报告\\n")
            f.write("# Advanced DRL Algorithm Comparison Report\\n\\n")
            
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"**测试算法数量**: {len(results)}\\n\\n")
            
            # 结果汇总表格
            f.write("## 📊 算法性能对比\\n\\n")
            f.write("| 算法名称 | 平均奖励 | 标准差 | 训练时间(s) | 状态 |\\n")
            f.write("|---------|---------|--------|------------|------|\\n")
            
            # 收集所有结果用于排序
            algorithm_summaries = []
            
            for algorithm_name, algorithm_results in results.items():
                if algorithm_results:  # 如果有成功的运行
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
                        'status': '✅'
                    })
                else:
                    algorithm_summaries.append({
                        'name': algorithm_name,
                        'avg_reward': 0,
                        'std_reward': 0,
                        'avg_time': 0,
                        'status': '❌'
                    })
            
            # 按平均奖励排序
            algorithm_summaries.sort(key=lambda x: x['avg_reward'], reverse=True)
            
            for summary in algorithm_summaries:
                f.write(f"| {summary['name']} | "
                       f"{summary['avg_reward']:.2f} | "
                       f"{summary['std_reward']:.2f} | "
                       f"{summary['avg_time']:.1f} | "
                       f"{summary['status']} |\\n")
            
            # 详细分析
            f.write("\\n## 🔍 详细分析\\n\\n")
            
            if algorithm_summaries:
                best_algorithm = algorithm_summaries[0]
                f.write(f"### 最佳算法: {best_algorithm['name']}\\n")
                f.write(f"- 平均奖励: {best_algorithm['avg_reward']:.2f} ± {best_algorithm['std_reward']:.2f}\\n")
                f.write(f"- 训练时间: {best_algorithm['avg_time']:.1f}s\\n\\n")
            
            # 算法特点分析
            f.write("### 算法特点\\n\\n")
            for name, info in self.advanced_algorithms.items():
                if name in results:
                    f.write(f"**{info['name']}**: {info['description']}\\n")
                    f.write(f"- 类型: {info['type']}\\n")
                    f.write(f"- 论文: {info['paper']}\\n\\n")
        
        print(f"📄 Report generated: {report_path}")
    
    def _generate_comprehensive_report(self, results: Dict):
        """生成综合对比报告（包含基线算法）"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S') 
        report_path = f"{self.save_dir}/comprehensive_report_{timestamp}.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 高级算法vs基线算法综合对比报告\\n")
            f.write("# Advanced vs Baseline Algorithms Comprehensive Report\\n\\n")
            
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n\\n")
            
            # 获取所有算法的性能数据
            all_algorithms = []
            
            # 高级算法
            for name, algorithm_results in results.get('advanced_algorithms', {}).items():
                if algorithm_results:
                    mean_rewards = [r['eval_results']['mean_reward'] for r in algorithm_results]
                    all_algorithms.append({
                        'name': name,
                        'type': 'Advanced',
                        'reward': np.mean(mean_rewards),
                        'std': np.std(mean_rewards)
                    })
            
            # 基线算法（从现有结果获取）
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
                    'std': 0  # 单次运行，无标准差
                })
            
            # 排序
            all_algorithms.sort(key=lambda x: x['reward'], reverse=True)
            
            # 生成综合对比表
            f.write("## 📊 算法性能综合排名\\n\\n")
            f.write("| 排名 | 算法名称 | 类型 | 平均奖励 | 标准差 |\\n")
            f.write("|-----|---------|------|---------|--------|\\n")
            
            for i, algo in enumerate(all_algorithms, 1):
                f.write(f"| {i} | {algo['name']} | {algo['type']} | "
                       f"{algo['reward']:.2f} | {algo['std']:.2f} |\\n")
            
            # 分析
            f.write("\\n## 🎯 关键发现\\n\\n")
            
            if all_algorithms:
                top3 = all_algorithms[:3]
                f.write("### 前三名算法\\n")
                for i, algo in enumerate(top3, 1):
                    f.write(f"{i}. **{algo['name']}** ({algo['type']}): {algo['reward']:.2f}\\n")
            
            f.write("\\n### 高级算法vs基线算法\\n")
            advanced_best = max([a for a in all_algorithms if a['type'] == 'Advanced'], 
                              key=lambda x: x['reward'], default={'reward': 0})
            baseline_best = max([a for a in all_algorithms if a['type'] == 'Baseline'],
                              key=lambda x: x['reward'], default={'reward': 0})
            
            f.write(f"- 最佳高级算法: {advanced_best.get('name', 'N/A')} ({advanced_best.get('reward', 0):.2f})\\n")
            f.write(f"- 最佳基线算法: {baseline_best.get('name', 'N/A')} ({baseline_best.get('reward', 0):.2f})\\n")
            
            if advanced_best.get('reward', 0) > baseline_best.get('reward', 0):
                improvement = (advanced_best['reward'] - baseline_best['reward']) / baseline_best['reward'] * 100
                f.write(f"- **高级算法性能提升**: {improvement:.1f}%\\n")
            else:
                f.write("- 基线算法在此实验中表现更佳\\n")
        
        print(f"📄 Comprehensive report generated: {report_path}")


def main():
    """主函数"""
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
    
    # 创建实验运行器
    runner = AdvancedAlgorithmComparisonRunner()

    # 处理IMPALA优化选项
    algorithms_to_run = args.algorithms

    if args.compare_impala_versions:
        # 比较两个版本的IMPALA
        algorithms_to_run = ['impala', 'impala_optimized']
        print("🔍 Comparing Original vs Optimized IMPALA")

    # 🔧 修改：不再自动切换，让用户显式指定版本
    # elif args.use_optimized_impala or (algorithms_to_run and 'impala' in algorithms_to_run):
    #     # 如果用户指定了使用优化版，或者在算法列表中指定了impala，自动替换为优化版
    #     if algorithms_to_run and 'impala' in algorithms_to_run:
    #         algorithms_to_run = [algo if algo != 'impala' else 'impala_optimized' for algo in algorithms_to_run]
    #         print("✨ Automatically using optimized IMPALA (recommended)")
    #     elif args.use_optimized_impala:
    #         algorithms_to_run = ['impala_optimized']
    #         print("✨ Using optimized IMPALA")

    elif args.use_optimized_impala:
        # 仅当显式指定 --use-optimized-impala 时才切换
        if algorithms_to_run and 'impala' in algorithms_to_run:
            algorithms_to_run = [algo if algo != 'impala' else 'impala_optimized' for algo in algorithms_to_run]
            print("✨ Using impala_optimized (explicitly requested)")
        else:
            algorithms_to_run = ['impala_optimized']
            print("✨ Using impala_optimized")

    # 调整参数（快速测试）
    if args.quick_test:
        args.timesteps = 50000
        args.eval_episodes = 10
        print("🚀 Quick test mode enabled")

    print(f"\n🧪 Advanced DRL Algorithm Comparison Experiment")
    print(f"   Algorithms: {algorithms_to_run if algorithms_to_run else 'All available'}")
    print(f"   Timesteps: {args.timesteps:,}")
    print(f"   Evaluation episodes: {args.eval_episodes}")
    print(f"   Runs per algorithm: {args.runs}")

    # 显示IMPALA优化信息
    if algorithms_to_run and 'impala_optimized' in algorithms_to_run:
        print(f"\n🎯 IMPALA Optimizations Applied:")
        print(f"   ✅ Mixed action space support (continuous + discrete)")
        print(f"   ✅ Queue-specific network architecture")
        print(f"   ✅ Conservative V-trace parameters (avoid training collapse)")
        print(f"   ✅ Lower learning rate with scheduling")
        print(f"   ✅ Enhanced stability mechanisms")

    try:
        if args.comprehensive:
            # 综合对比实验
            results = runner.run_comprehensive_comparison(
                total_timesteps=args.timesteps,
                n_eval_episodes=args.eval_episodes,
                include_baselines=True
            )
        else:
            # 仅高级算法对比
            results = runner.run_advanced_algorithms_comparison(
                algorithms=algorithms_to_run,
                total_timesteps=args.timesteps,
                n_eval_episodes=args.eval_episodes,
                n_runs=args.runs
            )
        
        print("\\n🎉 Experiment completed successfully!")
        
    except KeyboardInterrupt:
        print("\\n\\n⚠️  Experiment interrupted by user")
    except Exception as e:
        print(f"\\n\\n❌ Experiment failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()