"""
垂直分层队列系统的帕累托最优解集分析实现（最终修正版）
Pareto Optimal Set Analysis for Vertical Stratified Queuing System (Final Fixed Version)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import seaborn as sns
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from typing import List, Tuple, Dict, Optional
import pandas as pd
from itertools import combinations
import warnings
import time
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from env.drl_optimized_env_fixed import DRLOptimizedQueueEnvFixed


class ParetoAnalyzer:
    """帕累托最优解集分析器（最终修正版）"""
    
    def __init__(self, env):
        """
        Args:
            env: 垂直分层队列环境实例
        """
        self.env = env
        self.objective_names = [
            'Throughput', 'Balance', 'Efficiency',
            'Transfer', 'Stability', 'Anti-Penalty'
        ]
        self.n_objectives = len(self.objective_names)
        
        # 存储评估结果
        self.solutions = []
        self.objective_values = []
        self.pareto_indices = []
        self.pareto_front = []
        
        print(f"ParetoAnalyzer initialized with {self.n_objectives} objectives")
    
    def evaluate_solution(self, policy_params: Dict, n_episodes: int = 5) -> np.ndarray:
        """
        评估单个解的多目标性能
        
        Args:
            policy_params: 策略参数字典
            n_episodes: 评估轮数
            
        Returns:
            6维目标向量
        """
        objective_values = np.zeros(self.n_objectives)
        
        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            episode_objectives = np.zeros(self.n_objectives)
            steps = 0
            
            while steps < 200:  # 最大步数限制
                # 根据策略参数生成动作
                action = self._policy_to_action(obs, policy_params)
                
                # 执行动作
                next_obs, reward, terminated, truncated, info = self.env.step(action)

                # 提取多目标奖励分量（使用step后的观测和info）
                objectives = self._extract_objectives(next_obs, action, reward, info)
                episode_objectives += objectives
                
                obs = next_obs
                steps += 1
                
                if terminated or truncated:
                    break
            
            # 严格评估：确保策略的持续性和稳定性
            if steps >= 50:  # 提高最小步数要求到50步
                # 正常的按步数平均化
                objective_values += episode_objectives / max(steps, 1)
            elif steps >= 20:
                # 中等长度episode给予部分分数（惩罚不够稳定的策略）
                penalty_factor = steps / 50.0  # 线性惩罚
                objective_values += (episode_objectives / max(steps, 1)) * penalty_factor
            else:
                # 短期episode（<20步）视为无效策略
                objective_values += np.zeros(self.n_objectives)
        
        return objective_values / n_episodes
    
    def _policy_to_action(self, obs, policy_params: Dict) -> Dict:
        """将策略参数转换为环境动作"""
        # 提取观察信息
        if isinstance(obs, dict):
            utilization = obs.get('utilization_rates', np.ones(5) * 0.5)
            queue_lengths = obs.get('queue_lengths', np.ones(5))
        elif isinstance(obs, np.ndarray):
            queue_lengths = obs[:5] if len(obs) >= 5 else np.ones(5)
            utilization = obs[5:10] if len(obs) >= 10 else np.ones(5) * 0.5
        else:
            queue_lengths = np.ones(5)
            utilization = np.ones(5) * 0.5
        
        # 基于策略参数生成动作
        action = {
            'service_intensities': np.array([
                policy_params.get('base_service', 1.0) + 
                policy_params.get('adaptation', 0.1) * (util - 0.5)
                for util in utilization
            ], dtype=np.float32).clip(0.1, 2.0),
            
            'arrival_multiplier': np.array([policy_params.get('arrival_factor', 1.0)], dtype=np.float32).clip(0.5, 5.0),
            
            'emergency_transfers': (utilization > policy_params.get('transfer_threshold', 0.8)).astype(np.int8)
        }
        
        return action
    
    def _extract_objectives(self, obs, action: Dict, reward: float, info: Dict) -> np.ndarray:
        """
        从环境step返回中提取多目标向量（修复版-使用环境奖励组件）

        核心修复：
        1. 优先使用环境提供的奖励组件分解（reward_components）
        2. 避免重复计算和时序不一致问题
        3. 确保与环境奖励函数完全一致

        Returns:
            6维目标向量，所有目标都是越大越好
        """

        # 方法1：直接使用环境提供的奖励组件（最准确）
        if 'reward_components' in info:
            components = info['reward_components']

            throughput_obj = components['throughput']
            balance_obj = components['balance']
            efficiency_obj = components['efficiency']
            transfer_obj = components['transfer']
            stability_obj = components['stability']

            # 将负惩罚转换为正向目标
            penalty_obj = -(components['congestion'] + components['instability'])

            return np.array([throughput_obj, balance_obj, efficiency_obj, transfer_obj, stability_obj, penalty_obj])

        # 方法2：备用计算（如果环境未提供reward_components）
        print("警告：环境未提供reward_components，使用备用计算方法")

        # 从info中提取统计数据
        service_counts = np.array(info.get('service_counts', np.zeros(5)))
        transfer_counts = np.array(info.get('transfer_counts', np.zeros(5)))

        # 从obs中提取队列状态
        if isinstance(obs, dict):
            queue_lengths = obs.get('queue_lengths', np.zeros(5))
        elif isinstance(obs, np.ndarray) and len(obs) >= 5:
            queue_lengths = obs[:5]
        else:
            queue_lengths = np.zeros(5)

        # 环境固定参数
        capacities = np.array([8, 6, 4, 3, 2])

        # 1. 吞吐量目标
        throughput_obj = 10.0 * np.sum(service_counts)

        # 2. 负载均衡目标（基尼系数）
        utilization_rates = queue_lengths / (capacities + 1e-8)
        if np.sum(utilization_rates) > 1e-6:
            sorted_util = np.sort(utilization_rates)
            n = len(sorted_util)
            gini = (2 * np.sum((np.arange(n) + 1) * sorted_util)) / (n * np.sum(sorted_util)) - (n + 1) / n
            balance_obj = 5.0 * (1.0 - gini)
        else:
            balance_obj = 5.0

        # 3. 效率目标
        service_total = np.sum(service_counts)
        base_energy = 1.0 + np.sum(action['service_intensities']) + action['arrival_multiplier'][0] * 0.5 + np.sum(action['emergency_transfers']) * 0.2
        if base_energy > 1e-6:
            efficiency_obj = 3.0 * service_total / base_energy
        else:
            efficiency_obj = 0.0

        # 4. 转移效率目标
        transfer_obj = 0.0
        for i in range(4):
            if transfer_counts[i] > 0:
                upper_pressure = queue_lengths[i] / (capacities[i] + 1e-8)
                lower_util = queue_lengths[i+1] / (capacities[i+1] + 1e-8)
                if upper_pressure > lower_util:
                    transfer_obj += 2.0 * transfer_counts[i]

        # 5. 稳定性目标
        stability_obj = info.get('stability_score', 0.0)

        # 6. 反惩罚目标
        congestion_levels = np.maximum(0, (queue_lengths - 0.8 * capacities) / capacities)
        congestion_penalty = -20.0 * np.sum(congestion_levels)

        # 使用info中的current_service_rates和current_arrival_rate计算不稳定惩罚
        if 'current_service_rates' in info and 'current_arrival_rate' in info:
            current_service_rates = np.array(info['current_service_rates'])
            current_arrival_rate = info['current_arrival_rate']
            arrival_weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
            current_arrivals = current_arrival_rate * arrival_weights
            load_rates = current_arrivals / np.maximum(current_service_rates, 1e-6)
            instability_levels = np.maximum(0, load_rates - 0.95)
            instability_penalty = -15.0 * np.sum(instability_levels)
        else:
            instability_penalty = 0.0

        penalty_obj = -(congestion_penalty + instability_penalty)

        return np.array([throughput_obj, balance_obj, efficiency_obj, transfer_obj, stability_obj, penalty_obj])
    
    def generate_random_solutions(self, n_solutions: int = 10000) -> None:
        """生成随机解集进行帕累托分析"""
        print(f"Generating {n_solutions} random solutions...")
        start_time = time.time()
        
        self.solutions = []
        self.objective_values = []
        
        for i in range(n_solutions):
            if i % 500 == 0:
                elapsed = time.time() - start_time
                eta = elapsed * (n_solutions - i) / (i + 1) if i > 0 else 0
                print(f"  Progress: {i}/{n_solutions} ({i/n_solutions*100:.1f}%) - ETA: {eta/60:.1f}min")
            
            # 修复：生成物理合理的策略参数组合
            # 避免极端的到达率+服务率组合导致系统过载

            arrival_factor = np.random.uniform(0.5, 3.0)  # 限制到达倍数最大3x

            # 根据到达倍数调整服务能力范围，确保系统可运行
            if arrival_factor > 2.0:
                # 高到达率时，需要较高的基础服务能力
                base_service_range = (0.8, 1.5)
                adaptation_range = (0.1, 0.4)
            elif arrival_factor > 1.5:
                # 中等到达率时，中等服务能力
                base_service_range = (0.5, 1.3)
                adaptation_range = (0.0, 0.6)
            else:
                # 低到达率时，可以使用更广泛的服务能力
                base_service_range = (0.3, 1.2)
                adaptation_range = (0.0, 0.8)

            policy_params = {
                'base_service': np.random.uniform(*base_service_range),
                'adaptation': np.random.uniform(*adaptation_range),
                'arrival_factor': arrival_factor,
                'transfer_threshold': np.random.uniform(0.4, 0.9)  # 更合理的转移阈值
            }
            
            # 评估解
            objectives = self.evaluate_solution(policy_params, n_episodes=5)
            
            self.solutions.append(policy_params)
            self.objective_values.append(objectives)
        
        self.objective_values = np.array(self.objective_values)
        elapsed = time.time() - start_time
        print(f"Generated {len(self.solutions)} solutions in {elapsed/60:.1f} minutes")
    
    def find_pareto_front_efficient(self) -> None:
        """高效的帕累托前沿识别算法（Non-dominated Sorting）+ 可行性过滤"""
        print(f"Finding Pareto front among {len(self.objective_values)} solutions...")
        start_time = time.time()

        n_solutions = len(self.objective_values)

        # 🔧 修复1: 先过滤掉不可行解（Stability=0的系统崩溃解）
        STABILITY_THRESHOLD = 0.5  # 稳定性最低阈值
        feasible_mask = self.objective_values[:, 4] > STABILITY_THRESHOLD  # Stability是第5个目标

        feasible_indices = np.where(feasible_mask)[0]
        print(f"  Filtering feasible solutions: {len(feasible_indices)}/{n_solutions} are stable")

        if len(feasible_indices) == 0:
            print("  ⚠️  No feasible solutions found! Using all solutions...")
            feasible_indices = np.arange(n_solutions)

        feasible_objectives = self.objective_values[feasible_indices]
        n_feasible = len(feasible_indices)

        domination_count = np.zeros(n_feasible)  # 被支配次数
        dominated_solutions = [[] for _ in range(n_feasible)]  # 支配的解列表

        # 计算支配关系（仅在可行解中）
        for i in range(n_feasible):
            if i % 1000 == 0:
                print(f"  Processing solution {i}/{n_feasible}")

            for j in range(i + 1, n_feasible):
                # 检查i是否支配j
                i_dominates_j = (np.all(feasible_objectives[i] >= feasible_objectives[j]) and
                               np.any(feasible_objectives[i] > feasible_objectives[j]))

                # 检查j是否支配i
                j_dominates_i = (np.all(feasible_objectives[j] >= feasible_objectives[i]) and
                               np.any(feasible_objectives[j] > feasible_objectives[i]))

                if i_dominates_j:
                    dominated_solutions[i].append(j)
                    domination_count[j] += 1
                elif j_dominates_i:
                    dominated_solutions[j].append(i)
                    domination_count[i] += 1

        # 找到非支配解（被支配次数为0）
        pareto_mask = domination_count == 0
        local_pareto_indices = np.where(pareto_mask)[0]

        # 映射回原始索引
        self.pareto_indices = feasible_indices[local_pareto_indices]
        self.pareto_front = self.objective_values[self.pareto_indices]

        elapsed = time.time() - start_time
        print(f"Found {len(self.pareto_indices)} Pareto optimal solutions in {elapsed:.1f} seconds")
        print(f"  Pareto ratio: {len(self.pareto_indices)/n_solutions*100:.2f}%")
    
    def find_pareto_front(self) -> None:
        """帕累托前沿识别（调用高效版本）"""
        self.find_pareto_front_efficient()
    
    def find_knee_points_improved(self) -> List[int]:
        """
        改进的膝点检测算法（基于稀疏性和trade-off分析）

        膝点定义：帕累托前沿上最具代表性的解，满足：
        1. 到理想点距离较近（高质量）
        2. 在前沿上分布稀疏（代表性强）
        3. 目标之间trade-off合理
        """
        if len(self.pareto_front) < 3:
            return list(range(len(self.pareto_front)))

        print("Finding knee points using improved method...")

        # 固定膝点数量（避免阈值方法的不稳定性）
        n_pareto = len(self.pareto_front)
        target_knees = max(5, min(15, n_pareto // 20))  # 5-15个，约占5%

        print(f"  Target knee points: {target_knees} (from {n_pareto} Pareto solutions)")

        # 归一化帕累托前沿
        ideal_point = np.max(self.pareto_front, axis=0)
        nadir_point = np.min(self.pareto_front, axis=0)
        range_vector = np.maximum(ideal_point - nadir_point, 1e-8)
        normalized_front = (self.pareto_front - nadir_point) / range_vector

        # 方法1：计算到理想点的距离（越小越好）
        ideal_distances = np.linalg.norm(normalized_front - 1.0, axis=1)

        # 方法2：计算稀疏性得分（使用k近邻距离）
        # 距离最近的k个邻居的平均距离（越大说明越稀疏/代表性越强）
        k = min(10, n_pareto // 10)
        distances_matrix = cdist(normalized_front, normalized_front, metric='euclidean')
        np.fill_diagonal(distances_matrix, np.inf)  # 排除自己

        sparsity_scores = np.zeros(n_pareto)
        for i in range(n_pareto):
            # 找最近的k个邻居的平均距离
            nearest_k_distances = np.partition(distances_matrix[i], k)[:k]
            sparsity_scores[i] = np.mean(nearest_k_distances)

        # 方法3：目标均衡性（避免极端解）
        # 使用变异系数（CV）：std/mean，越小说明越均衡
        uniformity_scores = np.zeros(n_pareto)
        for i in range(n_pareto):
            point = normalized_front[i]
            mean_val = np.mean(point)
            if mean_val > 1e-6:
                cv = np.std(point) / mean_val
                uniformity_scores[i] = 1.0 / (1.0 + cv)  # 转换为得分（越大越均衡）
            else:
                uniformity_scores[i] = 0.0

        # 综合得分（多准则决策）
        # 归一化各项得分到[0,1]
        quality_score = 1.0 - (ideal_distances - ideal_distances.min()) / (ideal_distances.max() - ideal_distances.min() + 1e-8)
        diversity_score = (sparsity_scores - sparsity_scores.min()) / (sparsity_scores.max() - sparsity_scores.min() + 1e-8)
        balance_score = uniformity_scores

        # 加权综合（质量40%，多样性40%，均衡性20%）
        total_scores = quality_score * 0.4 + diversity_score * 0.4 + balance_score * 0.2

        # 直接选择得分最高的target_knees个点
        top_k_indices = np.argsort(total_scores)[-target_knees:]

        # 映射回原始solutions索引
        knee_indices = [self.pareto_indices[i] for i in top_k_indices]

        # 调试信息
        print(f"  Quality scores range: [{quality_score.min():.3f}, {quality_score.max():.3f}]")
        print(f"  Diversity scores range: [{diversity_score.min():.3f}, {diversity_score.max():.3f}]")
        print(f"  Balance scores range: [{balance_score.min():.3f}, {balance_score.max():.3f}]")
        print(f"  Final knee points: {len(knee_indices)}")

        return knee_indices
    
    def analyze_objective_conflicts(self) -> Dict:
        """分析目标冲突"""
        if len(self.pareto_front) == 0:
            self.find_pareto_front()
        
        # 计算帕累托前沿上的相关系数矩阵
        corr_matrix = np.corrcoef(self.pareto_front.T)
        
        # 提取冲突关系（负相关）
        conflicts = {}
        for i in range(self.n_objectives):
            for j in range(i+1, self.n_objectives):
                correlation = corr_matrix[i, j]
                if abs(correlation) > 0.1:  # 相关性阈值（包括正负）
                    name1 = self.objective_names[i]
                    name2 = self.objective_names[j]
                    conflicts[f"{name1} vs {name2}"] = correlation  # 保留符号
        
        return conflicts
    
    def plot_pareto_analysis_clean(self, save_path: str = "./pareto_analysis_complete.png") -> None:
        """生成清晰的帕累托分析可视化"""
        if len(self.pareto_front) == 0:
            self.find_pareto_front()
        
        knee_indices = self.find_knee_points_improved()
        
        # 创建三个主要图：散点图矩阵、3D图、冲突矩阵
        
        # 1. 关键目标对的散点图 (2x3布局)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # 选择最重要的6个目标对
        important_pairs = [
            (0, 1),  # Throughput vs Balance
            (0, 2),  # Throughput vs Efficiency
            (0, 4),  # Throughput vs Stability
            (1, 2),  # Balance vs Efficiency
            (2, 4),  # Efficiency vs Stability
            (1, 4)   # Balance vs Stability
        ]
        
        for idx, (i, j) in enumerate(important_pairs):
            ax = axes[idx]
            
            # 所有解
            ax.scatter(self.objective_values[:, j], self.objective_values[:, i], 
                      alpha=0.3, s=1, color='lightblue', label='All solutions')
            
            # 帕累托前沿
            ax.scatter(self.pareto_front[:, j], self.pareto_front[:, i], 
                      alpha=0.8, s=15, color='red', label='Pareto front')
            
            # 膝点
            if knee_indices:
                knee_objectives = self.objective_values[knee_indices]
                ax.scatter(knee_objectives[:, j], knee_objectives[:, i], 
                          alpha=1.0, s=40, color='gold', marker='*', 
                          label='Knee points', edgecolors='black', linewidth=0.5)
            
            ax.set_xlabel(self.objective_names[j], fontsize=12)
            ax.set_ylabel(self.objective_names[i], fontsize=12)
            ax.set_title(f'{self.objective_names[i]} vs {self.objective_names[j]}', fontsize=12)
            
            if idx == 0:
                ax.legend()
        
        plt.suptitle('Pareto Analysis: Key Objective Pairs', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 目标冲突矩阵
        corr_matrix = np.corrcoef(self.pareto_front.T)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                   xticklabels=self.objective_names, yticklabels=self.objective_names,
                   ax=ax, vmin=-1, vmax=1, fmt='.3f')
        ax.set_title('Objective Conflicts Matrix (Red = Conflict)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path.replace('.png', '_conflicts.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 3D帕累托前沿（前3个最重要目标）
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 所有解（采样以提高性能）
        n_sample = min(1000, len(self.objective_values))
        sample_idx = np.random.choice(len(self.objective_values), n_sample, replace=False)
        
        ax.scatter(self.objective_values[sample_idx, 0], 
                  self.objective_values[sample_idx, 1], 
                  self.objective_values[sample_idx, 2],
                  alpha=0.3, s=1, color='lightblue', label='All solutions')
        
        # 帕累托前沿
        ax.scatter(self.pareto_front[:, 0], self.pareto_front[:, 1], self.pareto_front[:, 2],
                  alpha=0.8, s=20, color='red', label='Pareto front')
        
        # 膝点
        if knee_indices:
            knee_objectives = self.objective_values[knee_indices]
            ax.scatter(knee_objectives[:, 0], knee_objectives[:, 1], knee_objectives[:, 2],
                      alpha=1.0, s=50, color='gold', marker='*', label='Knee points',
                      edgecolors='black', linewidth=1)
        
        ax.set_xlabel(self.objective_names[0])
        ax.set_ylabel(self.objective_names[1])
        ax.set_zlabel(self.objective_names[2])
        ax.set_title('3D Pareto Front Visualization')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(save_path.replace('.png', '_3d.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Clean Pareto analysis plots saved")
    
    def generate_report(self, save_path: str = "./pareto_analysis_report.txt") -> None:
        """生成详细的帕累托分析报告"""
        if len(self.pareto_front) == 0:
            self.find_pareto_front()

        conflicts = self.analyze_objective_conflicts()
        knee_indices = self.find_knee_points_improved()
        hypervolume = self.compute_hypervolume()

        # 🔧 修复4: 报告中添加可行性检查
        unstable_count = np.sum(self.objective_values[:, 4] < 0.5)
        unstable_in_pareto = np.sum(self.pareto_front[:, 4] < 0.5)

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("垂直分层队列系统帕累托最优解集分析报告\n")
            f.write("="*60 + "\n\n")

            f.write(f"总解数: {len(self.objective_values)}\n")
            f.write(f"不稳定解数 (Stability<0.5): {unstable_count} ({unstable_count/len(self.objective_values)*100:.1f}%)\n")
            f.write(f"帕累托最优解数: {len(self.pareto_front)}\n")
            f.write(f"帕累托中不稳定解: {unstable_in_pareto}\n")
            f.write(f"帕累托比例: {len(self.pareto_front)/len(self.objective_values)*100:.2f}%\n")
            f.write(f"超体积指标: {hypervolume:.4f}\n")
            f.write(f"膝点数量: {len(knee_indices)} ({len(knee_indices)/len(self.pareto_front)*100:.1f}%)\n\n")
            
            f.write("目标统计信息:\n")
            f.write("-"*40 + "\n")
            for i, name in enumerate(self.objective_names):
                all_values = self.objective_values[:, i]
                pareto_values = self.pareto_front[:, i]
                
                f.write(f"{name}:\n")
                f.write(f"  全体解: {np.mean(all_values):.3f} ± {np.std(all_values):.3f}\n")
                f.write(f"  帕累托解: {np.mean(pareto_values):.3f} ± {np.std(pareto_values):.3f}\n")
                f.write(f"  范围: [{np.min(pareto_values):.3f}, {np.max(pareto_values):.3f}]\n\n")
            
            f.write("主要目标关系:\n")
            f.write("-"*40 + "\n")
            # 按相关性绝对值排序，显示最强的关系（正负都显示）
            for conflict_pair, strength in sorted(conflicts.items(), key=lambda x: abs(x[1]), reverse=True):
                f.write(f"{conflict_pair}: {strength:.3f}\n")
            
            if knee_indices:
                f.write(f"\n膝点解详情:\n")
                f.write("-"*40 + "\n")
                for i, idx in enumerate(knee_indices):
                    f.write(f"膝点 {i+1}:\n")
                    for j, name in enumerate(self.objective_names):
                        f.write(f"  {name}: {self.objective_values[idx, j]:.3f}\n")
                    f.write("\n")
        
        print(f"Report saved to: {save_path}")
    
    def compute_hypervolume(self, reference_point: Optional[np.ndarray] = None) -> float:
        """计算超体积指标"""
        if len(self.pareto_front) == 0:
            return 0.0
        
        if reference_point is None:
            # 使用最小值作为参考点
            reference_point = np.min(self.objective_values, axis=0) - 0.1
        
        # 简化的超体积计算
        hypervolume = 0.0
        for point in self.pareto_front:
            volume = np.prod(np.maximum(0, point - reference_point))
            if volume > 0:
                hypervolume += volume
        
        return hypervolume


def main():
    """主函数：运行完整的帕累托分析（最终版）"""
    print("Starting Pareto Optimal Set Analysis for Vertical Stratified Queuing System (Enhanced Version)")
    print("=" * 80)
    
    # 创建环境
    env = DRLOptimizedQueueEnvFixed()
    
    # 验证环境配置
    print(f"✅ Environment Configuration:")
    print(f"   Layers: {env.n_layers}")
    print(f"   Heights: {env.heights}")
    print(f"   Capacities: {env.capacities}")
    print(f"   Service rates: {env.base_service_rates}")
    
    # 创建分析器
    analyzer = ParetoAnalyzer(env)
    
    # 生成解集 - 增加到10000个点
    print("\n1. Generating random solutions...")
    analyzer.generate_random_solutions(n_solutions=10000)
    
    # 识别帕累托前沿
    print("\n2. Finding Pareto front...")
    analyzer.find_pareto_front()
    
    # 分析目标冲突
    print("\n3. Analyzing objective conflicts...")
    conflicts = analyzer.analyze_objective_conflicts()
    print("Main correlations:")
    for conflict, strength in sorted(conflicts.items(), key=lambda x: abs(x[1]), reverse=True)[:3]:
        print(f"  {conflict}: {strength:.3f}")
    
    # 生成可视化
    print("\n4. Generating visualizations...")
    analyzer.plot_pareto_analysis_clean("./pareto_analysis_complete.png")
    
    # 生成报告
    print("\n5. Generating report...")
    analyzer.generate_report("./pareto_analysis_report.txt")
    
    print("\n" + "=" * 80)
    print("Pareto analysis completed!")
    print("Check the following files:")
    print("  - pareto_analysis_complete.png: Key objective pairs")
    print("  - pareto_analysis_complete_conflicts.png: Conflict matrix")
    print("  - pareto_analysis_complete_3d.png: 3D Pareto front")
    print("  - pareto_analysis_report.txt: Detailed analysis report")


if __name__ == "__main__":
    main()