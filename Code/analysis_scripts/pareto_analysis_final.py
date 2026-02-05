"""
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
    """Pareto Optimal Set Analyzer (Final Fixed Version)"""
    
    def __init__(self, env):
        """
        Args:
            env: Vertical stratified queue environment instance
        """
        self.env = env
        self.objective_names = [
            'Throughput', 'Balance', 'Efficiency',
            'Transfer', 'Stability', 'Anti-Penalty'
        ]
        self.n_objectives = len(self.objective_names)

        # Store evaluation results
        self.solutions = []
        self.objective_values = []
        self.pareto_indices = []
        self.pareto_front = []

        print(f"ParetoAnalyzer initialized with {self.n_objectives} objectives")
    
    def evaluate_solution(self, policy_params: Dict, n_episodes: int = 5) -> np.ndarray:
        """
        Evaluate multi-objective performance of a single solution

        Args:
            policy_params: Policy parameters dictionary
            n_episodes: Number of evaluation episodes

        Returns:
            6-dimensional objective vector
        """
        objective_values = np.zeros(self.n_objectives)
        
        for episode in range(n_episodes):
            obs, _ = self.env.reset()
            episode_objectives = np.zeros(self.n_objectives)
            steps = 0
            
            while steps < 200:  # Maximum step limit
                # Generate action based on policy parameters
                action = self._policy_to_action(obs, policy_params)
                
                # Execute action
                next_obs, reward, terminated, truncated, info = self.env.step(action)

                # Extract multi-objective reward components (using observation and info after step)
                objectives = self._extract_objectives(next_obs, action, reward, info)
                episode_objectives += objectives
                
                obs = next_obs
                steps += 1
                
                if terminated or truncated:
                    break
            
            # Strict evaluation: ensure policy continuity and stability
            if steps >= 50:  # Increase minimum step requirement to 50 steps
                # Normal averaging by steps
                objective_values += episode_objectives / max(steps, 1)
            elif steps >= 20:
                # Medium length episodes get partial score (penalize unstable policies)
                penalty_factor = steps / 50.0  # Linear penalty
                objective_values += (episode_objectives / max(steps, 1)) * penalty_factor
            else:
                # Short episodes (<20 steps) considered invalid policies
                objective_values += np.zeros(self.n_objectives)
        
        return objective_values / n_episodes
    
    def _policy_to_action(self, obs, policy_params: Dict) -> Dict:
        """Convert policy parameters to environment action"""
        # Extract observation information
        if isinstance(obs, dict):
            utilization = obs.get('utilization_rates', np.ones(5) * 0.5)
            queue_lengths = obs.get('queue_lengths', np.ones(5))
        elif isinstance(obs, np.ndarray):
            queue_lengths = obs[:5] if len(obs) >= 5 else np.ones(5)
            utilization = obs[5:10] if len(obs) >= 10 else np.ones(5) * 0.5
        else:
            queue_lengths = np.ones(5)
            utilization = np.ones(5) * 0.5
        
        # Generate action based on policy parameters
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
        Extract multi-objective vector from environment step return (fixed version - use environment reward components)

        Core fix:
        1. Prioritize using environment-provided reward component decomposition (reward_components)
        2. Avoid duplicate calculation and timing inconsistency issues
        3. Ensure complete consistency with environment reward function

        Returns:
            6-dimensional objective vector, all objectives are larger is better
        """

        # Method 1: Directly use environment-provided reward components (most accurate)
        if 'reward_components' in info:
            components = info['reward_components']

            throughput_obj = components['throughput']
            balance_obj = components['balance']
            efficiency_obj = components['efficiency']
            transfer_obj = components['transfer']
            stability_obj = components['stability']

            # Convert negative penalty to positive objective
            penalty_obj = -(components['congestion'] + components['instability'])

            return np.array([throughput_obj, balance_obj, efficiency_obj, transfer_obj, stability_obj, penalty_obj])

        # Method 2: Backup calculation (if environment does not provide reward_components)
        print("Warning: Environment does not provide reward_components, using backup calculation method")

        # Extract statistics from info
        service_counts = np.array(info.get('service_counts', np.zeros(5)))
        transfer_counts = np.array(info.get('transfer_counts', np.zeros(5)))

        # Extract queue state from obs
        if isinstance(obs, dict):
            queue_lengths = obs.get('queue_lengths', np.zeros(5))
        elif isinstance(obs, np.ndarray) and len(obs) >= 5:
            queue_lengths = obs[:5]
        else:
            queue_lengths = np.zeros(5)

        # Environment fixed parameters
        capacities = np.array([8, 6, 4, 3, 2])

        # 1. Throughput objective
        throughput_obj = 10.0 * np.sum(service_counts)

        # 2. Load balance objective (Gini coefficient)
        utilization_rates = queue_lengths / (capacities + 1e-8)
        if np.sum(utilization_rates) > 1e-6:
            sorted_util = np.sort(utilization_rates)
            n = len(sorted_util)
            gini = (2 * np.sum((np.arange(n) + 1) * sorted_util)) / (n * np.sum(sorted_util)) - (n + 1) / n
            balance_obj = 5.0 * (1.0 - gini)
        else:
            balance_obj = 5.0

        # 3. Efficiency objective
        service_total = np.sum(service_counts)
        base_energy = 1.0 + np.sum(action['service_intensities']) + action['arrival_multiplier'][0] * 0.5 + np.sum(action['emergency_transfers']) * 0.2
        if base_energy > 1e-6:
            efficiency_obj = 3.0 * service_total / base_energy
        else:
            efficiency_obj = 0.0

        # 4. 转移Efficiency objective
        transfer_obj = 0.0
        for i in range(4):
            if transfer_counts[i] > 0:
                upper_pressure = queue_lengths[i] / (capacities[i] + 1e-8)
                lower_util = queue_lengths[i+1] / (capacities[i+1] + 1e-8)
                if upper_pressure > lower_util:
                    transfer_obj += 2.0 * transfer_counts[i]

        # 5. Stability objective
        stability_obj = info.get('stability_score', 0.0)

        # 6. Anti-penalty objective
        congestion_levels = np.maximum(0, (queue_lengths - 0.8 * capacities) / capacities)
        congestion_penalty = -20.0 * np.sum(congestion_levels)

        # Use current_service_rates and current_arrival_rate from info to calculate instability penalty
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
        """Generate random solution set for Pareto analysis"""
        print(f"Generating {n_solutions} random solutions...")
        start_time = time.time()
        
        self.solutions = []
        self.objective_values = []
        
        for i in range(n_solutions):
            if i % 500 == 0:
                elapsed = time.time() - start_time
                eta = elapsed * (n_solutions - i) / (i + 1) if i > 0 else 0
                print(f"  Progress: {i}/{n_solutions} ({i/n_solutions*100:.1f}%) - ETA: {eta/60:.1f}min")
            
            # Fix: Generate physically reasonable policy parameter combinations
            # Avoid extreme arrival rate + service rate combinations causing system overload

            arrival_factor = np.random.uniform(0.5, 3.0)  # Limit arrival multiplier to max 3x

            # Adjust service capacity range based on arrival multiplier to ensure system operability
            if arrival_factor > 2.0:
                # High arrival rate requires higher base service capacity
                base_service_range = (0.8, 1.5)
                adaptation_range = (0.1, 0.4)
            elif arrival_factor > 1.5:
                # Medium arrival rate requires medium service capacity
                base_service_range = (0.5, 1.3)
                adaptation_range = (0.0, 0.6)
            else:
                # Low arrival rate can use wider service capacity range
                base_service_range = (0.3, 1.2)
                adaptation_range = (0.0, 0.8)

            policy_params = {
                'base_service': np.random.uniform(*base_service_range),
                'adaptation': np.random.uniform(*adaptation_range),
                'arrival_factor': arrival_factor,
                'transfer_threshold': np.random.uniform(0.4, 0.9)  # More reasonable transfer threshold
            }
            
            # Evaluate solution
            objectives = self.evaluate_solution(policy_params, n_episodes=5)
            
            self.solutions.append(policy_params)
            self.objective_values.append(objectives)
        
        self.objective_values = np.array(self.objective_values)
        elapsed = time.time() - start_time
        print(f"Generated {len(self.solutions)} solutions in {elapsed/60:.1f} minutes")
    
    def find_pareto_front_efficient(self) -> None:
        """Efficient Pareto front identification algorithm (Non-dominated Sorting) + feasibility filtering"""
        print(f"Finding Pareto front among {len(self.objective_values)} solutions...")
        start_time = time.time()

        n_solutions = len(self.objective_values)

        # Fix 1: First filter out infeasible solutions (system crash solutions with Stability=0)
        STABILITY_THRESHOLD = 0.5  # Minimum stability threshold
        feasible_mask = self.objective_values[:, 4] > STABILITY_THRESHOLD  # Stability is the 5th objective

        feasible_indices = np.where(feasible_mask)[0]
        print(f"  Filtering feasible solutions: {len(feasible_indices)}/{n_solutions} are stable")

        if len(feasible_indices) == 0:
            print("  ⚠️  No feasible solutions found! Using all solutions...")
            feasible_indices = np.arange(n_solutions)

        feasible_objectives = self.objective_values[feasible_indices]
        n_feasible = len(feasible_indices)

        domination_count = np.zeros(n_feasible)  # Domination count
        dominated_solutions = [[] for _ in range(n_feasible)]  # List of dominated solutions

        # Calculate dominance relations (only among feasible solutions)
        for i in range(n_feasible):
            if i % 1000 == 0:
                print(f"  Processing solution {i}/{n_feasible}")

            for j in range(i + 1, n_feasible):
                # Check if i dominates j
                i_dominates_j = (np.all(feasible_objectives[i] >= feasible_objectives[j]) and
                               np.any(feasible_objectives[i] > feasible_objectives[j]))

                # Check if j dominates i
                j_dominates_i = (np.all(feasible_objectives[j] >= feasible_objectives[i]) and
                               np.any(feasible_objectives[j] > feasible_objectives[i]))

                if i_dominates_j:
                    dominated_solutions[i].append(j)
                    domination_count[j] += 1
                elif j_dominates_i:
                    dominated_solutions[j].append(i)
                    domination_count[i] += 1

        # Find non-dominated solutions (Domination count is 0)
        pareto_mask = domination_count == 0
        local_pareto_indices = np.where(pareto_mask)[0]

        # Map back to original indices
        self.pareto_indices = feasible_indices[local_pareto_indices]
        self.pareto_front = self.objective_values[self.pareto_indices]

        elapsed = time.time() - start_time
        print(f"Found {len(self.pareto_indices)} Pareto optimal solutions in {elapsed:.1f} seconds")
        print(f"  Pareto ratio: {len(self.pareto_indices)/n_solutions*100:.2f}%")
    
    def find_pareto_front(self) -> None:
        """Pareto front identification (call efficient version)"""
        self.find_pareto_front_efficient()
    
    def find_knee_points_improved(self) -> List[int]:
        """
        Improved knee point detection algorithm (based on sparsity and trade-off analysis)

        Knee point definition: Most representative solutions on Pareto front, satisfying:
        1. Close to ideal point (high quality)
        2. Sparse distribution on front (strong representativeness)
        3. Reasonable trade-off between objectives
        """
        if len(self.pareto_front) < 3:
            return list(range(len(self.pareto_front)))

        print("Finding knee points using improved method...")

        # Fixed number of knee points (avoid instability of threshold method)
        n_pareto = len(self.pareto_front)
        target_knees = max(5, min(15, n_pareto // 20))  # 5-15 points, about 5%

        print(f"  Target knee points: {target_knees} (from {n_pareto} Pareto solutions)")

        # Normalize Pareto front
        ideal_point = np.max(self.pareto_front, axis=0)
        nadir_point = np.min(self.pareto_front, axis=0)
        range_vector = np.maximum(ideal_point - nadir_point, 1e-8)
        normalized_front = (self.pareto_front - nadir_point) / range_vector

        # Method 1: Calculate distance to ideal point (smaller is better)
        ideal_distances = np.linalg.norm(normalized_front - 1.0, axis=1)

        # Method 2: Calculate sparsity score (using k-nearest neighbor distance)
        # Average distance to k nearest neighbors (larger means more sparse/representative)
        k = min(10, n_pareto // 10)
        distances_matrix = cdist(normalized_front, normalized_front, metric='euclidean')
        np.fill_diagonal(distances_matrix, np.inf)  # Exclude self

        sparsity_scores = np.zeros(n_pareto)
        for i in range(n_pareto):
            # Find average distance to k nearest neighbors
            nearest_k_distances = np.partition(distances_matrix[i], k)[:k]
            sparsity_scores[i] = np.mean(nearest_k_distances)

        # Method 3: Objective balance (avoid extreme solutions)
        # Use coefficient of variation (CV): std/mean, smaller means more balanced
        uniformity_scores = np.zeros(n_pareto)
        for i in range(n_pareto):
            point = normalized_front[i]
            mean_val = np.mean(point)
            if mean_val > 1e-6:
                cv = np.std(point) / mean_val
                uniformity_scores[i] = 1.0 / (1.0 + cv)  # Convert to score (larger means more balanced)
            else:
                uniformity_scores[i] = 0.0

        # Comprehensive score (multi-criteria decision)
        # Normalize each score to [0,1]
        quality_score = 1.0 - (ideal_distances - ideal_distances.min()) / (ideal_distances.max() - ideal_distances.min() + 1e-8)
        diversity_score = (sparsity_scores - sparsity_scores.min()) / (sparsity_scores.max() - sparsity_scores.min() + 1e-8)
        balance_score = uniformity_scores

        # Weighted combination (quality 40%, diversity 40%, balance 20%)
        total_scores = quality_score * 0.4 + diversity_score * 0.4 + balance_score * 0.2

        # Directly select top target_knees points with highest scores
        top_k_indices = np.argsort(total_scores)[-target_knees:]

        # Map back to original solutions indices
        knee_indices = [self.pareto_indices[i] for i in top_k_indices]

        # Debug information
        print(f"  Quality scores range: [{quality_score.min():.3f}, {quality_score.max():.3f}]")
        print(f"  Diversity scores range: [{diversity_score.min():.3f}, {diversity_score.max():.3f}]")
        print(f"  Balance scores range: [{balance_score.min():.3f}, {balance_score.max():.3f}]")
        print(f"  Final knee points: {len(knee_indices)}")

        return knee_indices
    
    def analyze_objective_conflicts(self) -> Dict:
        """Analyze objective conflicts"""
        if len(self.pareto_front) == 0:
            self.find_pareto_front()
        
        # Calculate correlation coefficient matrix on Pareto front
        corr_matrix = np.corrcoef(self.pareto_front.T)
        
        # Extract conflict relations (negative correlation)
        conflicts = {}
        for i in range(self.n_objectives):
            for j in range(i+1, self.n_objectives):
                correlation = corr_matrix[i, j]
                if abs(correlation) > 0.1:  # Correlation threshold (including positive and negative)
                    name1 = self.objective_names[i]
                    name2 = self.objective_names[j]
                    conflicts[f"{name1} vs {name2}"] = correlation  # Keep sign
        
        return conflicts
    
    def plot_pareto_analysis_clean(self, save_path: str = "./pareto_analysis_complete.png") -> None:
        """Generate clear Pareto analysis visualization"""
        if len(self.pareto_front) == 0:
            self.find_pareto_front()
        
        knee_indices = self.find_knee_points_improved()
        
        # Create three main plots: scatter plot matrix, 3D plot, conflict matrix
        
        # 1. Scatter plots of key objective pairs (2x3layout)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Select 6 most important objective pairs
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
            
            # All solutions
            ax.scatter(self.objective_values[:, j], self.objective_values[:, i], 
                      alpha=0.3, s=1, color='lightblue', label='All solutions')
            
            # Pareto front
            ax.scatter(self.pareto_front[:, j], self.pareto_front[:, i], 
                      alpha=0.8, s=15, color='red', label='Pareto front')
            
            # Knee points
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
        
        # 2. Objective conflict matrix
        corr_matrix = np.corrcoef(self.pareto_front.T)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, 
                   xticklabels=self.objective_names, yticklabels=self.objective_names,
                   ax=ax, vmin=-1, vmax=1, fmt='.3f')
        ax.set_title('Objective Conflicts Matrix (Red = Conflict)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path.replace('.png', '_conflicts.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 3DPareto front（前3个最重要目标）
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # All solutions（采样以提高性能）
        n_sample = min(1000, len(self.objective_values))
        sample_idx = np.random.choice(len(self.objective_values), n_sample, replace=False)
        
        ax.scatter(self.objective_values[sample_idx, 0], 
                  self.objective_values[sample_idx, 1], 
                  self.objective_values[sample_idx, 2],
                  alpha=0.3, s=1, color='lightblue', label='All solutions')
        
        # Pareto front
        ax.scatter(self.pareto_front[:, 0], self.pareto_front[:, 1], self.pareto_front[:, 2],
                  alpha=0.8, s=20, color='red', label='Pareto front')
        
        # Knee points
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
        """Generate detailed Pareto analysis report"""
        if len(self.pareto_front) == 0:
            self.find_pareto_front()

        conflicts = self.analyze_objective_conflicts()
        knee_indices = self.find_knee_points_improved()
        hypervolume = self.compute_hypervolume()

        # 🔧 Fix 4: Add feasibility check in report
        unstable_count = np.sum(self.objective_values[:, 4] < 0.5)
        unstable_in_pareto = np.sum(self.pareto_front[:, 4] < 0.5)

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("Vertical Stratified Queue System Pareto Optimal Set Analysis Report\n")
            f.write("="*60 + "\n\n")

            f.write(f"Total solutions: {len(self.objective_values)}\n")
            f.write(f"Number of unstable solutions (Stability<0.5): {unstable_count} ({unstable_count/len(self.objective_values)*100:.1f}%)\n")
            f.write(f"Pareto optimal solutions: {len(self.pareto_front)}\n")
            f.write(f"Unstable solutions in Pareto: {unstable_in_pareto}\n")
            f.write(f"Pareto ratio: {len(self.pareto_front)/len(self.objective_values)*100:.2f}%\n")
            f.write(f"Hypervolume indicator: {hypervolume:.4f}\n")
            f.write(f"Number of knee points: {len(knee_indices)} ({len(knee_indices)/len(self.pareto_front)*100:.1f}%)\n\n")
            
            f.write("Objective statistics:\n")
            f.write("-"*40 + "\n")
            for i, name in enumerate(self.objective_names):
                all_values = self.objective_values[:, i]
                pareto_values = self.pareto_front[:, i]
                
                f.write(f"{name}:\n")
                f.write(f"  All solutions: {np.mean(all_values):.3f} ± {np.std(all_values):.3f}\n")
                f.write(f"  Pareto solutions: {np.mean(pareto_values):.3f} ± {np.std(pareto_values):.3f}\n")
                f.write(f"  Range: [{np.min(pareto_values):.3f}, {np.max(pareto_values):.3f}]\n\n")
            
            f.write("Main objective relations:\n")
            f.write("-"*40 + "\n")
            # Sort by absolute correlation, show strongest relations (both positive and negative)
            for conflict_pair, strength in sorted(conflicts.items(), key=lambda x: abs(x[1]), reverse=True):
                f.write(f"{conflict_pair}: {strength:.3f}\n")
            
            if knee_indices:
                f.write(f"\nKnee points details:\n")
                f.write("-"*40 + "\n")
                for i, idx in enumerate(knee_indices):
                    f.write(f"Knee point {i+1}:\n")
                    for j, name in enumerate(self.objective_names):
                        f.write(f"  {name}: {self.objective_values[idx, j]:.3f}\n")
                    f.write("\n")
        
        print(f"Report saved to: {save_path}")
    
    def compute_hypervolume(self, reference_point: Optional[np.ndarray] = None) -> float:
        """Calculate Hypervolume indicator"""
        if len(self.pareto_front) == 0:
            return 0.0
        
        if reference_point is None:
            # Use minimum value as reference point
            reference_point = np.min(self.objective_values, axis=0) - 0.1
        
        # Simplified hypervolume calculation
        hypervolume = 0.0
        for point in self.pareto_front:
            volume = np.prod(np.maximum(0, point - reference_point))
            if volume > 0:
                hypervolume += volume
        
        return hypervolume


def main():
    """Main function: Run complete Pareto analysis (final version)"""
    print("Starting Pareto Optimal Set Analysis for Vertical Stratified Queuing System (Enhanced Version)")
    print("=" * 80)
    
    # Create environment
    env = DRLOptimizedQueueEnvFixed()

    # Verify environment configuration
    print(f"✅ Environment Configuration:")
    print(f"   Layers: {env.n_layers}")
    print(f"   Heights: {env.heights}")
    print(f"   Capacities: {env.capacities}")
    print(f"   Service rates: {env.base_service_rates}")
    
    # Create analyzer
    analyzer = ParetoAnalyzer(env)
    
    # Generate solution set - increase to 10000 points
    print("\n1. Generating random solutions...")
    analyzer.generate_random_solutions(n_solutions=10000)
    
    # Identify Pareto front
    print("\n2. Finding Pareto front...")
    analyzer.find_pareto_front()
    
    # Analyze objective conflicts
    print("\n3. Analyzing objective conflicts...")
    conflicts = analyzer.analyze_objective_conflicts()
    print("Main correlations:")
    for conflict, strength in sorted(conflicts.items(), key=lambda x: abs(x[1]), reverse=True)[:3]:
        print(f"  {conflict}: {strength:.3f}")
    
    # Generate visualizations
    print("\n4. Generating visualizations...")
    analyzer.plot_pareto_analysis_clean("./pareto_analysis_complete.png")
    
    # Generate report
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