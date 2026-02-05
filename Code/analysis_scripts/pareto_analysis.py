"""
Pareto Optimal Set Analysis for Vertical Stratified Queuing System
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
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from env.drl_optimized_env_fixed import DRLOptimizedQueueEnvFixed


class ParetoAnalyzer:
    """Pareto Optimal Set Analyzer"""
    
    def __init__(self, env):
        """
        Args:
            env: Vertical stratified queue environment instance
        """
        self.env = env
        self.objective_names = [
            'Throughput', 'Balance', 'Efficiency',
            'Transfer', 'Stability', 'Penalty'
        ]
        self.n_objectives = len(self.objective_names)

        # Store evaluation results
        self.solutions = []
        self.objective_values = []
        self.pareto_indices = []
        self.pareto_front = []

        print(f"ParetoAnalyzer initialized with {self.n_objectives} objectives")
    
    def evaluate_solution(self, policy_params: Dict, n_episodes: int = 10) -> np.ndarray:
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
                
                # Extract multi-objective reward components
                objectives = self._extract_objectives(obs, action, reward, info)
                episode_objectives += objectives
                
                obs = next_obs
                steps += 1
                
                if terminated or truncated:
                    break
            
            objective_values += episode_objectives / steps
        
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
            
            'arrival_multiplier': np.array([policy_params.get('arrival_factor', 1.0)], dtype=np.float32),
            
            'emergency_transfers': (utilization > policy_params.get('transfer_threshold', 0.8)).astype(np.int8)
        }
        
        return action
    
    def _extract_objectives(self, obs, action: Dict, reward: float, info: Dict) -> np.ndarray:
        """Extract multi-objective components from environment feedback"""
        # Extract key information from observation
        if isinstance(obs, dict):
            queue_lengths = obs.get('queue_lengths', np.ones(5))
            utilization = obs.get('utilization_rates', np.ones(5) * 0.5)
            service_rates = obs.get('service_rates', np.ones(5))
        elif isinstance(obs, np.ndarray) and len(obs) >= 10:
            queue_lengths = obs[:5]
            utilization = obs[5:10]
            service_rates = obs[15:20] if len(obs) >= 20 else np.ones(5)
        else:
            queue_lengths = np.ones(5)
            utilization = np.ones(5) * 0.5
            service_rates = np.ones(5)
        
        # Use information provided by environment
        throughput = info.get('throughput', np.sum(action['service_intensities'] * utilization))
        service_counts = info.get('service_counts', [0] * 5)
        
        # Calculate each objective component
        # 1. Throughput objective
        throughput_obj = float(throughput)
        
        # 2. Load balance objective (Gini coefficient)
        if np.sum(utilization) > 0:
            sorted_util = np.sort(utilization)
            n = len(sorted_util)
            gini = (2 * np.sum((np.arange(n) + 1) * sorted_util)) / (n * np.sum(sorted_util) + 1e-8) - (n + 1) / n
            balance = 1.0 - abs(gini)
        else:
            balance = 1.0
        
        # 3. Efficiency objective
        total_service = np.sum(action['service_intensities']) + action['arrival_multiplier'][0]
        efficiency = throughput_obj / (1.0 + total_service)
        
        # 4. Transfer efficiency
        transfer_count = np.sum(action['emergency_transfers'])
        transfer_efficiency = transfer_count * balance if transfer_count > 0 else 0.1
        
        # 5. Stability objective
        stability_score = info.get('stability_score', 1.0 / (1.0 + np.var(utilization)))
        stability = float(stability_score) if stability_score > 0 else 0.1
        
        # 6. Penalty minimization (convert to positive)
        capacities = np.array([8, 6, 4, 3, 2])
        congestion_penalty = np.sum(np.maximum(0, queue_lengths - 0.8 * capacities))
        penalty = 1.0 / (1.0 + congestion_penalty)  # Convert to positive value, larger is better
        
        return np.array([throughput_obj, balance, efficiency, transfer_efficiency, stability, penalty])
    
    def generate_random_solutions(self, n_solutions: int = 1000) -> None:
        """Generate random solution set for Pareto analysis"""
        print(f"Generating {n_solutions} random solutions...")
        
        self.solutions = []
        self.objective_values = []
        
        for i in range(n_solutions):
            if i % 100 == 0:
                print(f"  Progress: {i}/{n_solutions}")
            
            # Randomly generate policy parameters
            policy_params = {
                'base_service': np.random.uniform(0.5, 1.5),
                'adaptation': np.random.uniform(0.0, 0.5),
                'arrival_factor': np.random.uniform(0.8, 1.2),
                'transfer_threshold': np.random.uniform(0.6, 0.9)
            }
            
            # Evaluate solution
            objectives = self.evaluate_solution(policy_params, n_episodes=3)
            
            self.solutions.append(policy_params)
            self.objective_values.append(objectives)
        
        self.objective_values = np.array(self.objective_values)
        print(f"Generated {len(self.solutions)} solutions")
    
    def find_pareto_front(self) -> None:
        """Identify Pareto front"""
        n_solutions = len(self.objective_values)
        is_pareto = np.ones(n_solutions, dtype=bool)
        
        for i in range(n_solutions):
            for j in range(n_solutions):
                if i != j:
                    # Check if j dominates i
                    if np.all(self.objective_values[j] >= self.objective_values[i]) and \
                       np.any(self.objective_values[j] > self.objective_values[i]):
                        is_pareto[i] = False
                        break
        
        self.pareto_indices = np.where(is_pareto)[0]
        self.pareto_front = self.objective_values[self.pareto_indices]
        
        print(f"Found {len(self.pareto_indices)} Pareto optimal solutions")
    
    def compute_dominance_matrix(self) -> np.ndarray:
        """Calculate dominance relation matrix"""
        n = len(self.objective_values)
        dominance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Check if i dominates j
                    if np.all(self.objective_values[i] >= self.objective_values[j]) and \
                       np.any(self.objective_values[i] > self.objective_values[j]):
                        dominance_matrix[i, j] = 1
                    # Check if j dominates i
                    elif np.all(self.objective_values[j] >= self.objective_values[i]) and \
                         np.any(self.objective_values[j] > self.objective_values[i]):
                        dominance_matrix[i, j] = -1
        
        return dominance_matrix
    
    def analyze_objective_conflicts(self) -> Dict:
        """Analyze objective conflicts"""
        if len(self.pareto_front) == 0:
            self.find_pareto_front()
        
        # Calculate correlation coefficient matrix on Pareto front
        corr_matrix = np.corrcoef(self.pareto_front.T)
        
        # Identify main conflicts
        conflicts = {}
        for i in range(self.n_objectives):
            for j in range(i + 1, self.n_objectives):
                conflict_strength = -corr_matrix[i, j]  # Negative correlation indicates conflict
                if conflict_strength > 0.3:  # Threshold
                    conflicts[f"{self.objective_names[i]} vs {self.objective_names[j]}"] = conflict_strength
        
        return conflicts
    
    def compute_hypervolume(self, reference_point: Optional[np.ndarray] = None) -> float:
        """Calculate hypervolume indicator"""
        if len(self.pareto_front) == 0:
            return 0.0
        
        if reference_point is None:
            # Use minimum value as reference point
            reference_point = np.min(self.objective_values, axis=0) - 0.1
        
        # Simplified hypervolume calculation (only for small-scale problems)
        hypervolume = 0.0
        for point in self.pareto_front:
            volume = np.prod(point - reference_point)
            if volume > 0:
                hypervolume += volume
        
        return hypervolume
    
    def find_knee_points(self) -> List[int]:
        """Find knee points"""
        if len(self.pareto_front) < 3:
            return []
        
        # Use PCA to reduce to 2D for knee point detection
        pca = PCA(n_components=2)
        front_2d = pca.fit_transform(self.pareto_front)
        
        # Calculate curvature
        knee_indices = []
        for i in range(1, len(front_2d) - 1):
            # Simplified curvature calculation
            v1 = front_2d[i] - front_2d[i-1]
            v2 = front_2d[i+1] - front_2d[i]
            
            # Angle change as proxy for curvature
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            curvature = 1 - cos_angle
            
            if curvature > 0.5:  # Threshold
                knee_indices.append(self.pareto_indices[i])
        
        return knee_indices
    
    def weight_optimization(self, weights: np.ndarray) -> Dict:
        """Weight optimization method to solve Pareto optimal solution"""
        def objective(params_array):
            # Convert parameter array to policy parameters
            policy_params = {
                'base_service': params_array[0],
                'adaptation': params_array[1],
                'arrival_factor': params_array[2],
                'transfer_threshold': params_array[3]
            }
            
            # Evaluate objectives
            objectives = self.evaluate_solution(policy_params, n_episodes=1)
            
            # Weighted sum
            weighted_objective = -np.dot(weights, objectives)  # Minimize negative value means maximize
            return weighted_objective
        
        # Parameter bounds
        bounds = [
            (0.5, 1.5),    # base_service
            (0.0, 0.5),    # adaptation
            (0.8, 1.2),    # arrival_factor
            (0.6, 0.9)     # transfer_threshold
        ]
        
        # Optimization
        result = differential_evolution(objective, bounds, maxiter=50, seed=42)
        
        # Convert optimal parameters
        optimal_params = {
            'base_service': result.x[0],
            'adaptation': result.x[1],
            'arrival_factor': result.x[2],
            'transfer_threshold': result.x[3]
        }
        
        return optimal_params
    
    def generate_pareto_front_systematically(self, n_weights: int = 20) -> None:
        """Systematically generate Pareto front"""
        print(f"Systematically generating Pareto front with {n_weights} weight vectors...")
        
        # Generate weight vectors (simplified: only consider first 3 main objectives)
        weight_vectors = []
        for i in range(n_weights):
            for j in range(n_weights - i):
                k = n_weights - i - j
                if k >= 0:
                    w1, w2, w3 = i/n_weights, j/n_weights, k/n_weights
                    if w1 + w2 + w3 > 0.99:  # Normalization check
                        weights = np.array([w1, w2, w3, 0.1, 0.1, 0.1])
                        weights = weights / np.sum(weights)
                        weight_vectors.append(weights)
        
        # Solve optimal solution for each weight vector
        systematic_solutions = []
        systematic_objectives = []
        
        for i, weights in enumerate(weight_vectors[:50]):  # Limit quantity
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(weight_vectors[:50])}")
            
            optimal_params = self.weight_optimization(weights)
            objectives = self.evaluate_solution(optimal_params, n_episodes=2)
            
            systematic_solutions.append(optimal_params)
            systematic_objectives.append(objectives)
        
        # Merge into existing solution set
        self.solutions.extend(systematic_solutions)
        if len(self.objective_values) > 0:
            self.objective_values = np.vstack([self.objective_values, np.array(systematic_objectives)])
        else:
            self.objective_values = np.array(systematic_objectives)
        
        print(f"Added {len(systematic_solutions)} systematic solutions")
    
    def plot_pareto_analysis(self, save_path: str = "./pareto_analysis.png") -> None:
        """Plot Pareto analysis results"""
        if len(self.pareto_front) == 0:
            self.find_pareto_front()
        
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Objective space scatter plot matrix
        n_plots = self.n_objectives
        for i in range(n_plots):
            for j in range(n_plots):
                ax = plt.subplot(n_plots, n_plots, i * n_plots + j + 1)
                
                if i == j:
                    # Diagonal: histogram
                    ax.hist(self.objective_values[:, i], bins=20, alpha=0.7, color='lightblue')
                    ax.hist(self.pareto_front[:, i], bins=10, alpha=0.8, color='red')
                    ax.set_title(f'{self.objective_names[i]}')
                else:
                    # Off-diagonal: scatter plot
                    ax.scatter(self.objective_values[:, j], self.objective_values[:, i], 
                             alpha=0.5, s=10, color='lightblue', label='All solutions')
                    ax.scatter(self.pareto_front[:, j], self.pareto_front[:, i], 
                             alpha=0.8, s=30, color='red', label='Pareto front')
                    
                    if i == 0 and j == 1:
                        ax.legend()
                
                if i == n_plots - 1:
                    ax.set_xlabel(self.objective_names[j])
                if j == 0:
                    ax.set_ylabel(self.objective_names[i])
        
        plt.suptitle('Pareto Analysis: Objective Space Visualization', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Objective conflict heatmap
        conflicts = self.analyze_objective_conflicts()
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Create correlation coefficient matrix
        corr_matrix = np.corrcoef(self.pareto_front.T)
        
        # Plot heatmap
        sns.heatmap(-corr_matrix, annot=True, cmap='RdYlBu', center=0,
                   xticklabels=self.objective_names, 
                   yticklabels=self.objective_names,
                   ax=ax)
        ax.set_title('Objective Conflicts Matrix (Red = Conflict)')
        
        plt.tight_layout()
        plt.savefig(save_path.replace('.png', '_conflicts.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Pareto analysis plots saved")
        
        # 3. 3DPareto front可视化（前3个目标）
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # All solutions
        ax.scatter(self.objective_values[:, 0], self.objective_values[:, 1], self.objective_values[:, 2],
                  alpha=0.3, s=10, color='lightblue', label='All solutions')
        
        # Pareto front
        ax.scatter(self.pareto_front[:, 0], self.pareto_front[:, 1], self.pareto_front[:, 2],
                  alpha=0.8, s=50, color='red', label='Pareto front')
        
        # Knee points
        knee_indices = self.find_knee_points()
        if knee_indices:
            knee_objectives = self.objective_values[knee_indices]
            ax.scatter(knee_objectives[:, 0], knee_objectives[:, 1], knee_objectives[:, 2],
                      s=100, color='gold', marker='*', label='Knee points')
        
        ax.set_xlabel(self.objective_names[0])
        ax.set_ylabel(self.objective_names[1])
        ax.set_zlabel(self.objective_names[2])
        ax.set_title('3D Pareto Front Visualization')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(save_path.replace('.png', '_3d.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, save_path: str = "./pareto_report.txt") -> None:
        """Generate Pareto analysis report"""
        if len(self.pareto_front) == 0:
            self.find_pareto_front()
        
        conflicts = self.analyze_objective_conflicts()
        knee_indices = self.find_knee_points()
        hypervolume = self.compute_hypervolume()
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("垂直分层队列系统帕累托最优解集分析报告\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"Total solutions: {len(self.objective_values)}\n")
            f.write(f"Pareto optimal solutions: {len(self.pareto_front)}\n")
            f.write(f"Pareto ratio: {len(self.pareto_front)/len(self.objective_values)*100:.2f}%\n")
            f.write(f"Hypervolume indicator: {hypervolume:.4f}\n")
            f.write(f"Knee points数量: {len(knee_indices)}\n\n")
            
            f.write("Objective statistics:\n")
            f.write("-"*40 + "\n")
            for i, name in enumerate(self.objective_names):
                all_values = self.objective_values[:, i]
                pareto_values = self.pareto_front[:, i]
                
                f.write(f"{name}:\n")
                f.write(f"  All solutions: {np.mean(all_values):.3f} ± {np.std(all_values):.3f}\n")
                f.write(f"  Pareto solutions: {np.mean(pareto_values):.3f} ± {np.std(pareto_values):.3f}\n")
                f.write(f"  Range: [{np.min(pareto_values):.3f}, {np.max(pareto_values):.3f}]\n\n")
            
            f.write("Main objective conflicts:\n")
            f.write("-"*40 + "\n")
            for conflict_pair, strength in sorted(conflicts.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{conflict_pair}: {strength:.3f}\n")
            
            if knee_indices:
                f.write(f"\nKnee points解详情:\n")
                f.write("-"*40 + "\n")
                for i, idx in enumerate(knee_indices):
                    f.write(f"Knee points {i+1}:\n")
                    for j, name in enumerate(self.objective_names):
                        f.write(f"  {name}: {self.objective_values[idx, j]:.3f}\n")
                    f.write("\n")
        
        print(f"Report saved to: {save_path}")


def main():
    """Main function: Run complete Pareto analysis"""
    print("Starting Pareto Optimal Set Analysis for Vertical Stratified Queuing System")
    
    # Create environment
    env = DRLOptimizedQueueEnvFixed()
    
    # Create analyzer
    analyzer = ParetoAnalyzer(env)
    
    # Generate solution set
    print("\n1. Generating random solutions...")
    analyzer.generate_random_solutions(n_solutions=500)
    
    # Systematically generate Pareto front
    print("\n2. Generating systematic Pareto front...")
    analyzer.generate_pareto_front_systematically(n_weights=10)
    
    # Identify Pareto front
    print("\n3. Finding Pareto front...")
    analyzer.find_pareto_front()
    
    # Analyze objective conflicts
    print("\n4. Analyzing objective conflicts...")
    conflicts = analyzer.analyze_objective_conflicts()
    print("Main conflicts:")
    for conflict, strength in sorted(conflicts.items(), key=lambda x: x[1], reverse=True)[:3]:
        print(f"  {conflict}: {strength:.3f}")
    
    # Generate visualizations
    print("\n5. Generating visualizations...")
    analyzer.plot_pareto_analysis("./pareto_analysis_complete.png")
    
    # Generate report
    print("\n6. Generating report...")
    analyzer.generate_report("./pareto_analysis_report.txt")
    
    print("\nPareto analysis completed!")
    print("Check the following files:")
    print("  - pareto_analysis_complete.png: Objective space visualization")
    print("  - pareto_analysis_complete_conflicts.png: Conflict matrix")
    print("  - pareto_analysis_complete_3d.png: 3D Pareto front")
    print("  - pareto_analysis_report.txt: Detailed analysis report")


if __name__ == "__main__":
    main()