"""
Utility Functions Module

Provides mathematical calculations, statistical analysis, performance evaluation and other auxiliary functions:
- Mathematical tools: Vector normalization, statistical calculations, probability distributions
- Performance analysis: Stability checks, load calculations, efficiency evaluation
- Visualization tools: State plotting, performance curves, experimental result display
- Data processing: Logging, result saving, configuration management
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
from collections import deque, defaultdict
import json
import pickle
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class MathUtils:
    """
    Mathematical Utilities Class
    
    Provides various mathematical calculations and statistical analysis functions
    """
    
    @staticmethod
    def normalize_vector(vector: np.ndarray, method: str = 'sum') -> np.ndarray:
        """
        Vector normalization
        
        Args:
            vector: Input vector
            method: Normalization method ('sum', 'max', 'minmax', 'zscore')
        """
        vector = np.array(vector, dtype=np.float32)
        
        if method == 'sum':
            # Sum normalization (probability distribution)
            total = np.sum(vector)
            return vector / total if total > 0 else vector
            
        elif method == 'max':
            # Max normalization
            max_val = np.max(vector)
            return vector / max_val if max_val > 0 else vector
            
        elif method == 'minmax':
            # Min-max normalization to [0,1]
            min_val, max_val = np.min(vector), np.max(vector)
            if max_val > min_val:
                return (vector - min_val) / (max_val - min_val)
            else:
                return np.zeros_like(vector)
                
        elif method == 'zscore':
            # Z-score standardization
            mean_val, std_val = np.mean(vector), np.std(vector)
            if std_val > 0:
                return (vector - mean_val) / std_val
            else:
                return vector - mean_val
                
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    @staticmethod
    def calculate_entropy(probabilities: np.ndarray) -> float:
        """
        Calculate entropy of probability distribution
        """
        probs = np.array(probabilities)
        probs = probs[probs > 0]  # Remove zero probabilities
        return -np.sum(probs * np.log2(probs))
    
    @staticmethod
    def calculate_gini_coefficient(values: np.ndarray) -> float:
        """
        Calculate Gini coefficient (inequality measure)
        """
        values = np.sort(np.array(values))
        n = len(values)
        cumsum = np.cumsum(values)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0
    
    @staticmethod
    def moving_average(data: List[float], window: int) -> List[float]:
        """
        Calculate moving average
        """
        if len(data) < window:
            return data
        
        result = []
        for i in range(window - 1, len(data)):
            avg = sum(data[i - window + 1:i + 1]) / window
            result.append(avg)
        
        return result
    
    @staticmethod
    def exponential_moving_average(data: List[float], alpha: float = 0.1) -> List[float]:
        """
        Calculate exponential moving average
        """
        if not data:
            return []
        
        ema = [data[0]]
        for i in range(1, len(data)):
            ema_val = alpha * data[i] + (1 - alpha) * ema[-1]
            ema.append(ema_val)
        
        return ema
    
    @staticmethod
    def linear_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
        """
        Simple linear regression
        
        Returns:
            slope, intercept, r_squared
        """
        n = len(x)
        if n < 2:
            return 0, 0, 0
        
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if denominator == 0:
            return 0, y_mean, 0
        
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        
        # Calculate R²
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return slope, intercept, r_squared


class PerformanceAnalyzer:
    """
    Performance Analyzer
    
    Used for analyzing system performance and validating theoretical hypotheses
    """
    
    def __init__(self):
        self.math_utils = MathUtils()
    
    def analyze_stability_condition(self, arrival_rates: List[float], 
                                   service_rates: List[float], 
                                   capacities: List[int]) -> Dict:
        """
        Analyze stability condition
        
        Verify ρᵢ = λᵢᵉᶠᶠ/(μᵢ·cᵢ) < 1
        """
        load_factors = []
        is_stable = True
        
        for i in range(len(arrival_rates)):
            service_capacity = service_rates[i] * capacities[i]
            rho = arrival_rates[i] / service_capacity if service_capacity > 0 else float('inf')
            load_factors.append(rho)
            
            if rho >= 1.0:
                is_stable = False
        
        return {
            'load_factors': load_factors,
            'is_stable': is_stable,
            'max_load_factor': max(load_factors),
            'avg_load_factor': np.mean(load_factors),
            'stability_margin': 1.0 - max(load_factors)
        }
    
    def calculate_efficiency_metrics(self, throughput_data: List[float],
                                   theoretical_max: float,
                                   waiting_times: List[List[float]]) -> Dict:
        """
        Calculate efficiency metrics
        """
        if not throughput_data:
            return {}
        
        # Throughput efficiency
        avg_throughput = np.mean(throughput_data)
        throughput_efficiency = avg_throughput / theoretical_max if theoretical_max > 0 else 0
        
        # Waiting time analysis
        if waiting_times:
            avg_waiting_times = [np.mean(layer_waits) for layer_waits in zip(*waiting_times)]
            overall_avg_wait = np.mean(avg_waiting_times)
            wait_time_variance = np.var(avg_waiting_times)
            wait_time_fairness = 1 - (max(avg_waiting_times) - min(avg_waiting_times)) / max(avg_waiting_times) if max(avg_waiting_times) > 0 else 1
        else:
            overall_avg_wait = 0
            wait_time_variance = 0
            wait_time_fairness = 1
        
        return {
            'throughput_efficiency': throughput_efficiency,
            'avg_throughput': avg_throughput,
            'avg_waiting_time': overall_avg_wait,
            'waiting_time_variance': wait_time_variance,
            'waiting_time_fairness': wait_time_fairness,
            'efficiency_score': throughput_efficiency * wait_time_fairness  # Composite efficiency score
        }
    
    def analyze_layer_performance(self, queue_lengths_history: List[List[int]],
                                 capacities: List[int]) -> Dict:
        """
        Analyze layer performance
        """
        if not queue_lengths_history:
            return {}
        
        num_layers = len(capacities)
        layer_analysis = {}
        
        for i in range(num_layers):
            layer_lengths = [step_data[i] for step_data in queue_lengths_history if i < len(step_data)]
            
            if layer_lengths:
                avg_length = np.mean(layer_lengths)
                utilization = avg_length / capacities[i] if capacities[i] > 0 else 0
                variance = np.var(layer_lengths)
                max_utilization = max(layer_lengths) / capacities[i] if capacities[i] > 0 else 0
                
                layer_analysis[f'layer_{i}'] = {
                    'avg_queue_length': avg_length,
                    'utilization': utilization,
                    'variance': variance,
                    'max_utilization': max_utilization,
                    'capacity': capacities[i]
                }
        
        return layer_analysis
    
    def calculate_pyramid_efficiency(self, capacities: List[int], 
                                   utilizations: List[float]) -> Dict:
        """
        Calculate inverted pyramid structure efficiency
        
        Verify H1: Inverted pyramid capacity allocation is optimal structure
        """
        # Check if it's an inverted pyramid structure
        is_pyramid = all(capacities[i] >= capacities[i+1] for i in range(len(capacities)-1))
        
        # Calculate capacity utilization efficiency
        weighted_utilization = sum(u * c for u, c in zip(utilizations, capacities)) / sum(capacities)
        
        # Calculate load balance
        load_balance = 1 - self.math_utils.calculate_gini_coefficient(utilizations)
        
        # Calculate capacity gradient rationality
        if len(capacities) > 1:
            capacity_ratios = [capacities[i] / capacities[i+1] for i in range(len(capacities)-1) 
                             if capacities[i+1] > 0]
            capacity_gradient = np.std(capacity_ratios) if capacity_ratios else 0
        else:
            capacity_gradient = 0
        
        return {
            'is_pyramid_structure': is_pyramid,
            'weighted_utilization': weighted_utilization,
            'load_balance': load_balance,
            'capacity_gradient': capacity_gradient,
            'pyramid_efficiency_score': weighted_utilization * load_balance * (1 / (1 + capacity_gradient))
        }


class VisualizationUtils:
    """
    Visualization Utilities Class
    
    Used for plotting performance charts and displaying experimental results
    """
    
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        self.colors = plt.cm.Set3(np.linspace(0, 1, 12))
    
    def plot_queue_evolution(self, queue_history: List[List[int]], 
                           capacities: List[int],
                           save_path: Optional[str] = None):
        """
        Plot queue length evolution
        """
        if not queue_history:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        num_layers = len(capacities)
        steps = range(len(queue_history))
        
        for i in range(min(num_layers, 6)):  # Display at most 6 layers
            ax = axes[i]
            
            # Extract historical data for this layer
            layer_data = [step_data[i] if i < len(step_data) else 0 
                         for step_data in queue_history]
            
            # Plot queue length
            ax.plot(steps, layer_data, label=f'L{i+1} Queue Length', 
                   color=self.colors[i], linewidth=2)
            
            # Plot capacity line
            ax.axhline(y=capacities[i], color='red', linestyle='--', 
                      label=f'Capacity={capacities[i]}')
            
            ax.set_title(f'Layer {i+1} Queue Evolution (H={100-i*20}m)')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Queue Length')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide extra subplots
        for i in range(num_layers, 6):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle('Vertical Layered Queue Evolution Analysis', fontsize=16, y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_performance_metrics(self, metrics_history: Dict[str, List],
                               save_path: Optional[str] = None):
        """
        Plot performance metrics
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Throughput
        if 'throughput' in metrics_history:
            axes[0, 0].plot(metrics_history['throughput'], color='blue', linewidth=2)
            axes[0, 0].set_title('System Throughput')
            axes[0, 0].set_ylabel('Throughput (orders/step)')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Average waiting time
        if 'avg_waiting_time' in metrics_history:
            axes[0, 1].plot(metrics_history['avg_waiting_time'], color='green', linewidth=2)
            axes[0, 1].set_title('Average Waiting Time')
            axes[0, 1].set_ylabel('Waiting Time (steps)')
            axes[0, 1].grid(True, alpha=0.3)
        
        # System utilization
        if 'system_utilization' in metrics_history:
            axes[1, 0].plot(metrics_history['system_utilization'], color='orange', linewidth=2)
            axes[1, 0].set_title('System Utilization')
            axes[1, 0].set_ylabel('Utilization')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Stability metric
        if 'max_load_factor' in metrics_history:
            axes[1, 1].plot(metrics_history['max_load_factor'], color='red', linewidth=2)
            axes[1, 1].axhline(y=1.0, color='black', linestyle='--', label='Stability Threshold')
            axes[1, 1].set_title('Maximum Load Factor')
            axes[1, 1].set_ylabel('Load Factor ρ')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        for ax in axes.flat:
            ax.set_xlabel('Time Step')
        
        plt.tight_layout()
        plt.suptitle('System Performance Metrics Monitoring', fontsize=16, y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_hypothesis_comparison(self, results: Dict[str, Dict],
                                 hypothesis: str,
                                 save_path: Optional[str] = None):
        """
        Plot hypothesis validation comparison
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        configurations = list(results.keys())
        metrics = ['throughput', 'avg_waiting_time', 'efficiency_score']
        metric_names = ['Throughput', 'Average Waiting Time', 'Efficiency Score']
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            values = [results[config].get(metric, 0) for config in configurations]
            
            bars = axes[i].bar(configurations, values, color=self.colors[:len(configurations)])
            axes[i].set_title(f'{name} Comparison')
            axes[i].set_ylabel(name)
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.suptitle(f'{hypothesis} Hypothesis Validation Results Comparison', fontsize=16, y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class DataLogger:
    """
    Data Logger
    
    Used for recording experimental data, saving results, generating reports
    """
    
    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.experiment_data = defaultdict(list)
        self.metadata = {}
        
    def log_step_data(self, step: int, data: Dict):
        """
        Log single step data
        """
        data['step'] = step
        data['timestamp'] = time.time()
        
        for key, value in data.items():
            self.experiment_data[key].append(value)
    
    def log_episode_summary(self, episode: int, summary: Dict):
        """
        Log episode summary
        """
        summary['episode'] = episode
        summary['timestamp'] = time.time()
        
        # Save episode summary
        summary_file = self.log_dir / f"episode_{episode}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    
    def save_experiment_data(self, experiment_name: str):
        """
        Save complete experimental data
        """
        # Save raw data
        data_file = self.log_dir / f"{experiment_name}_data.pkl"
        with open(data_file, 'wb') as f:
            pickle.dump(dict(self.experiment_data), f)
        
        # Save metadata
        metadata_file = self.log_dir / f"{experiment_name}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        
        print(f"Experimental data saved to: {self.log_dir}")
    
    def load_experiment_data(self, experiment_name: str) -> Dict:
        """
        Load experimental data
        """
        data_file = self.log_dir / f"{experiment_name}_data.pkl"
        
        if data_file.exists():
            with open(data_file, 'rb') as f:
                return pickle.load(f)
        else:
            raise FileNotFoundError(f"Experimental data not found: {data_file}")
    
    def generate_report(self, experiment_name: str) -> str:
        """
        Generate experimental report
        """
        if not self.experiment_data:
            return "No data available to generate report"
        
        # Calculate basic statistics
        stats = {}
        for key, values in self.experiment_data.items():
            if isinstance(values[0], (int, float)):
                stats[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        # Generate report
        report = f"""
# {experiment_name} Experimental Report

## Experiment Overview
- Total Steps: {len(self.experiment_data.get('step', []))}
- Start Time: {time.ctime(min(self.experiment_data.get('timestamp', [time.time()])))}
- End Time: {time.ctime(max(self.experiment_data.get('timestamp', [time.time()])))}

## Performance Statistics
"""
        
        for key, stat in stats.items():
            report += f"""
### {key}
- Mean: {stat['mean']:.4f}
- Std Dev: {stat['std']:.4f}  
- Min: {stat['min']:.4f}
- Max: {stat['max']:.4f}
"""
        
        # Save report
        report_file = self.log_dir / f"{experiment_name}_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        return report
    
    def set_metadata(self, **kwargs):
        """
        Set experimental metadata
        """
        self.metadata.update(kwargs)


class ExperimentUtils:
    """
    Experiment Utilities Class
    
    Used for experimental design, hypothesis validation, result analysis
    """
    
    def __init__(self):
        self.performance_analyzer = PerformanceAnalyzer()
        self.visualizer = VisualizationUtils()
        self.logger = DataLogger()
    
    def run_hypothesis_test(self, env_class, configurations: Dict, 
                          num_episodes: int = 100) -> Dict:
        """
        Run hypothesis test
        
        Args:
            env_class: Environment class
            configurations: Configuration dictionary {name: config}
            num_episodes: Number of test episodes
        """
        results = {}
        
        for config_name, config in configurations.items():
            print(f"Testing configuration: {config_name}")
            
            # Create environment
            env = env_class(config)
            
            # Run experiment
            episode_results = []
            for episode in range(num_episodes):
                obs, info = env.reset()
                episode_data = []
                
                for step in range(config.max_episode_steps):
                    # Random action (can be replaced with trained policy)
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    episode_data.append({
                        'step': step,
                        'reward': reward,
                        'queue_lengths': info.get('queue_lengths', []),
                        'throughput': info.get('throughput', 0),
                        'waiting_times': info.get('waiting_times', [])
                    })
                    
                    if terminated or truncated:
                        break
                
                episode_results.append(episode_data)
            
            # Analyze results
            config_analysis = self._analyze_configuration_results(episode_results)
            results[config_name] = config_analysis
            
            env.close()
        
        return results
    
    def _analyze_configuration_results(self, episode_results: List) -> Dict:
        """
        Analyze configuration results
        """
        # Aggregate data from all episodes
        all_throughputs = []
        all_waiting_times = []
        all_queue_lengths = []
        
        for episode_data in episode_results:
            episode_throughputs = [step['throughput'] for step in episode_data]
            episode_waiting_times = [step['waiting_times'] for step in episode_data]
            episode_queue_lengths = [step['queue_lengths'] for step in episode_data]
            
            all_throughputs.extend(episode_throughputs)
            all_waiting_times.extend(episode_waiting_times)
            all_queue_lengths.extend(episode_queue_lengths)
        
        # Calculate statistical metrics
        return {
            'throughput': np.mean(all_throughputs) if all_throughputs else 0,
            'avg_waiting_time': np.mean([np.mean(wt) for wt in all_waiting_times if wt]) if all_waiting_times else 0,
            'max_queue_length': np.max([np.max(ql) for ql in all_queue_lengths if ql]) if all_queue_lengths else 0,
            'efficiency_score': np.mean(all_throughputs) * 0.7 + (1 / (1 + np.mean([np.mean(wt) for wt in all_waiting_times if wt]))) * 0.3 if all_throughputs and all_waiting_times else 0
        }
    
    def validate_theoretical_predictions(self, actual_results: Dict, 
                                       theoretical_bounds: Dict) -> Dict:
        """
        Validate theoretical predictions
        """
        validation_results = {}
        
        for metric, bounds in theoretical_bounds.items():
            if metric in actual_results:
                actual_value = actual_results[metric]
                
                if 'min' in bounds and 'max' in bounds:
                    in_range = bounds['min'] <= actual_value <= bounds['max']
                    validation_results[metric] = {
                        'actual': actual_value,
                        'theoretical_range': (bounds['min'], bounds['max']),
                        'in_range': in_range,
                        'deviation': min(abs(actual_value - bounds['min']), 
                                       abs(actual_value - bounds['max'])) if not in_range else 0
                    }
        
        return validation_results


# Test utility functions
if __name__ == "__main__":
    print("Testing utility functions...")
    
    # Test mathematical tools
    math_utils = MathUtils()
    vector = np.array([1, 2, 3, 4, 5])
    normalized = math_utils.normalize_vector(vector, 'sum')
    print(f"Vector normalization: {vector} -> {normalized}")
    
    # Test performance analysis
    analyzer = PerformanceAnalyzer()
    stability = analyzer.analyze_stability_condition([0.2, 0.3, 0.4], [1.0, 1.1, 1.2], [5, 4, 3])
    print(f"Stability analysis: {stability}")
    
    # Test data logging
    logger = DataLogger("./test_logs")
    logger.log_step_data(1, {'throughput': 0.8, 'waiting_time': 15})
    logger.log_step_data(2, {'throughput': 0.9, 'waiting_time': 12})
    
    report = logger.generate_report("test_experiment")
    print("Report generation successful")
    
    print("✅ Utility functions test completed!")
