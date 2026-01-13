"""
工具函数模块
Utility Functions Module

提供数学计算、统计分析、性能评估等辅助功能：
- 数学工具：向量归一化、统计计算、概率分布
- 性能分析：稳定性检查、负载计算、效率评估
- 可视化工具：状态绘制、性能曲线、实验结果展示
- 数据处理：日志记录、结果保存、配置管理
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
    数学工具类
    
    提供各种数学计算和统计分析功能
    """
    
    @staticmethod
    def normalize_vector(vector: np.ndarray, method: str = 'sum') -> np.ndarray:
        """
        向量归一化
        
        Args:
            vector: 输入向量
            method: 归一化方法 ('sum', 'max', 'minmax', 'zscore')
        """
        vector = np.array(vector, dtype=np.float32)
        
        if method == 'sum':
            # 和归一化 (概率分布)
            total = np.sum(vector)
            return vector / total if total > 0 else vector
            
        elif method == 'max':
            # 最大值归一化
            max_val = np.max(vector)
            return vector / max_val if max_val > 0 else vector
            
        elif method == 'minmax':
            # 最小-最大归一化到[0,1]
            min_val, max_val = np.min(vector), np.max(vector)
            if max_val > min_val:
                return (vector - min_val) / (max_val - min_val)
            else:
                return np.zeros_like(vector)
                
        elif method == 'zscore':
            # Z-score标准化
            mean_val, std_val = np.mean(vector), np.std(vector)
            if std_val > 0:
                return (vector - mean_val) / std_val
            else:
                return vector - mean_val
                
        else:
            raise ValueError(f"未知的归一化方法: {method}")
    
    @staticmethod
    def calculate_entropy(probabilities: np.ndarray) -> float:
        """
        计算概率分布的熵
        """
        probs = np.array(probabilities)
        probs = probs[probs > 0]  # 移除零概率
        return -np.sum(probs * np.log2(probs))
    
    @staticmethod
    def calculate_gini_coefficient(values: np.ndarray) -> float:
        """
        计算基尼系数 (不平等度量)
        """
        values = np.sort(np.array(values))
        n = len(values)
        cumsum = np.cumsum(values)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0
    
    @staticmethod
    def moving_average(data: List[float], window: int) -> List[float]:
        """
        计算移动平均
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
        计算指数移动平均
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
        简单线性回归
        
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
        
        # 计算R²
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return slope, intercept, r_squared


class PerformanceAnalyzer:
    """
    性能分析器
    
    用于分析系统性能、验证理论假设
    """
    
    def __init__(self):
        self.math_utils = MathUtils()
    
    def analyze_stability_condition(self, arrival_rates: List[float], 
                                   service_rates: List[float], 
                                   capacities: List[int]) -> Dict:
        """
        分析稳定性条件
        
        验证 ρᵢ = λᵢᵉᶠᶠ/(μᵢ·cᵢ) < 1
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
        计算效率指标
        """
        if not throughput_data:
            return {}
        
        # 吞吐量效率
        avg_throughput = np.mean(throughput_data)
        throughput_efficiency = avg_throughput / theoretical_max if theoretical_max > 0 else 0
        
        # 等待时间分析
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
            'efficiency_score': throughput_efficiency * wait_time_fairness  # 综合效率分数
        }
    
    def analyze_layer_performance(self, queue_lengths_history: List[List[int]],
                                 capacities: List[int]) -> Dict:
        """
        分析各层性能
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
        计算倒金字塔结构效率
        
        验证H1: 倒金字塔容量分配是最优结构
        """
        # 检查是否为倒金字塔结构
        is_pyramid = all(capacities[i] >= capacities[i+1] for i in range(len(capacities)-1))
        
        # 计算容量利用效率
        weighted_utilization = sum(u * c for u, c in zip(utilizations, capacities)) / sum(capacities)
        
        # 计算负载平衡度
        load_balance = 1 - self.math_utils.calculate_gini_coefficient(utilizations)
        
        # 计算容量梯度合理性
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
    可视化工具类
    
    用于绘制性能图表、实验结果展示
    """
    
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        self.colors = plt.cm.Set3(np.linspace(0, 1, 12))
    
    def plot_queue_evolution(self, queue_history: List[List[int]], 
                           capacities: List[int],
                           save_path: Optional[str] = None):
        """
        绘制队列长度演化图
        """
        if not queue_history:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        num_layers = len(capacities)
        steps = range(len(queue_history))
        
        for i in range(min(num_layers, 6)):  # 最多显示6层
            ax = axes[i]
            
            # 提取该层的历史数据
            layer_data = [step_data[i] if i < len(step_data) else 0 
                         for step_data in queue_history]
            
            # 绘制队列长度
            ax.plot(steps, layer_data, label=f'L{i+1}队列长度', 
                   color=self.colors[i], linewidth=2)
            
            # 绘制容量线
            ax.axhline(y=capacities[i], color='red', linestyle='--', 
                      label=f'容量={capacities[i]}')
            
            ax.set_title(f'第{i+1}层队列演化 (H={100-i*20}m)')
            ax.set_xlabel('时间步')
            ax.set_ylabel('队列长度')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(num_layers, 6):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.suptitle('垂直分层队列演化分析', fontsize=16, y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_performance_metrics(self, metrics_history: Dict[str, List],
                               save_path: Optional[str] = None):
        """
        绘制性能指标图
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 吞吐量
        if 'throughput' in metrics_history:
            axes[0, 0].plot(metrics_history['throughput'], color='blue', linewidth=2)
            axes[0, 0].set_title('系统吞吐量')
            axes[0, 0].set_ylabel('吞吐量 (订单/步)')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 平均等待时间
        if 'avg_waiting_time' in metrics_history:
            axes[0, 1].plot(metrics_history['avg_waiting_time'], color='green', linewidth=2)
            axes[0, 1].set_title('平均等待时间')
            axes[0, 1].set_ylabel('等待时间 (步)')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 系统利用率
        if 'system_utilization' in metrics_history:
            axes[1, 0].plot(metrics_history['system_utilization'], color='orange', linewidth=2)
            axes[1, 0].set_title('系统利用率')
            axes[1, 0].set_ylabel('利用率')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 稳定性指标
        if 'max_load_factor' in metrics_history:
            axes[1, 1].plot(metrics_history['max_load_factor'], color='red', linewidth=2)
            axes[1, 1].axhline(y=1.0, color='black', linestyle='--', label='稳定性阈值')
            axes[1, 1].set_title('最大负载系数')
            axes[1, 1].set_ylabel('负载系数 ρ')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        for ax in axes.flat:
            ax.set_xlabel('时间步')
        
        plt.tight_layout()
        plt.suptitle('系统性能指标监控', fontsize=16, y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_hypothesis_comparison(self, results: Dict[str, Dict],
                                 hypothesis: str,
                                 save_path: Optional[str] = None):
        """
        绘制假设验证对比图
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        configurations = list(results.keys())
        metrics = ['throughput', 'avg_waiting_time', 'efficiency_score']
        metric_names = ['吞吐量', '平均等待时间', '效率分数']
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            values = [results[config].get(metric, 0) for config in configurations]
            
            bars = axes[i].bar(configurations, values, color=self.colors[:len(configurations)])
            axes[i].set_title(f'{name}对比')
            axes[i].set_ylabel(name)
            axes[i].tick_params(axis='x', rotation=45)
            
            # 添加数值标签
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.suptitle(f'{hypothesis}假设验证结果对比', fontsize=16, y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


class DataLogger:
    """
    数据记录器
    
    用于记录实验数据、保存结果、生成报告
    """
    
    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.experiment_data = defaultdict(list)
        self.metadata = {}
        
    def log_step_data(self, step: int, data: Dict):
        """
        记录单步数据
        """
        data['step'] = step
        data['timestamp'] = time.time()
        
        for key, value in data.items():
            self.experiment_data[key].append(value)
    
    def log_episode_summary(self, episode: int, summary: Dict):
        """
        记录回合总结
        """
        summary['episode'] = episode
        summary['timestamp'] = time.time()
        
        # 保存回合总结
        summary_file = self.log_dir / f"episode_{episode}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    
    def save_experiment_data(self, experiment_name: str):
        """
        保存完整实验数据
        """
        # 保存原始数据
        data_file = self.log_dir / f"{experiment_name}_data.pkl"
        with open(data_file, 'wb') as f:
            pickle.dump(dict(self.experiment_data), f)
        
        # 保存元数据
        metadata_file = self.log_dir / f"{experiment_name}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        
        print(f"实验数据已保存到: {self.log_dir}")
    
    def load_experiment_data(self, experiment_name: str) -> Dict:
        """
        加载实验数据
        """
        data_file = self.log_dir / f"{experiment_name}_data.pkl"
        
        if data_file.exists():
            with open(data_file, 'rb') as f:
                return pickle.load(f)
        else:
            raise FileNotFoundError(f"未找到实验数据: {data_file}")
    
    def generate_report(self, experiment_name: str) -> str:
        """
        生成实验报告
        """
        if not self.experiment_data:
            return "无数据可生成报告"
        
        # 计算基本统计
        stats = {}
        for key, values in self.experiment_data.items():
            if isinstance(values[0], (int, float)):
                stats[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        # 生成报告
        report = f"""
# {experiment_name} 实验报告

## 实验概况
- 总步数: {len(self.experiment_data.get('step', []))}
- 开始时间: {time.ctime(min(self.experiment_data.get('timestamp', [time.time()])))}
- 结束时间: {time.ctime(max(self.experiment_data.get('timestamp', [time.time()])))}

## 性能统计
"""
        
        for key, stat in stats.items():
            report += f"""
### {key}
- 平均值: {stat['mean']:.4f}
- 标准差: {stat['std']:.4f}  
- 最小值: {stat['min']:.4f}
- 最大值: {stat['max']:.4f}
"""
        
        # 保存报告
        report_file = self.log_dir / f"{experiment_name}_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        return report
    
    def set_metadata(self, **kwargs):
        """
        设置实验元数据
        """
        self.metadata.update(kwargs)


class ExperimentUtils:
    """
    实验工具类
    
    用于实验设计、假设验证、结果分析
    """
    
    def __init__(self):
        self.performance_analyzer = PerformanceAnalyzer()
        self.visualizer = VisualizationUtils()
        self.logger = DataLogger()
    
    def run_hypothesis_test(self, env_class, configurations: Dict, 
                          num_episodes: int = 100) -> Dict:
        """
        运行假设测试
        
        Args:
            env_class: 环境类
            configurations: 配置字典 {name: config}
            num_episodes: 测试回合数
        """
        results = {}
        
        for config_name, config in configurations.items():
            print(f"测试配置: {config_name}")
            
            # 创建环境
            env = env_class(config)
            
            # 运行实验
            episode_results = []
            for episode in range(num_episodes):
                obs, info = env.reset()
                episode_data = []
                
                for step in range(config.max_episode_steps):
                    # 随机动作 (可以替换为训练好的策略)
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
            
            # 分析结果
            config_analysis = self._analyze_configuration_results(episode_results)
            results[config_name] = config_analysis
            
            env.close()
        
        return results
    
    def _analyze_configuration_results(self, episode_results: List) -> Dict:
        """
        分析配置结果
        """
        # 聚合所有回合的数据
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
        
        # 计算统计指标
        return {
            'throughput': np.mean(all_throughputs) if all_throughputs else 0,
            'avg_waiting_time': np.mean([np.mean(wt) for wt in all_waiting_times if wt]) if all_waiting_times else 0,
            'max_queue_length': np.max([np.max(ql) for ql in all_queue_lengths if ql]) if all_queue_lengths else 0,
            'efficiency_score': np.mean(all_throughputs) * 0.7 + (1 / (1 + np.mean([np.mean(wt) for wt in all_waiting_times if wt]))) * 0.3 if all_throughputs and all_waiting_times else 0
        }
    
    def validate_theoretical_predictions(self, actual_results: Dict, 
                                       theoretical_bounds: Dict) -> Dict:
        """
        验证理论预测
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


# 测试工具函数
if __name__ == "__main__":
    print("测试工具函数...")
    
    # 测试数学工具
    math_utils = MathUtils()
    vector = np.array([1, 2, 3, 4, 5])
    normalized = math_utils.normalize_vector(vector, 'sum')
    print(f"向量归一化: {vector} -> {normalized}")
    
    # 测试性能分析
    analyzer = PerformanceAnalyzer()
    stability = analyzer.analyze_stability_condition([0.2, 0.3, 0.4], [1.0, 1.1, 1.2], [5, 4, 3])
    print(f"稳定性分析: {stability}")
    
    # 测试数据记录
    logger = DataLogger("./test_logs")
    logger.log_step_data(1, {'throughput': 0.8, 'waiting_time': 15})
    logger.log_step_data(2, {'throughput': 0.9, 'waiting_time': 12})
    
    report = logger.generate_report("test_experiment")
    print("生成报告成功")
    
    print("✅ 工具函数测试完成!")