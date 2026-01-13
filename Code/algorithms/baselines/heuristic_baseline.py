"""
Heuristic Baseline
启发式基线算法 - 基于队列理论和系统专业知识的策略
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import time

from .base_baseline import BaseBaseline


class HeuristicBaseline(BaseBaseline):
    """启发式基线算法实现"""
    
    def __init__(self, 
                 env,
                 algorithm_name: str = "Heuristic",
                 config: Optional[Dict] = None):
        
        default_config = {
            # 启发式策略参数
            'load_balance_threshold': 0.8,  # 负载均衡阈值
            'utilization_target': 0.7,     # 目标利用率
            'emergency_threshold': 0.9,     # 紧急传输阈值
            'service_rate_bounds': (0.1, 2.0),
            'arrival_rate_bounds': (0.5, 5.0),
            'adaptive_factor': 0.1,         # 自适应调整因子
            'priority_weights': [1.0, 0.8, 0.6, 0.4, 0.2]  # 层级优先级权重
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(env, algorithm_name, default_config)
        
        # 记录历史状态用于自适应调整
        self.history_window = 10
        self.utilization_history = []
        self.reward_history = []
        
        print(f"Heuristic Baseline initialized")
        print(f"Load balance threshold: {self.config['load_balance_threshold']}")
        print(f"Target utilization: {self.config['utilization_target']}")
    
    def _extract_state_info(self, observation) -> Dict:
        """从观察中提取状态信息"""
        if isinstance(observation, dict):
            # 从Dict观测空间提取信息
            extracted = {}
            extracted['queue_lengths'] = observation.get('queue_lengths', np.zeros(5))
            extracted['utilization_rates'] = observation.get('utilization_rates', np.zeros(5))
            extracted['service_rates'] = observation.get('service_rates', np.ones(5))
            extracted['load_rates'] = observation.get('load_rates', np.ones(5))
            
            # 推算到达率 (从负载率和服务率)
            load_rates = extracted['load_rates']
            service_rates = extracted['service_rates']
            arrival_rates = load_rates * service_rates
            extracted['arrival_rates'] = arrival_rates
            
            # 系统指标
            system_metrics = observation.get('system_metrics', np.zeros(3))
            extracted['throughput'] = system_metrics[0] if len(system_metrics) > 0 else 0
            extracted['system_load'] = system_metrics[1] if len(system_metrics) > 1 else np.mean(extracted['utilization_rates'])
            extracted['emergency_flags'] = [0, 0]  # 简化
            extracted['waiting_times'] = np.zeros(5)  # 简化
            
            return extracted
            
        elif isinstance(observation, (list, np.ndarray)):
            # 处理扁平化观测
            obs_array = np.array(observation).flatten()
            
            # 基于29维观察空间的结构
            if len(obs_array) >= 29:
                return {
                    'queue_lengths': obs_array[0:5],
                    'utilization_rates': obs_array[5:10],
                    'load_rates': obs_array[10:15],
                    'service_rates': obs_array[15:20],
                    'arrival_rates': obs_array[10:15] * obs_array[15:20],  # load_rates * service_rates
                    'waiting_times': np.zeros(5),
                    'throughput': obs_array[25] if len(obs_array) > 25 else 0,
                    'system_load': obs_array[26] if len(obs_array) > 26 else np.mean(obs_array[5:10]),
                    'emergency_flags': [0, 0]
                }
            else:
                # 简化处理
                n_layers = 5
                return {
                    'queue_lengths': obs_array[0:n_layers] if len(obs_array) >= n_layers else np.zeros(n_layers),
                    'utilization_rates': obs_array[0:n_layers] * 0.5 if len(obs_array) >= n_layers else np.ones(n_layers) * 0.5,
                    'arrival_rates': np.ones(n_layers),
                    'service_rates': np.ones(n_layers),
                    'load_rates': np.ones(n_layers),
                    'waiting_times': np.zeros(n_layers),
                    'throughput': 0,
                    'system_load': np.mean(obs_array),
                    'emergency_flags': [0, 0]
                }
        else:
            # 默认状态
            return {
                'queue_lengths': np.zeros(5),
                'utilization_rates': np.zeros(5),
                'arrival_rates': np.ones(5),
                'service_rates': np.ones(5),
                'load_rates': np.ones(5),
                'waiting_times': np.zeros(5),
                'throughput': 0,
                'system_load': 0,
                'emergency_flags': [0, 0]
            }
    
    def predict(self, observation, deterministic: bool = True) -> Tuple[np.ndarray, Optional[Dict]]:
        """基于启发式规则预测动作"""
        state_info = self._extract_state_info(observation)
        
        # 提取关键状态信息
        queue_lengths = np.array(state_info['queue_lengths'])
        utilization_rates = np.array(state_info['utilization_rates'])
        arrival_rates = np.array(state_info['arrival_rates'])
        service_rates = np.array(state_info['service_rates'])
        system_load = state_info['system_load']
        
        # 启发式决策
        action = self._make_heuristic_decision(
            queue_lengths, utilization_rates, arrival_rates, 
            service_rates, system_load
        )
        
        return action, {'strategy': 'heuristic'}
    
    def _make_heuristic_decision(self, queue_lengths, utilization_rates, 
                               arrival_rates, service_rates, system_load) -> Dict:
        """基于启发式规则做出决策"""
        
        # 1. 服务强度调整策略
        service_intensities = self._adjust_service_intensities(
            queue_lengths, utilization_rates, service_rates
        )
        
        # 2. 到达率调整策略
        arrival_multiplier = self._adjust_arrival_rate(
            utilization_rates, system_load
        )
        
        # 3. 紧急传输策略
        emergency_transfers = self._decide_emergency_transfers(
            queue_lengths, utilization_rates
        )
        
        return {
            'service_intensities': service_intensities.astype(np.float32),
            'arrival_multiplier': np.array([arrival_multiplier], dtype=np.float32),
            'emergency_transfers': emergency_transfers.astype(np.int8)
        }
    
    def _adjust_service_intensities(self, queue_lengths, utilization_rates, service_rates):
        """调整服务强度"""
        service_intensities = np.ones(5)
        
        target_util = self.config['utilization_target']
        priority_weights = self.config['priority_weights']
        
        for i in range(5):
            current_util = utilization_rates[i]
            queue_len = queue_lengths[i]
            priority = priority_weights[i]
            
            # 基于利用率和队列长度调整
            if current_util > target_util or queue_len > 10:
                # 增加服务强度
                adjustment = min(1.5, 1 + (current_util - target_util) * 2 + queue_len * 0.1)
            elif current_util < target_util * 0.5:
                # 减少服务强度以节省资源
                adjustment = max(0.5, 1 - (target_util - current_util) * 1.5)
            else:
                adjustment = 1.0
            
            # 考虑层级优先级
            adjustment *= (0.8 + 0.4 * priority)
            
            # 应用边界约束
            service_intensities[i] = np.clip(
                adjustment, 
                self.config['service_rate_bounds'][0], 
                self.config['service_rate_bounds'][1]
            )
        
        return service_intensities
    
    def _adjust_arrival_rate(self, utilization_rates, system_load):
        """调整系统到达率"""
        avg_utilization = np.mean(utilization_rates)
        target_util = self.config['utilization_target']
        
        if avg_utilization > self.config['load_balance_threshold']:
            # 系统过载，减少到达率
            multiplier = max(0.5, 1.0 - (avg_utilization - target_util) * 2)
        elif avg_utilization < target_util * 0.6:
            # 系统负载低，可以增加到达率
            multiplier = min(3.0, 1.0 + (target_util - avg_utilization) * 1.5)
        else:
            # 正常范围
            multiplier = 1.0
        
        # 考虑系统总体负载
        if system_load > 0.8:
            multiplier *= 0.8
        elif system_load < 0.3:
            multiplier *= 1.2
        
        return np.clip(
            multiplier,
            self.config['arrival_rate_bounds'][0],
            self.config['arrival_rate_bounds'][1]
        )
    
    def _decide_emergency_transfers(self, queue_lengths, utilization_rates):
        """决定紧急传输"""
        emergency_transfers = np.zeros(5, dtype=int)
        emergency_threshold = self.config['emergency_threshold']
        
        for i in range(4):  # 只有前4层可以向下传输
            current_util = utilization_rates[i]
            current_queue = queue_lengths[i]
            next_util = utilization_rates[i + 1]
            
            # 紧急传输条件
            should_transfer = (
                current_util > emergency_threshold or  # 当前层过载
                current_queue > 15 or                  # 队列过长
                (current_util > 0.8 and next_util < 0.5)  # 当前高负载且下层有余量
            )
            
            if should_transfer:
                emergency_transfers[i] = 1
        
        return emergency_transfers
    
    def train(self, total_timesteps: int, **kwargs) -> Dict:
        """训练过程（启发式策略可以通过收集数据进行自适应）"""
        print(f"Running Heuristic Baseline for {total_timesteps} timesteps...")
        
        # 重置训练记录
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'avg_rewards': [],
            'training_time': [],
            'loss_values': []
        }
        
        start_time = time.time()
        
        state, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        episode_count = 0
        
        for timestep in range(total_timesteps):
            # 预测动作
            action, _ = self.predict(state, deterministic=True)
            
            # 执行动作
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # 记录状态用于自适应
            state_info = self._extract_state_info(state)
            self._update_adaptive_parameters(state_info, reward)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            if done:
                # 记录episode信息
                self.training_history['episode_rewards'].append(episode_reward)
                self.training_history['episode_lengths'].append(episode_length)
                
                # 计算平均奖励
                if len(self.training_history['episode_rewards']) >= 100:
                    avg_reward = np.mean(self.training_history['episode_rewards'][-100:])
                    self.training_history['avg_rewards'].append(avg_reward)
                
                if episode_count % 100 == 0:
                    print(f"Episode {episode_count}, Timestep {timestep}, Reward: {episode_reward:.2f}")
                
                # 重置环境
                state, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0
                episode_count += 1
        
        end_time = time.time()
        training_time = end_time - start_time
        self.training_history['training_time'].append(training_time)
        
        self.total_timesteps = total_timesteps
        self.episode_count = episode_count
        
        print(f"Heuristic Baseline completed in {training_time:.2f} seconds")
        
        return {
            'total_timesteps': total_timesteps,
            'episodes': episode_count,
            'training_time': training_time,
            'final_reward': self.training_history['episode_rewards'][-1] if self.training_history['episode_rewards'] else 0
        }
    
    def _update_adaptive_parameters(self, state_info, reward):
        """根据历史表现自适应调整参数"""
        utilization_rates = state_info['utilization_rates']
        avg_utilization = np.mean(utilization_rates)
        
        # 更新历史记录
        self.utilization_history.append(avg_utilization)
        self.reward_history.append(reward)
        
        # 保持历史窗口大小
        if len(self.utilization_history) > self.history_window:
            self.utilization_history.pop(0)
            self.reward_history.pop(0)
        
        # 自适应调整（简单版本）
        if len(self.reward_history) >= self.history_window:
            recent_reward = np.mean(self.reward_history[-5:])
            early_reward = np.mean(self.reward_history[:5])
            
            if recent_reward < early_reward:
                # 性能下降，调整阈值
                self.config['load_balance_threshold'] *= (1 - self.config['adaptive_factor'])
                self.config['utilization_target'] *= (1 - self.config['adaptive_factor'] * 0.5)
            else:
                # 性能改善，保持或微调
                self.config['load_balance_threshold'] *= (1 + self.config['adaptive_factor'] * 0.5)
                self.config['utilization_target'] *= (1 + self.config['adaptive_factor'] * 0.3)
        
        # 边界约束
        self.config['load_balance_threshold'] = np.clip(self.config['load_balance_threshold'], 0.6, 0.95)
        self.config['utilization_target'] = np.clip(self.config['utilization_target'], 0.5, 0.8)
    
    def save(self, path: str) -> None:
        """保存模型"""
        import json
        
        # 转换numpy类型为Python原生类型
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        save_data = {
            'algorithm_name': self.algorithm_name,
            'config': convert_numpy(self.config),
            'total_timesteps': int(self.total_timesteps),
            'episode_count': int(self.episode_count),
            'training_history': convert_numpy(self.training_history),
            'utilization_history': [float(x) for x in self.utilization_history],
            'reward_history': [float(x) for x in self.reward_history]
        }
        
        with open(path, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"Heuristic Baseline saved to: {path}")
    
    def load(self, path: str) -> None:
        """加载模型"""
        import json
        
        with open(path, 'r') as f:
            save_data = json.load(f)
        
        self.config.update(save_data['config'])
        self.total_timesteps = save_data.get('total_timesteps', 0)
        self.episode_count = save_data.get('episode_count', 0)
        self.training_history = save_data.get('training_history', {
            'episode_rewards': [],
            'episode_lengths': [],
            'avg_rewards': [],
            'training_time': [],
            'loss_values': []
        })
        self.utilization_history = save_data.get('utilization_history', [])
        self.reward_history = save_data.get('reward_history', [])
        
        print(f"Heuristic Baseline loaded from: {path}")
    
    def get_info(self) -> Dict:
        """获取算法信息"""
        info = super().get_info()
        info.update({
            'description': 'Heuristic policy based on queuing theory',
            'deterministic': True,
            'adaptive': True,
            'current_thresholds': {
                'load_balance': self.config['load_balance_threshold'],
                'utilization_target': self.config['utilization_target']
            }
        })
        return info