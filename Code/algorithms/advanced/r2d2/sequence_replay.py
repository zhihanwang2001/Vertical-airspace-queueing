"""
R2D2 Sequence Replay Buffer
R2D2算法的序列化经验回放缓冲区，支持重叠序列和burn-in
"""

import numpy as np
import torch
import random
from typing import List, Dict, Tuple, Any, Optional
from collections import deque
import copy


class SequenceBuffer:
    """单个序列存储"""
    
    def __init__(self, max_length: int = 200):
        self.max_length = max_length
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.hidden_states = []  # 存储RNN隐藏状态
        self.priorities = []  # 优先级（可选）
        
        # 序列元数据
        self.episode_id = None
        self.start_step = 0
    
    def add_step(self, state, action, reward, done, hidden_state=None, priority=1.0):
        """添加一个时间步"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.hidden_states.append(hidden_state)
        self.priorities.append(priority)
        
        # 如果超过最大长度，移除最早的数据
        if len(self.states) > self.max_length:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.dones.pop(0)
            self.hidden_states.pop(0)
            self.priorities.pop(0)
            self.start_step += 1
    
    def get_sequence(self, start_idx: int, length: int) -> Dict:
        """获取指定长度的序列"""
        end_idx = min(start_idx + length, len(self.states))
        actual_length = end_idx - start_idx
        
        sequence = {
            'states': np.array(self.states[start_idx:end_idx]),
            'actions': np.array(self.actions[start_idx:end_idx]),
            'rewards': np.array(self.rewards[start_idx:end_idx]),
            'dones': np.array(self.dones[start_idx:end_idx], dtype=np.float32),
            'hidden_states': self.hidden_states[start_idx:end_idx],
            'priorities': np.array(self.priorities[start_idx:end_idx]),
            'length': actual_length,
            'episode_id': self.episode_id,
            'start_step': self.start_step + start_idx
        }
        
        return sequence
    
    def __len__(self):
        return len(self.states)
    
    def is_valid_start(self, start_idx: int, min_length: int) -> bool:
        """检查是否可以从该位置开始采样"""
        return start_idx + min_length <= len(self.states)


class R2D2SequenceReplayBuffer:
    """R2D2序列经验回放缓冲区"""
    
    def __init__(self,
                 capacity: int = 10000,
                 sequence_length: int = 40,
                 burn_in_length: int = 20,
                 overlap_length: int = 10,
                 device: torch.device = torch.device('cpu')):
        """
        初始化序列回放缓冲区
        
        Args:
            capacity: 最大序列数量
            sequence_length: 训练序列长度
            burn_in_length: burn-in序列长度（用于RNN warm-up）
            overlap_length: 序列重叠长度
            device: 计算设备
        """
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.burn_in_length = burn_in_length
        self.overlap_length = overlap_length
        self.device = device
        
        self.sequences = deque(maxlen=capacity)
        self.current_sequence = SequenceBuffer()
        self.episode_count = 0
        
        # 统计信息
        self.total_samples = 0
        self.total_episodes = 0
    
    def add_step(self, state, action, reward, done, hidden_state=None, priority=1.0):
        """添加一个时间步到当前序列"""
        self.current_sequence.add_step(state, action, reward, done, hidden_state, priority)
        self.total_samples += 1
        
        # 如果episode结束，保存当前序列并开始新序列
        if done:
            if len(self.current_sequence) > 0:
                self.current_sequence.episode_id = self.episode_count
                self.sequences.append(copy.deepcopy(self.current_sequence))
                
            # 开始新序列
            self.current_sequence = SequenceBuffer()
            self.episode_count += 1
            self.total_episodes += 1
    
    def sample_sequences(self, batch_size: int) -> Optional[Dict]:
        """
        采样序列批次
        
        Args:
            batch_size: 批次大小
            
        Returns:
            包含序列批次的字典，如果样本不足返回None
        """
        if len(self.sequences) < batch_size:
            return None
        
        # 随机选择序列
        selected_sequences = random.sample(list(self.sequences), batch_size)
        
        # 从每个序列中随机采样一个子序列
        batch_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'dones': [],
            'burn_in_states': [],
            'burn_in_actions': [],
            'burn_in_hidden_states': [],
            'initial_hidden_states': [],
            'sequence_lengths': [],
            'priorities': []
        }
        
        for seq_buffer in selected_sequences:
            # 确定采样位置
            total_needed = self.burn_in_length + self.sequence_length
            
            if len(seq_buffer) >= total_needed:
                # 序列足够长，随机选择起始位置
                max_start = len(seq_buffer) - total_needed
                start_idx = random.randint(0, max_start) if max_start > 0 else 0
            else:
                # 序列太短，从头开始
                start_idx = 0
            
            # 获取burn-in序列
            burn_in_end = start_idx + self.burn_in_length
            burn_in_seq = None  # 初始化变量
            
            if burn_in_end <= len(seq_buffer):
                burn_in_seq = seq_buffer.get_sequence(start_idx, self.burn_in_length)
                batch_data['burn_in_states'].append(burn_in_seq['states'])
                batch_data['burn_in_actions'].append(burn_in_seq['actions'])
                batch_data['burn_in_hidden_states'].append(burn_in_seq['hidden_states'])
            else:
                # Burn-in序列不足，用零填充
                actual_burn_in = max(0, len(seq_buffer) - self.sequence_length)
                if actual_burn_in > 0:
                    burn_in_seq = seq_buffer.get_sequence(0, actual_burn_in)
                    # 填充到所需长度
                    padded_states = self._pad_sequence(burn_in_seq['states'], self.burn_in_length)
                    padded_actions = self._pad_sequence(burn_in_seq['actions'], self.burn_in_length)
                    batch_data['burn_in_states'].append(padded_states)
                    batch_data['burn_in_actions'].append(padded_actions)
                else:
                    # 没有burn-in数据，创建零填充
                    zero_states = np.zeros((self.burn_in_length,) + seq_buffer.states[0].shape)
                    zero_actions = np.zeros((self.burn_in_length,), dtype=np.int32)
                    batch_data['burn_in_states'].append(zero_states)
                    batch_data['burn_in_actions'].append(zero_actions)
                    # 创建空的burn_in_seq用于后续引用
                    burn_in_seq = {'hidden_states': []}
                batch_data['burn_in_hidden_states'].append([None] * self.burn_in_length)
            
            # 获取训练序列
            train_start = max(start_idx, burn_in_end - self.overlap_length)
            train_seq = seq_buffer.get_sequence(train_start, self.sequence_length)
            
            # 填充序列到固定长度
            padded_states = self._pad_sequence(train_seq['states'], self.sequence_length)
            padded_actions = self._pad_sequence(train_seq['actions'], self.sequence_length)
            padded_rewards = self._pad_sequence(train_seq['rewards'], self.sequence_length)
            padded_dones = self._pad_sequence(train_seq['dones'], self.sequence_length)
            padded_priorities = self._pad_sequence(train_seq['priorities'], self.sequence_length)
            
            batch_data['states'].append(padded_states)
            batch_data['actions'].append(padded_actions)
            batch_data['rewards'].append(padded_rewards)
            batch_data['dones'].append(padded_dones)
            batch_data['priorities'].append(padded_priorities)
            batch_data['sequence_lengths'].append(train_seq['length'])
            
            # 初始隐藏状态（burn-in后的状态，或零状态）
            if burn_in_seq and len(burn_in_seq['hidden_states']) > 0:
                initial_hidden = burn_in_seq['hidden_states'][-1]
            else:
                initial_hidden = None
            batch_data['initial_hidden_states'].append(initial_hidden)
        
        # 转换为张量
        tensor_batch = {}
        for key, value in batch_data.items():
            if key in ['burn_in_hidden_states', 'initial_hidden_states']:
                tensor_batch[key] = value  # 保持列表格式
            elif key == 'sequence_lengths':
                tensor_batch[key] = torch.LongTensor(value).to(self.device)
            else:
                # 确保正确的张量形状 [batch_size, seq_len, ...]
                if key in ['states', 'burn_in_states']:
                    # 状态需要特殊处理以确保正确的维度顺序
                    # 检查所有序列是否有相同形状
                    if len(value) > 0:
                        shapes = [v.shape for v in value]
                        if all(s == shapes[0] for s in shapes):
                            # 所有形状相同，可以直接堆叠
                            numpy_array = np.stack(value, axis=0)
                        else:
                            # 形状不同，需要手动填充到相同大小
                            max_seq_len = max(v.shape[0] for v in value)
                            state_dim = value[0].shape[1] if len(value[0].shape) > 1 else 1
                            
                            # 创建填充后的数组
                            batch_size = len(value)
                            if len(value[0].shape) > 1:
                                padded_array = np.zeros((batch_size, max_seq_len, state_dim), dtype=value[0].dtype)
                            else:
                                padded_array = np.zeros((batch_size, max_seq_len), dtype=value[0].dtype)
                            
                            # 填充每个序列
                            for i, seq in enumerate(value):
                                seq_len = seq.shape[0]
                                if len(seq.shape) > 1:
                                    padded_array[i, :seq_len, :] = seq
                                else:
                                    padded_array[i, :seq_len] = seq
                            
                            numpy_array = padded_array
                    else:
                        numpy_array = np.array(value)
                else:
                    numpy_array = np.array(value)  # 其他数据正常处理
                
                tensor_batch[key] = torch.FloatTensor(numpy_array).to(self.device)
        
        return tensor_batch
    
    def _pad_sequence(self, sequence: np.ndarray, target_length: int) -> np.ndarray:
        """将序列填充到目标长度"""
        current_length = len(sequence)
        
        if current_length >= target_length:
            return sequence[:target_length]
        
        # 需要填充
        if len(sequence.shape) == 1:
            # 1D数组
            padding = np.zeros(target_length - current_length, dtype=sequence.dtype)
            return np.concatenate([sequence, padding])
        else:
            # 多维数组
            pad_shape = (target_length - current_length,) + sequence.shape[1:]
            padding = np.zeros(pad_shape, dtype=sequence.dtype)
            return np.concatenate([sequence, padding], axis=0)
    
    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """更新优先级（用于优先级经验回放）"""
        # 这里可以实现优先级更新逻辑
        # 当前简化版本不实现
        pass
    
    def clear(self):
        """清空缓冲区"""
        self.sequences.clear()
        self.current_sequence = SequenceBuffer()
        self.total_samples = 0
        self.total_episodes = 0
    
    def __len__(self):
        return len(self.sequences)
    
    @property
    def size(self):
        """缓冲区中的总步数"""
        return sum(len(seq) for seq in self.sequences) + len(self.current_sequence)
    
    @property 
    def is_ready(self):
        """是否有足够的数据进行采样"""
        return len(self.sequences) >= 10  # 至少需要10个序列
    
    def get_stats(self) -> Dict:
        """获取缓冲区统计信息"""
        return {
            'num_sequences': len(self.sequences),
            'total_samples': self.total_samples,
            'total_episodes': self.total_episodes,
            'current_sequence_length': len(self.current_sequence),
            'average_sequence_length': np.mean([len(seq) for seq in self.sequences]) if self.sequences else 0
        }