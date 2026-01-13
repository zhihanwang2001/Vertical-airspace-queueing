"""
外卖柜模型
Delivery Cabinet Model

实现24格外卖柜的数字孪生建模：
- 24格存储系统 (8+8+8三温区)
- 温区管理数学模型
- 订单存储策略
- 与垂直分层队列的协同
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import random

from .config import VerticalQueueConfig
from .utils import MathUtils


class TemperatureZone(Enum):
    """温区类型"""
    COLD = "cold"      # 冷藏区 (0-5°C)
    HOT = "hot"        # 保温区 (55-65°C)  
    NORMAL = "normal"  # 常温区 (15-25°C)


@dataclass
class GridCell:
    """外卖柜格子"""
    cell_id: int
    zone: TemperatureZone
    is_occupied: bool = False
    order_id: Optional[int] = None
    size_capacity: str = "medium"  # "small", "medium", "large"
    temperature: float = 20.0
    storage_time: int = 0  # 存储时长


@dataclass
class StoredOrder:
    """存储的订单"""
    order_id: int
    zone_required: TemperatureZone
    size: str
    storage_start_time: int
    max_storage_time: int = 120  # 最大存储时间(步数)
    priority: str = "medium"
    temperature_sensitive: bool = True


class DeliveryCabinet:
    """
    外卖柜模型类
    
    实现完整的24格外卖柜数字孪生：
    1. 物理结构：24格 = 8(冷) + 8(热) + 8(常温)
    2. 温区管理：动态温度控制
    3. 存储策略：基于温度、大小的智能分配
    4. 性能监控：利用率、温控效果、服务质量
    """
    
    def __init__(self, config: VerticalQueueConfig):
        self.config = config
        self.math_utils = MathUtils()
        
        # 系统参数
        self.total_cells = 24
        self.cells_per_zone = 8
        self.current_step = 0
        
        # 初始化格子系统
        self.grid_cells = self._initialize_grid()
        
        # 温控系统
        self.temperature_controller = self._initialize_temperature_controller()
        
        # 存储管理
        self.stored_orders = {}  # order_id -> StoredOrder
        self.service_queue = []  # 等待存储的订单队列
        
        # 性能统计
        self.performance_stats = {
            'total_stored': 0,
            'total_retrieved': 0,
            'storage_failures': 0,
            'temperature_violations': 0,
            'overtime_orders': 0,
            'zone_utilizations': {zone: 0.0 for zone in TemperatureZone}
        }
        
        # 故障状态
        self.is_system_failed = False
        self.failure_reason = None
    
    def _initialize_grid(self) -> List[GridCell]:
        """
        初始化24格存储系统
        
        布局：
        - 格子 0-7: 冷藏区 (COLD)
        - 格子 8-15: 保温区 (HOT)  
        - 格子 16-23: 常温区 (NORMAL)
        """
        cells = []
        
        # 冷藏区 (0-7)
        for i in range(8):
            cell = GridCell(
                cell_id=i,
                zone=TemperatureZone.COLD,
                size_capacity="medium",
                temperature=2.5  # 冷藏温度
            )
            cells.append(cell)
        
        # 保温区 (8-15)
        for i in range(8, 16):
            cell = GridCell(
                cell_id=i,
                zone=TemperatureZone.HOT,
                size_capacity="medium",
                temperature=60.0  # 保温温度
            )
            cells.append(cell)
        
        # 常温区 (16-23)
        for i in range(16, 24):
            cell = GridCell(
                cell_id=i,
                zone=TemperatureZone.NORMAL,
                size_capacity="medium",
                temperature=20.0  # 常温
            )
            cells.append(cell)
        
        return cells
    
    def _initialize_temperature_controller(self) -> Dict:
        """
        初始化温控系统
        
        基于01理论的温度保持模型：
        temp_i(t+1) = temp_i(t) + α_i·(target_i - temp_i(t)) + β_i·load_i(t) + ε_i(t)
        """
        return {
            TemperatureZone.COLD: {
                'target_temp': 2.5,
                'current_temp': 2.5,
                'alpha': 0.1,  # 温控响应系数
                'beta': 0.05,  # 负载影响系数
                'tolerance': 2.0,  # 温度容忍范围
                'is_active': True
            },
            TemperatureZone.HOT: {
                'target_temp': 60.0,
                'current_temp': 60.0,
                'alpha': 0.08,
                'beta': 0.03,
                'tolerance': 3.0,
                'is_active': True
            },
            TemperatureZone.NORMAL: {
                'target_temp': 20.0,
                'current_temp': 20.0,
                'alpha': 0.05,
                'beta': 0.02,
                'tolerance': 5.0,
                'is_active': True
            }
        }
    
    def reset(self):
        """
        重置外卖柜状态
        """
        self.current_step = 0
        
        # 清空所有格子
        for cell in self.grid_cells:
            cell.is_occupied = False
            cell.order_id = None
            cell.storage_time = 0
        
        # 重置温度
        for zone, controller in self.temperature_controller.items():
            controller['current_temp'] = controller['target_temp']
        
        # 清空存储和队列
        self.stored_orders.clear()
        self.service_queue.clear()
        
        # 重置统计
        for key in self.performance_stats:
            if isinstance(self.performance_stats[key], dict):
                for subkey in self.performance_stats[key]:
                    self.performance_stats[key][subkey] = 0.0
            else:
                self.performance_stats[key] = 0
        
        # 重置故障状态
        self.is_system_failed = False
        self.failure_reason = None
    
    def step(self, service_requests: List[Dict]) -> Dict:
        """
        外卖柜系统步进
        
        处理：
        1. 新的存储请求
        2. 订单取出
        3. 温控系统更新
        4. 超时订单处理
        5. 性能统计更新
        
        Args:
            service_requests: 来自队列系统的服务请求列表
            
        Returns:
            外卖柜状态信息字典
        """
        self.current_step += 1
        
        # 1. 处理新的存储请求
        storage_results = self._process_storage_requests(service_requests)
        
        # 2. 处理订单取出 (模拟客户取货)
        retrieval_results = self._process_order_retrievals()
        
        # 3. 更新温控系统
        temperature_info = self._update_temperature_control()
        
        # 4. 处理超时订单
        timeout_info = self._handle_timeout_orders()
        
        # 5. 更新存储时间
        self._update_storage_times()
        
        # 6. 检查系统故障
        self._check_system_health()
        
        # 7. 更新性能统计
        self._update_performance_stats()
        
        return {
            'grid_states': self._get_grid_states(),
            'temperatures': self._get_zone_temperatures(),
            'zone_loads': self._get_zone_loads(),
            'thermal_status': self._get_thermal_status(),
            'service_queue_length': len(self.service_queue),
            'occupancy_rate': self.get_occupancy_rate(),
            'storage_results': storage_results,
            'retrieval_results': retrieval_results,
            'temperature_info': temperature_info,
            'timeout_info': timeout_info,
            'performance_stats': self.performance_stats.copy(),
            'system_failed': self.is_system_failed,
            'failure_reason': self.failure_reason
        }
    
    def _process_storage_requests(self, service_requests: List[Dict]) -> Dict:
        """
        处理存储请求
        
        实现智能存储分配策略：
        1. 温度匹配优先
        2. 大小适配
        3. 负载均衡
        """
        successful_storage = 0
        failed_storage = 0
        storage_details = []
        
        for request in service_requests:
            # 创建存储订单
            stored_order = StoredOrder(
                order_id=request.get('order_id', random.randint(1000, 9999)),
                zone_required=self._determine_temperature_zone(request),
                size=request.get('size', 'medium'),
                storage_start_time=self.current_step,
                priority=request.get('priority', 'medium')
            )
            
            # 寻找合适的格子
            target_cell = self._find_optimal_cell(stored_order)
            
            if target_cell is not None:
                # 执行存储
                target_cell.is_occupied = True
                target_cell.order_id = stored_order.order_id
                target_cell.storage_time = 0
                
                self.stored_orders[stored_order.order_id] = stored_order
                successful_storage += 1
                
                storage_details.append({
                    'order_id': stored_order.order_id,
                    'cell_id': target_cell.cell_id,
                    'zone': target_cell.zone.value,
                    'success': True
                })
            else:
                # 存储失败，加入服务队列
                self.service_queue.append(stored_order)
                failed_storage += 1
                
                storage_details.append({
                    'order_id': stored_order.order_id,
                    'success': False,
                    'reason': 'no_available_cell'
                })
        
        return {
            'successful': successful_storage,
            'failed': failed_storage,
            'details': storage_details,
            'queue_length': len(self.service_queue)
        }
    
    def _determine_temperature_zone(self, request: Dict) -> TemperatureZone:
        """
        确定订单所需温区
        """
        temp_zone = request.get('temperature_zone', 'normal')
        
        zone_mapping = {
            'cold': TemperatureZone.COLD,
            'hot': TemperatureZone.HOT,
            'normal': TemperatureZone.NORMAL
        }
        
        return zone_mapping.get(temp_zone, TemperatureZone.NORMAL)
    
    def _find_optimal_cell(self, order: StoredOrder) -> Optional[GridCell]:
        """
        寻找最优存储格子
        
        分配策略基于01理论中的订单分配函数：
        P(assign_o→(i,j)) = exp(φ_temp·match(o.temp,i) + φ_size·fit(o.size,j)) / Σ exp(...)
        """
        available_cells = [cell for cell in self.grid_cells 
                          if not cell.is_occupied and cell.zone == order.zone_required]
        
        if not available_cells:
            return None
        
        # 计算每个可用格子的适配分数
        cell_scores = []
        for cell in available_cells:
            score = self._calculate_cell_score(cell, order)
            cell_scores.append((cell, score))
        
        # 按分数排序，选择最佳格子
        cell_scores.sort(key=lambda x: x[1], reverse=True)
        
        return cell_scores[0][0] if cell_scores else None
    
    def _calculate_cell_score(self, cell: GridCell, order: StoredOrder) -> float:
        """
        计算格子-订单适配分数
        
        考虑因素：
        1. 温度匹配度
        2. 大小适配度  
        3. 负载均衡
        """
        # 温度匹配度 (完美匹配得分1.0)
        temp_match = 1.0 if cell.zone == order.zone_required else 0.0
        
        # 大小适配度
        size_match = self._calculate_size_match(cell.size_capacity, order.size)
        
        # 负载均衡因子 (优先选择负载较低的区域)
        zone_load = self._get_zone_load(order.zone_required)
        load_factor = 1.0 - zone_load  # 负载越低，因子越高
        
        # 优先级权重
        priority_weight = {'low': 0.8, 'medium': 1.0, 'high': 1.2}
        priority_factor = priority_weight.get(order.priority, 1.0)
        
        # 综合评分
        score = (temp_match * 0.4 + 
                size_match * 0.3 + 
                load_factor * 0.2 + 
                priority_factor * 0.1)
        
        return score
    
    def _calculate_size_match(self, cell_capacity: str, order_size: str) -> float:
        """
        计算大小匹配度
        """
        size_order = {'small': 1, 'medium': 2, 'large': 3}
        capacity_order = {'small': 1, 'medium': 2, 'large': 3}
        
        order_val = size_order.get(order_size, 2)
        capacity_val = capacity_order.get(cell_capacity, 2)
        
        if order_val <= capacity_val:
            return 1.0 - (capacity_val - order_val) * 0.2  # 优先大小匹配
        else:
            return 0.0  # 无法容纳
    
    def _process_order_retrievals(self) -> Dict:
        """
        处理订单取出 (模拟客户取货)
        """
        retrieval_count = 0
        retrieved_orders = []
        
        # 模拟取货过程 (基于简单概率模型)
        for cell in self.grid_cells:
            if cell.is_occupied and cell.order_id in self.stored_orders:
                order = self.stored_orders[cell.order_id]
                
                # 取货概率 (基于存储时间和优先级)
                retrieval_prob = self._calculate_retrieval_probability(order, cell)
                
                if random.random() < retrieval_prob:
                    # 执行取货
                    retrieved_orders.append({
                        'order_id': order.order_id,
                        'cell_id': cell.cell_id,
                        'storage_duration': cell.storage_time,
                        'zone': cell.zone.value
                    })
                    
                    # 清空格子
                    cell.is_occupied = False
                    cell.order_id = None
                    cell.storage_time = 0
                    
                    # 移除订单记录
                    del self.stored_orders[order.order_id]
                    retrieval_count += 1
        
        # 处理服务队列中的订单 (如果有空位)
        self._process_waiting_queue()
        
        return {
            'count': retrieval_count,
            'orders': retrieved_orders
        }
    
    def _calculate_retrieval_probability(self, order: StoredOrder, cell: GridCell) -> float:
        """
        计算取货概率
        
        基于存储时间和订单特性
        """
        # 基础取货概率
        base_prob = 0.1
        
        # 时间因子 (存储时间越长，取货概率越高)
        time_factor = min(cell.storage_time / 30.0, 1.0)  # 30步后达到最大概率
        
        # 优先级因子
        priority_factor = {'low': 0.8, 'medium': 1.0, 'high': 1.5}
        priority_mult = priority_factor.get(order.priority, 1.0)
        
        # 最终概率
        prob = base_prob + time_factor * 0.3 * priority_mult
        
        return min(prob, 0.8)  # 最大80%取货概率
    
    def _process_waiting_queue(self):
        """
        处理等待队列中的订单
        """
        if not self.service_queue:
            return
        
        # 尝试为等待队列中的订单分配格子
        processed_orders = []
        for order in self.service_queue:
            target_cell = self._find_optimal_cell(order)
            if target_cell is not None:
                # 分配成功
                target_cell.is_occupied = True
                target_cell.order_id = order.order_id
                target_cell.storage_time = 0
                
                self.stored_orders[order.order_id] = order
                processed_orders.append(order)
        
        # 从等待队列中移除已处理的订单
        for order in processed_orders:
            self.service_queue.remove(order)
    
    def _update_temperature_control(self) -> Dict:
        """
        更新温控系统
        
        基于01理论温度保持模型：
        temp_i(t+1) = temp_i(t) + α_i·(target_i - temp_i(t)) + β_i·load_i(t) + ε_i(t)
        """
        temp_info = {}
        
        for zone, controller in self.temperature_controller.items():
            if not controller['is_active']:
                continue
            
            # 当前温度
            current_temp = controller['current_temp']
            target_temp = controller['target_temp']
            alpha = controller['alpha']
            beta = controller['beta']
            
            # 计算负载影响
            zone_load = self._get_zone_load(zone)
            load_effect = beta * zone_load * (random.random() - 0.5) * 2  # 负载带来的温度波动
            
            # 随机温度波动
            epsilon = random.gauss(0, 0.5)
            
            # 更新温度
            new_temp = (current_temp + 
                       alpha * (target_temp - current_temp) + 
                       load_effect + 
                       epsilon)
            
            controller['current_temp'] = new_temp
            
            # 更新格子温度
            for cell in self.grid_cells:
                if cell.zone == zone:
                    cell.temperature = new_temp
            
            # 检查温度违规
            temp_violation = abs(new_temp - target_temp) > controller['tolerance']
            if temp_violation:
                self.performance_stats['temperature_violations'] += 1
            
            temp_info[zone.value] = {
                'current': new_temp,
                'target': target_temp,
                'load': zone_load,
                'violation': temp_violation
            }
        
        return temp_info
    
    def _handle_timeout_orders(self) -> Dict:
        """
        处理超时订单
        """
        timeout_orders = []
        
        for cell in self.grid_cells:
            if cell.is_occupied and cell.order_id in self.stored_orders:
                order = self.stored_orders[cell.order_id]
                
                # 检查是否超时
                if cell.storage_time > order.max_storage_time:
                    timeout_orders.append({
                        'order_id': order.order_id,
                        'cell_id': cell.cell_id,
                        'overtime': cell.storage_time - order.max_storage_time
                    })
                    
                    # 强制取出超时订单
                    cell.is_occupied = False
                    cell.order_id = None
                    cell.storage_time = 0
                    
                    del self.stored_orders[order.order_id]
                    self.performance_stats['overtime_orders'] += 1
        
        return {
            'count': len(timeout_orders),
            'orders': timeout_orders
        }
    
    def _update_storage_times(self):
        """
        更新所有存储订单的时间
        """
        for cell in self.grid_cells:
            if cell.is_occupied:
                cell.storage_time += 1
    
    def _check_system_health(self):
        """
        检查系统健康状态
        """
        # 检查温控系统故障
        temp_failures = 0
        for zone, controller in self.temperature_controller.items():
            temp_diff = abs(controller['current_temp'] - controller['target_temp'])
            if temp_diff > controller['tolerance'] * 2:  # 严重温度偏差
                temp_failures += 1
        
        if temp_failures >= 2:  # 多个温区同时故障
            self.is_system_failed = True
            self.failure_reason = "multiple_temperature_control_failure"
        
        # 检查过载情况
        if len(self.service_queue) > 20:  # 等待队列过长
            self.is_system_failed = True
            self.failure_reason = "service_queue_overflow"
    
    def _update_performance_stats(self):
        """
        更新性能统计
        """
        # 更新区域利用率
        for zone in TemperatureZone:
            zone_cells = [cell for cell in self.grid_cells if cell.zone == zone]
            occupied_cells = [cell for cell in zone_cells if cell.is_occupied]
            utilization = len(occupied_cells) / len(zone_cells) if zone_cells else 0
            self.performance_stats['zone_utilizations'][zone] = utilization
    
    def _get_grid_states(self) -> List[int]:
        """
        获取24格状态 (0/1表示空/占用)
        """
        return [1 if cell.is_occupied else 0 for cell in self.grid_cells]
    
    def _get_zone_temperatures(self) -> List[float]:
        """
        获取三个温区的当前温度
        """
        temps = []
        for zone in [TemperatureZone.COLD, TemperatureZone.HOT, TemperatureZone.NORMAL]:
            temp = self.temperature_controller[zone]['current_temp']
            temps.append(temp)
        return temps
    
    def _get_zone_loads(self) -> List[float]:
        """
        获取三个温区的负载率
        """
        loads = []
        for zone in [TemperatureZone.COLD, TemperatureZone.HOT, TemperatureZone.NORMAL]:
            load = self._get_zone_load(zone)
            loads.append(load)
        return loads
    
    def _get_zone_load(self, zone: TemperatureZone) -> float:
        """
        获取指定温区的负载率
        """
        zone_cells = [cell for cell in self.grid_cells if cell.zone == zone]
        occupied_cells = [cell for cell in zone_cells if cell.is_occupied]
        return len(occupied_cells) / len(zone_cells) if zone_cells else 0
    
    def _get_thermal_status(self) -> float:
        """
        获取温控系统整体状态 (0-1，1表示最佳)
        """
        total_score = 0
        active_controllers = 0
        
        for zone, controller in self.temperature_controller.items():
            if controller['is_active']:
                temp_diff = abs(controller['current_temp'] - controller['target_temp'])
                max_diff = controller['tolerance']
                score = max(0, 1 - temp_diff / max_diff)
                total_score += score
                active_controllers += 1
        
        return total_score / active_controllers if active_controllers > 0 else 0
    
    def get_occupancy_rate(self) -> float:
        """
        获取整体占用率
        """
        occupied = sum(1 for cell in self.grid_cells if cell.is_occupied)
        return occupied / len(self.grid_cells)
    
    def is_failed(self) -> bool:
        """
        检查系统是否故障
        """
        return self.is_system_failed
    
    def get_detailed_info(self) -> Dict:
        """
        获取详细的外卖柜信息
        """
        return {
            'total_cells': self.total_cells,
            'occupied_cells': sum(1 for cell in self.grid_cells if cell.is_occupied),
            'occupancy_rate': self.get_occupancy_rate(),
            'zone_info': {
                zone.value: {
                    'cells': [cell.cell_id for cell in self.grid_cells if cell.zone == zone],
                    'occupied': len([cell for cell in self.grid_cells 
                                   if cell.zone == zone and cell.is_occupied]),
                    'temperature': self.temperature_controller[zone]['current_temp'],
                    'target_temperature': self.temperature_controller[zone]['target_temp'],
                    'utilization': self.performance_stats['zone_utilizations'][zone]
                }
                for zone in TemperatureZone
            },
            'service_queue_length': len(self.service_queue),
            'stored_orders_count': len(self.stored_orders),
            'performance_stats': self.performance_stats,
            'system_status': {
                'is_failed': self.is_system_failed,
                'failure_reason': self.failure_reason,
                'thermal_status': self._get_thermal_status()
            }
        }


# 测试外卖柜模型
if __name__ == "__main__":
    from .config import VerticalQueueConfig
    
    config = VerticalQueueConfig()
    cabinet = DeliveryCabinet(config)
    
    print("外卖柜模型创建成功!")
    print(f"总格子数: {cabinet.total_cells}")
    print(f"初始占用率: {cabinet.get_occupancy_rate():.1%}")
    
    # 测试存储请求
    test_requests = [
        {'order_id': 1001, 'temperature_zone': 'cold', 'size': 'medium'},
        {'order_id': 1002, 'temperature_zone': 'hot', 'size': 'small'},
        {'order_id': 1003, 'temperature_zone': 'normal', 'size': 'large'},
    ]
    
    print("\n开始外卖柜仿真...")
    for step in range(10):
        # 随机添加一些服务请求
        requests = test_requests if step < 3 else []
        
        info = cabinet.step(requests)
        
        print(f"Step {step + 1}: 占用率{info['occupancy_rate']:.1%}, "
              f"服务队列{info['service_queue_length']}, "
              f"温控状态{info['thermal_status']:.2f}")
    
    # 显示详细信息
    detail_info = cabinet.get_detailed_info()
    print(f"\n最终状态:")
    print(f"占用格子: {detail_info['occupied_cells']}/{detail_info['total_cells']}")
    zone_utilizations = [f'{zone}: {info["utilization"]:.1%}' for zone, info in detail_info['zone_info'].items()]
    print(f"各温区利用率: {zone_utilizations}")
    
    print("\n外卖柜模型测试完成!")