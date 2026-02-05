"""
Delivery Cabinet Model

Implements digital twin modeling of 24-compartment delivery cabinet:
- 24-compartment storage system (8+8+8 three temperature zones)
- Temperature zone management mathematical model
- Order storage strategy
- Coordination with vertical stratified queue
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import random

from .config import VerticalQueueConfig
from .utils import MathUtils


class TemperatureZone(Enum):
    """Temperature zone types"""
    COLD = "cold"      # Cold storage zone (0-5°C)
    HOT = "hot"        # Warming zone (55-65°C)  
    NORMAL = "normal"  # Normal temperature zone (15-25°C)


@dataclass
class GridCell:
    """Delivery cabinet compartment"""
    cell_id: int
    zone: TemperatureZone
    is_occupied: bool = False
    order_id: Optional[int] = None
    size_capacity: str = "medium"  # "small", "medium", "large"
    temperature: float = 20.0
    storage_time: int = 0  # Storage duration


@dataclass
class StoredOrder:
    """Stored order"""
    order_id: int
    zone_required: TemperatureZone
    size: str
    storage_start_time: int
    max_storage_time: int = 120  # Maximum storage time (steps)
    priority: str = "medium"
    temperature_sensitive: bool = True


class DeliveryCabinet:
    """
    Delivery Cabinet Model Class
    
    Implements complete 24-compartment delivery cabinet digital twin:
    1. Physical structure: 24 compartments = 8(cold) + 8(hot) + 8(normal)
    2. Temperature zone management: Dynamic temperature control
    3. Storage strategy: Intelligent allocation based on temperature and size
    4. Performance monitoring: Utilization rate, temperature control effectiveness, service quality
    """
    
    def __init__(self, config: VerticalQueueConfig):
        self.config = config
        self.math_utils = MathUtils()
        
        # System parameters
        self.total_cells = 24
        self.cells_per_zone = 8
        self.current_step = 0
        
        # Initialize compartment system
        self.grid_cells = self._initialize_grid()
        
        # Temperature control system
        self.temperature_controller = self._initialize_temperature_controller()
        
        # Storage management
        self.stored_orders = {}  # order_id -> StoredOrder
        self.service_queue = []  # Queue of orders waiting for storage
        
        # Performance statistics
        self.performance_stats = {
            'total_stored': 0,
            'total_retrieved': 0,
            'storage_failures': 0,
            'temperature_violations': 0,
            'overtime_orders': 0,
            'zone_utilizations': {zone: 0.0 for zone in TemperatureZone}
        }
        
        # Failure status
        self.is_system_failed = False
        self.failure_reason = None
    
    def _initialize_grid(self) -> List[GridCell]:
        """
        Initialize 24-compartment storage system
        
        Layout:
        - Compartments 0-7: Cold storage zone (COLD)
        - Compartments 8-15: Warming zone (HOT)  
        - Compartments 16-23: Normal temperature zone (NORMAL)
        """
        cells = []
        
        # Cold storage zone (0-7)
        for i in range(8):
            cell = GridCell(
                cell_id=i,
                zone=TemperatureZone.COLD,
                size_capacity="medium",
                temperature=2.5  # Cold storage temperature
            )
            cells.append(cell)
        
        # Warming zone (8-15)
        for i in range(8, 16):
            cell = GridCell(
                cell_id=i,
                zone=TemperatureZone.HOT,
                size_capacity="medium",
                temperature=60.0  # Warming temperature
            )
            cells.append(cell)
        
        # Normal temperature zone (16-23)
        for i in range(16, 24):
            cell = GridCell(
                cell_id=i,
                zone=TemperatureZone.NORMAL,
                size_capacity="medium",
                temperature=20.0  # Normal temperature
            )
            cells.append(cell)
        
        return cells
    
    def _initialize_temperature_controller(self) -> Dict:
        """
        Initialize temperature control system
        
        Temperature maintenance model based on 01 theory:
        temp_i(t+1) = temp_i(t) + α_i·(target_i - temp_i(t)) + β_i·load_i(t) + ε_i(t)
        """
        return {
            TemperatureZone.COLD: {
                'target_temp': 2.5,
                'current_temp': 2.5,
                'alpha': 0.1,  # Temperature control response coefficient
                'beta': 0.05,  # Load impact coefficient
                'tolerance': 2.0,  # Temperature tolerance range
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
        Reset delivery cabinet state
        """
        self.current_step = 0
        
        # Clear all compartments
        for cell in self.grid_cells:
            cell.is_occupied = False
            cell.order_id = None
            cell.storage_time = 0
        
        # Reset temperature
        for zone, controller in self.temperature_controller.items():
            controller['current_temp'] = controller['target_temp']
        
        # Clear storage and queue
        self.stored_orders.clear()
        self.service_queue.clear()
        
        # Reset statistics
        for key in self.performance_stats:
            if isinstance(self.performance_stats[key], dict):
                for subkey in self.performance_stats[key]:
                    self.performance_stats[key][subkey] = 0.0
            else:
                self.performance_stats[key] = 0
        
        # Reset failure status
        self.is_system_failed = False
        self.failure_reason = None
    
    def step(self, service_requests: List[Dict]) -> Dict:
        """
        Delivery cabinet system step
        
        Processes:
        1. New storage requests
        2. Order retrievals
        3. Temperature control system update
        4. Timeout order handling
        5. Performance statistics update
        
        Args:
            service_requests: List of service requests from queue system
            
        Returns:
            Dictionary of delivery cabinet state information
        """
        self.current_step += 1
        
        # 1. Process new storage requests
        storage_results = self._process_storage_requests(service_requests)
        
        # 2. Process order retrievals (simulate customer pickups)
        retrieval_results = self._process_order_retrievals()
        
        # 3. Update temperature control system
        temperature_info = self._update_temperature_control()
        
        # 4. Handle timeout orders
        timeout_info = self._handle_timeout_orders()
        
        # 5. Update storage times
        self._update_storage_times()
        
        # 6. Check system health
        self._check_system_health()
        
        # 7. Update performance statistics
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
        Process storage requests
        
        Implements intelligent storage allocation strategy:
        1. Temperature matching priority
        2. Size fitting
        3. Load balancing
        """
        successful_storage = 0
        failed_storage = 0
        storage_details = []
        
        for request in service_requests:
            # Create stored order
            stored_order = StoredOrder(
                order_id=request.get('order_id', random.randint(1000, 9999)),
                zone_required=self._determine_temperature_zone(request),
                size=request.get('size', 'medium'),
                storage_start_time=self.current_step,
                priority=request.get('priority', 'medium')
            )
            
            # Find suitable compartment
            target_cell = self._find_optimal_cell(stored_order)
            
            if target_cell is not None:
                # Execute storage
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
                # Storage failed, add to service queue
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
        Determine required temperature zone for order
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
        Find optimal storage compartment
        
        Allocation strategy based on order assignment function in 01 theory:
        P(assign_o→(i,j)) = exp(φ_temp·match(o.temp,i) + φ_size·fit(o.size,j)) / Σ exp(...)
        """
        available_cells = [cell for cell in self.grid_cells 
                          if not cell.is_occupied and cell.zone == order.zone_required]
        
        if not available_cells:
            return None
        
        # Calculate fitness score for each available compartment
        cell_scores = []
        for cell in available_cells:
            score = self._calculate_cell_score(cell, order)
            cell_scores.append((cell, score))
        
        # Sort by score, select best compartment
        cell_scores.sort(key=lambda x: x[1], reverse=True)
        
        return cell_scores[0][0] if cell_scores else None
    
    def _calculate_cell_score(self, cell: GridCell, order: StoredOrder) -> float:
        """
        Calculate compartment-order fitness score
        
        Considers:
        1. Temperature matching
        2. Size fitting  
        3. Load balancing
        """
        # Temperature matching (perfect match scores 1.0)
        temp_match = 1.0 if cell.zone == order.zone_required else 0.0
        
        # Size fitting
        size_match = self._calculate_size_match(cell.size_capacity, order.size)
        
        # Load balancing factor (prefer lower load zones)
        zone_load = self._get_zone_load(order.zone_required)
        load_factor = 1.0 - zone_load  # Lower load, higher factor
        
        # Priority weight
        priority_weight = {'low': 0.8, 'medium': 1.0, 'high': 1.2}
        priority_factor = priority_weight.get(order.priority, 1.0)
        
        # Composite score
        score = (temp_match * 0.4 + 
                size_match * 0.3 + 
                load_factor * 0.2 + 
                priority_factor * 0.1)
        
        return score
    
    def _calculate_size_match(self, cell_capacity: str, order_size: str) -> float:
        """
        Calculate size matching degree
        """
        size_order = {'small': 1, 'medium': 2, 'large': 3}
        capacity_order = {'small': 1, 'medium': 2, 'large': 3}
        
        order_val = size_order.get(order_size, 2)
        capacity_val = capacity_order.get(cell_capacity, 2)
        
        if order_val <= capacity_val:
            return 1.0 - (capacity_val - order_val) * 0.2  # Prefer exact size match
        else:
            return 0.0  # Cannot accommodate
    
    def _process_order_retrievals(self) -> Dict:
        """
        Process order retrievals (simulate customer pickups)
        """
        retrieval_count = 0
        retrieved_orders = []
        
        # Simulate pickup process (based on simple probability model)
        for cell in self.grid_cells:
            if cell.is_occupied and cell.order_id in self.stored_orders:
                order = self.stored_orders[cell.order_id]
                
                # Retrieval probability (based on storage time and priority)
                retrieval_prob = self._calculate_retrieval_probability(order, cell)
                
                if random.random() < retrieval_prob:
                    # Execute retrieval
                    retrieved_orders.append({
                        'order_id': order.order_id,
                        'cell_id': cell.cell_id,
                        'storage_duration': cell.storage_time,
                        'zone': cell.zone.value
                    })
                    
                    # Clear compartment
                    cell.is_occupied = False
                    cell.order_id = None
                    cell.storage_time = 0
                    
                    # Remove order record
                    del self.stored_orders[order.order_id]
                    retrieval_count += 1
        
        # Process orders in service queue (if space available)
        self._process_waiting_queue()
        
        return {
            'count': retrieval_count,
            'orders': retrieved_orders
        }
    
    def _calculate_retrieval_probability(self, order: StoredOrder, cell: GridCell) -> float:
        """
        Calculate retrieval probability
        
        Based on storage time and order characteristics
        """
        # Base retrieval probability
        base_prob = 0.1
        
        # Time factor (longer storage time, higher retrieval probability)
        time_factor = min(cell.storage_time / 30.0, 1.0)  # Max probability after 30 steps
        
        # Priority factor
        priority_factor = {'low': 0.8, 'medium': 1.0, 'high': 1.5}
        priority_mult = priority_factor.get(order.priority, 1.0)
        
        # Final probability
        prob = base_prob + time_factor * 0.3 * priority_mult
        
        return min(prob, 0.8)  # Maximum 80% retrieval probability
    
    def _process_waiting_queue(self):
        """
        Process orders in waiting queue
        """
        if not self.service_queue:
            return
        
        # Try to allocate compartments for orders in waiting queue
        processed_orders = []
        for order in self.service_queue:
            target_cell = self._find_optimal_cell(order)
            if target_cell is not None:
                # Allocation successful
                target_cell.is_occupied = True
                target_cell.order_id = order.order_id
                target_cell.storage_time = 0
                
                self.stored_orders[order.order_id] = order
                processed_orders.append(order)
        
        # Remove processed orders from waiting queue
        for order in processed_orders:
            self.service_queue.remove(order)
    
    def _update_temperature_control(self) -> Dict:
        """
        Update temperature control system
        
        Based on 01 theory temperature maintenance model:
        temp_i(t+1) = temp_i(t) + α_i·(target_i - temp_i(t)) + β_i·load_i(t) + ε_i(t)
        """
        temp_info = {}
        
        for zone, controller in self.temperature_controller.items():
            if not controller['is_active']:
                continue
            
            # Current temperature
            current_temp = controller['current_temp']
            target_temp = controller['target_temp']
            alpha = controller['alpha']
            beta = controller['beta']
            
            # Calculate load effect
            zone_load = self._get_zone_load(zone)
            load_effect = beta * zone_load * (random.random() - 0.5) * 2  # Temperature fluctuation from load
            
            # Random temperature fluctuation
            epsilon = random.gauss(0, 0.5)
            
            # Update temperature
            new_temp = (current_temp + 
                       alpha * (target_temp - current_temp) + 
                       load_effect + 
                       epsilon)
            
            controller['current_temp'] = new_temp
            
            # Update compartment temperatures
            for cell in self.grid_cells:
                if cell.zone == zone:
                    cell.temperature = new_temp
            
            # Check temperature violation
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
        Handle timeout orders
        """
        timeout_orders = []
        
        for cell in self.grid_cells:
            if cell.is_occupied and cell.order_id in self.stored_orders:
                order = self.stored_orders[cell.order_id]
                
                # Check if timeout
                if cell.storage_time > order.max_storage_time:
                    timeout_orders.append({
                        'order_id': order.order_id,
                        'cell_id': cell.cell_id,
                        'overtime': cell.storage_time - order.max_storage_time
                    })
                    
                    # Force removal of timeout order
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
        Update storage time for all stored orders
        """
        for cell in self.grid_cells:
            if cell.is_occupied:
                cell.storage_time += 1
    
    def _check_system_health(self):
        """
        Check system health status
        """
        # Check temperature control system failure
        temp_failures = 0
        for zone, controller in self.temperature_controller.items():
            temp_diff = abs(controller['current_temp'] - controller['target_temp'])
            if temp_diff > controller['tolerance'] * 2:  # Severe temperature deviation
                temp_failures += 1
        
        if temp_failures >= 2:  # Multiple zones failing simultaneously
            self.is_system_failed = True
            self.failure_reason = "multiple_temperature_control_failure"
        
        # Check overload situation
        if len(self.service_queue) > 20:  # Waiting queue too long
            self.is_system_failed = True
            self.failure_reason = "service_queue_overflow"
    
    def _update_performance_stats(self):
        """
        Update performance statistics
        """
        # Update zone utilization rates
        for zone in TemperatureZone:
            zone_cells = [cell for cell in self.grid_cells if cell.zone == zone]
            occupied_cells = [cell for cell in zone_cells if cell.is_occupied]
            utilization = len(occupied_cells) / len(zone_cells) if zone_cells else 0
            self.performance_stats['zone_utilizations'][zone] = utilization
    
    def _get_grid_states(self) -> List[int]:
        """
        Get 24-compartment states (0/1 for empty/occupied)
        """
        return [1 if cell.is_occupied else 0 for cell in self.grid_cells]
    
    def _get_zone_temperatures(self) -> List[float]:
        """
        Get current temperatures of three zones
        """
        temps = []
        for zone in [TemperatureZone.COLD, TemperatureZone.HOT, TemperatureZone.NORMAL]:
            temp = self.temperature_controller[zone]['current_temp']
            temps.append(temp)
        return temps
    
    def _get_zone_loads(self) -> List[float]:
        """
        Get load rates of three zones
        """
        loads = []
        for zone in [TemperatureZone.COLD, TemperatureZone.HOT, TemperatureZone.NORMAL]:
            load = self._get_zone_load(zone)
            loads.append(load)
        return loads
    
    def _get_zone_load(self, zone: TemperatureZone) -> float:
        """
        Get load rate of specified zone
        """
        zone_cells = [cell for cell in self.grid_cells if cell.zone == zone]
        occupied_cells = [cell for cell in zone_cells if cell.is_occupied]
        return len(occupied_cells) / len(zone_cells) if zone_cells else 0
    
    def _get_thermal_status(self) -> float:
        """
        Get overall temperature control system status (0-1, 1 is optimal)
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
        Get overall occupancy rate
        """
        occupied = sum(1 for cell in self.grid_cells if cell.is_occupied)
        return occupied / len(self.grid_cells)
    
    def is_failed(self) -> bool:
        """
        Check if system has failed
        """
        return self.is_system_failed
    
    def get_detailed_info(self) -> Dict:
        """
        Get detailed delivery cabinet information
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


# Test delivery cabinet model
if __name__ == "__main__":
    from .config import VerticalQueueConfig
    
    config = VerticalQueueConfig()
    cabinet = DeliveryCabinet(config)
    
    print("Delivery cabinet model created successfully!")
    print(f"Total compartments: {cabinet.total_cells}")
    print(f"Initial occupancy rate: {cabinet.get_occupancy_rate():.1%}")
    
    # Test storage requests
    test_requests = [
        {'order_id': 1001, 'temperature_zone': 'cold', 'size': 'medium'},
        {'order_id': 1002, 'temperature_zone': 'hot', 'size': 'small'},
        {'order_id': 1003, 'temperature_zone': 'normal', 'size': 'large'},
    ]
    
    print("\nStarting delivery cabinet simulation...")
    for step in range(10):
        # Randomly add some service requests
        requests = test_requests if step < 3 else []
        
        info = cabinet.step(requests)
        
        print(f"Step {step + 1}: Occupancy rate {info['occupancy_rate']:.1%}, "
              f"Service queue {info['service_queue_length']}, "
              f"Thermal status {info['thermal_status']:.2f}")
    
    # Display detailed information
    detail_info = cabinet.get_detailed_info()
    print(f"\nFinal state:")
    print(f"Occupied compartments: {detail_info['occupied_cells']}/{detail_info['total_cells']}")
    zone_utilizations = [f'{zone}: {info["utilization"]:.1%}' for zone, info in detail_info['zone_info'].items()]
    print(f"Zone utilization rates: {zone_utilizations}")
    
    print("\nDelivery cabinet model test completed!")
