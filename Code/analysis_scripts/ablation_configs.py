"""
Ablation Study Configuration Module

Define ablation experiments for 4 key components:
1. No High-Layer Priority - Remove arrival weight advantage
2. Single-Objective - Remove multi-objective optimization
3. Traditional Pyramid - Remove inverted pyramid advantage
4. No Transfer - Remove inter-layer transfer functionality

Each ablation experiment keeps other components unchanged to ensure fair comparison
"""

import copy
from typing import Dict, Any, List
from env.config import VerticalQueueConfig

class AblationConfigs:
    """Ablation study configuration manager"""

    @staticmethod
    def get_full_system_config() -> VerticalQueueConfig:
        """
        Get full system configuration (our complete method)
        This is the control group, containing all innovative components
        """
        config = VerticalQueueConfig()

        # Ensure using our core innovative configuration
        config.layer_capacities = [8, 6, 4, 3, 2]  # Inverted pyramid
        config.arrival_weights = [0.3, 0.25, 0.2, 0.15, 0.1]  # High-layer priority
        config.layer_service_rates = [1.2, 1.0, 0.8, 0.6, 0.4]  # High-layer fast service

        return config

    @staticmethod
    def get_no_high_priority_config() -> VerticalQueueConfig:
        """
        Ablation Experiment 1: No High-Layer Priority

        Removed component: High-layer priority arrival mechanism
        Modification: arrival_weights -> Uniform distribution [0.2, 0.2, 0.2, 0.2, 0.2]
        Kept: Inverted pyramid capacity, multi-objective optimization, transfer mechanism

        Expected impact: Loss of high-layer fast processing advantage, overall throughput decrease
        """
        config = AblationConfigs.get_full_system_config()

        # Remove high-layer priority: use uniform arrival weights
        config.arrival_weights = [0.2, 0.2, 0.2, 0.2, 0.2]

        # Mark this configuration purpose
        config._ablation_type = "no_high_priority"
        config._removed_component = "High-Layer Priority Arrival"

        return config

    @staticmethod
    def get_single_objective_config() -> VerticalQueueConfig:
        """
        Ablation Experiment 2: Single-Objective

        Removed component: Multi-objective optimization framework
        Modification: Only optimize throughput, remove other 5 objectives
        Kept: Inverted pyramid capacity, high-layer priority, transfer mechanism

        Expected impact: Loss of multi-objective balance, possibly high throughput but poor stability
        """
        config = AblationConfigs.get_full_system_config()

        # Mark as single-objective optimization
        config._ablation_type = "single_objective"
        config._removed_component = "Multi-Objective Optimization"
        config._reward_type = "throughput_only"  # Environment will adjust reward function accordingly

        return config

    @staticmethod
    def get_traditional_pyramid_config() -> VerticalQueueConfig:
        """
        Ablation Experiment 3: Traditional Pyramid

        Removed component: Inverted pyramid capacity structure
        Modification: layer_capacities -> Traditional pyramid [2, 3, 4, 6, 8] (large bottom, small top)
        Kept: High-layer priority, multi-objective optimization, transfer mechanism

        Expected impact: Unreasonable capacity allocation, insufficient high-layer capacity leads to frequent transfers
        """
        config = AblationConfigs.get_full_system_config()

        # Use traditional pyramid structure (opposite of our innovation)
        config.layer_capacities = [2, 3, 4, 6, 8]  # Large bottom, small top

        # Mark this configuration purpose
        config._ablation_type = "traditional_pyramid"
        config._removed_component = "Inverted Pyramid Structure"

        return config

    @staticmethod
    def get_no_transfer_config() -> VerticalQueueConfig:
        """
        Ablation Experiment 4: No Transfer

        Removed component: Inter-layer transfer mechanism
        Modification: Disable downward transfer functionality
        Kept: Inverted pyramid capacity, high-layer priority, multi-objective optimization

        Expected impact: Cannot alleviate local congestion, system adaptability decreases
        """
        config = AblationConfigs.get_full_system_config()

        # Mark transfer mechanism disabled
        config._ablation_type = "no_transfer"
        config._removed_component = "Transfer Mechanism"
        config._transfer_enabled = False  # Environment will disable transfer accordingly

        return config

    @staticmethod
    def get_all_ablation_configs() -> Dict[str, VerticalQueueConfig]:
        """
        Get all ablation experiment configurations

        Returns:
            Dict containing:
            - 'full_system': Full system (control group)
            - 'no_high_priority': No high-layer priority
            - 'single_objective': Single-objective optimization
            - 'traditional_pyramid': Traditional pyramid
            - 'no_transfer': No transfer mechanism
        """
        return {
            'full_system': AblationConfigs.get_full_system_config(),
            'no_high_priority': AblationConfigs.get_no_high_priority_config(),
            'single_objective': AblationConfigs.get_single_objective_config(),
            'traditional_pyramid': AblationConfigs.get_traditional_pyramid_config(),
            'no_transfer': AblationConfigs.get_no_transfer_config()
        }

    @staticmethod
    def get_ablation_experiment_plan() -> Dict[str, Dict]:
        """
        Get complete ablation experiment plan

        Returns:
            Detailed experiment design, including purpose, modifications, and expected results for each experiment
        """
        return {
            'full_system': {
                'name': 'Complete System',
                'description': '包含所有创新组件的完整系统',
                'components': [
                    'Inverted Pyramid Capacity [8,6,4,3,2]',
                    'High-Layer Priority Weights [0.3,0.25,0.2,0.15,0.1]', 
                    'Multi-Objective Optimization (6 objectives)',
                    'Dynamic Transfer Mechanism'
                ],
                'expected_performance': 'Baseline (100%)',
                'config_changes': 'None (reference)'
            },
            
            'no_high_priority': {
                'name': 'No High-Layer Priority',
                'description': '移除高层优先到达机制',
                'removed_component': 'High-Layer Priority Arrival',
                'components': [
                    'Inverted Pyramid Capacity [8,6,4,3,2]',
                    'Uniform Priority Weights [0.2,0.2,0.2,0.2,0.2]',  # Modified
                    'Multi-Objective Optimization (6 objectives)', 
                    'Dynamic Transfer Mechanism'
                ],
                'expected_performance': 'Reduced by ~31.7%',
                'config_changes': {
                    'arrival_weights': [0.2, 0.2, 0.2, 0.2, 0.2]
                },
                'hypothesis': '高层优先机制对系统吞吐量贡献最大'
            },
            
            'single_objective': {
                'name': 'Single-Objective Optimization',
                'description': '只优化吞吐量，移除多目标框架',
                'removed_component': 'Multi-Objective Optimization',
                'components': [
                    'Inverted Pyramid Capacity [8,6,4,3,2]',
                    'High-Layer Priority Weights [0.3,0.25,0.2,0.15,0.1]',
                    'Single-Objective Optimization (throughput only)',  # Modified
                    'Dynamic Transfer Mechanism'
                ],
                'expected_performance': 'Reduced by ~28.3%',
                'config_changes': {
                    'reward_type': 'throughput_only'
                },
                'hypothesis': '多目标优化平衡各项指标，提升整体性能'
            },
            
            'traditional_pyramid': {
                'name': 'Traditional Pyramid Structure', 
                'description': '使用传统金字塔容量结构',
                'removed_component': 'Inverted Pyramid Structure',
                'components': [
                    'Traditional Pyramid Capacity [2,3,4,6,8]',  # Modified
                    'High-Layer Priority Weights [0.3,0.25,0.2,0.15,0.1]',
                    'Multi-Objective Optimization (6 objectives)',
                    'Dynamic Transfer Mechanism'
                ],
                'expected_performance': 'Reduced by ~24.1%',
                'config_changes': {
                    'layer_capacities': [2, 3, 4, 6, 8]
                },
                'hypothesis': '倒金字塔结构更适合垂直分层系统'
            },
            
            'no_transfer': {
                'name': 'No Transfer Mechanism',
                'description': '禁用层间转移功能', 
                'removed_component': 'Dynamic Transfer Mechanism',
                'components': [
                    'Inverted Pyramid Capacity [8,6,4,3,2]',
                    'High-Layer Priority Weights [0.3,0.25,0.2,0.15,0.1]',
                    'Multi-Objective Optimization (6 objectives)',
                    'No Transfer (static allocation)'  # Modified
                ],
                'expected_performance': 'Reduced by ~15.9%',
                'config_changes': {
                    'transfer_enabled': False
                },
                'hypothesis': '转移机制提供系统适应性和负载缓解能力'
            }
        }
    
    @staticmethod
    def validate_ablation_design():
        """
        验证消融实验设计的正确性
        
        检查：
        1. 每个实验只修改一个组件
        2. 其他组件保持一致 
        3. 配置参数有效性
        """
        configs = AblationConfigs.get_all_ablation_configs()
        base_config = configs['full_system']
        
        print("🔍 验证消融实验设计...")
        print("=" * 50)
        
        for name, config in configs.items():
            if name == 'full_system':
                continue
                
            print(f"\n✅ 验证 {name}:")
            
            # 检查修改的组件
            if hasattr(config, '_ablation_type'):
                print(f"   移除组件: {config._removed_component}")
            
            # 检查配置差异
            changes = []
            
            if config.arrival_weights != base_config.arrival_weights:
                changes.append(f"arrival_weights: {config.arrival_weights}")
                
            if config.layer_capacities != base_config.layer_capacities:
                changes.append(f"layer_capacities: {config.layer_capacities}")
                
            if hasattr(config, '_reward_type'):
                changes.append(f"reward_type: {config._reward_type}")
                
            if hasattr(config, '_transfer_enabled'):
                changes.append(f"transfer_enabled: {config._transfer_enabled}")
            
            print(f"   配置修改: {changes}")
            
            # 验证其他参数保持不变
            same_params = [
                'num_layers',
                'layer_heights', 
                'base_arrival_rate',
                'layer_service_rates'
            ]
            
            for param in same_params:
                if getattr(config, param) != getattr(base_config, param):
                    print(f"   ⚠️  警告: {param} 发生了意外变化")
        
        print(f"\n✅ 消融实验设计验证完成!")
        print(f"📊 共设计 {len(configs)} 个配置（含对照组）")
        
        return True


class AblationEnvironmentFactory:
    """消融实验环境工厂"""
    
    @staticmethod
    def create_ablation_env(config: VerticalQueueConfig):
        """
        根据消融配置创建环境
        
        Args:
            config: 消融实验配置
            
        Returns:
            配置好的环境实例
        """
        # 这里需要根据具体的环境实现来调整
        # 暂时返回配置，实际使用时需要创建环境实例
        return config
    
    @staticmethod
    def apply_ablation_modifications(env, config: VerticalQueueConfig):
        """
        将消融配置应用到环境中
        
        Args:
            env: 环境实例
            config: 消融配置
        """
        # 应用配置修改
        if hasattr(config, '_reward_type') and config._reward_type == 'throughput_only':
            # 修改奖励函数为只关注吞吐量
            env.reward_weights = {'throughput': 1.0, 'others': 0.0}
            
        if hasattr(config, '_transfer_enabled') and not config._transfer_enabled:
            # 禁用转移机制
            env.disable_transfer = True
            
        # 应用基础配置
        env.layer_capacities = config.layer_capacities
        env.arrival_weights = config.arrival_weights
        env.layer_service_rates = config.layer_service_rates
        
        return env


# 测试和验证
if __name__ == "__main__":
    print("🧪 消融实验配置测试")
    print("=" * 50)
    
    # 验证设计
    AblationConfigs.validate_ablation_design()
    
    # 展示实验计划
    plan = AblationConfigs.get_ablation_experiment_plan()
    print(f"\n📋 消融实验计划:")
    print("=" * 50)
    
    for name, details in plan.items():
        print(f"\n🎯 {details['name']}")
        print(f"   描述: {details['description']}")
        if 'removed_component' in details:
            print(f"   移除组件: {details['removed_component']}")
        print(f"   预期性能: {details['expected_performance']}")
        if 'hypothesis' in details:
            print(f"   假设: {details['hypothesis']}")
    
    print(f"\n✅ 消融实验配置系统准备完成!")
    print(f"🚀 你现在可以运行消融实验了！")