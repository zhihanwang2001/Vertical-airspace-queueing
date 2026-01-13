"""
异质性区域配置
Heterogeneous Region Configurations

为第3.4节"跨场景泛化性分析"创建4个测试区域：
- Region B: 气象异质性（恶劣天气）
- Region C: 流量模式异质性（高峰时段）
- Region D: 监管政策异质性（限高政策）
- Region E: 能耗成本异质性（能耗敏感）
"""

import copy
from env.config import VerticalQueueConfig


class HeterogeneousRegionConfigs:
    """异质性区域配置生成器"""

    def __init__(self):
        """初始化基线配置"""
        self.base_config = VerticalQueueConfig()  # Region A (Standard)

    def create_region_b_weather(self) -> VerticalQueueConfig:
        """
        Region B: 气象异质性

        场景：恶劣天气（大风/降雨）导致服务率下降

        修改：
        - 服务率 × 0.8 (所有层级降低20%)
        - 最小等待时间 × 1.25 (转移变慢)

        预期影响：无人机受天气影响，上升和下降速度变慢
        """
        config = copy.deepcopy(self.base_config)

        # 服务率下降20% (恶劣天气影响)
        config.layer_service_rates = [rate * 0.8 for rate in self.base_config.layer_service_rates]

        # 最小等待时间增加25%
        config.min_wait_times = [int(t * 1.25) for t in self.base_config.min_wait_times]

        # 更新随机种子以确保可复现性
        config.random_seed = 43

        return config

    def create_region_c_traffic(self) -> VerticalQueueConfig:
        """
        Region C: 流量模式异质性

        场景：高峰时段，订单到达率大幅增加

        修改：
        - 基础到达率 × 1.5 (0.25 → 0.375)
        - 到达权重重新分布（更多订单涌入中低层）

        预期影响：系统负载增加，可能接近稳定性边界
        """
        config = copy.deepcopy(self.base_config)

        # 到达率提高50%
        config.base_arrival_rate = self.base_config.base_arrival_rate * 1.5  # 0.25 → 0.375

        # 到达权重重新分布（高峰时段更多订单分配到中低层）
        # 原始: [0.3, 0.25, 0.2, 0.15, 0.1] (L5→L1)
        # 新分布: [0.2, 0.2, 0.25, 0.2, 0.15] (更均匀，中低层压力增加)
        config.arrival_weights = [0.2, 0.2, 0.25, 0.2, 0.15]

        config.random_seed = 44

        return config

    def create_region_d_regulation(self) -> VerticalQueueConfig:
        """
        Region D: 监管政策异质性

        场景：限高政策，禁止使用最高2层（100m, 80m）

        修改：
        - 高层容量减少 (L5: 8→5, L4: 6→4)
        - 到达权重重新分配到低层

        预期影响：可用空域减少20%，低层拥挤度增加
        """
        config = copy.deepcopy(self.base_config)

        # 容量约束（高层受限）
        # 原始: [8, 6, 4, 3, 2] (L5→L1)
        # 新配置: [5, 4, 4, 3, 2] (高层减少，总容量23→18，-22%)
        config.layer_capacities = [5, 4, 4, 3, 2]

        # 到达权重重新分配（无人机倾向于使用低层）
        # 原始: [0.3, 0.25, 0.2, 0.15, 0.1]
        # 新配置: [0.15, 0.20, 0.25, 0.25, 0.15] (压力转移到中低层)
        config.arrival_weights = [0.15, 0.20, 0.25, 0.25, 0.15]

        config.random_seed = 45

        return config

    def create_region_e_energy_cost(self) -> VerticalQueueConfig:
        """
        Region E: 能耗成本异质性

        场景：能源价格敏感区域，高层飞行成本更高

        修改：
        - 这里通过修改奖励函数隐式实现（在测试时会使用不同的reward wrapper）
        - 为了保持配置一致性，我们修改服务率来模拟"不愿使用高层"的效果

        预期影响：策略偏好使用低层，导致低层过载
        """
        config = copy.deepcopy(self.base_config)

        # 模拟能耗敏感：通过降低高层的"吸引力"
        # 方法：降低高层服务率（模拟"虽然能服务，但成本高不愿意用"）
        # 原始: [1.2, 1.0, 0.8, 0.6, 0.4] (L5→L1)
        # 新配置: [0.9, 0.9, 0.8, 0.7, 0.5] (高层服务率下降更多)
        config.layer_service_rates = [0.9, 0.9, 0.8, 0.7, 0.5]

        # 到达权重也向低层倾斜
        config.arrival_weights = [0.15, 0.18, 0.22, 0.25, 0.20]

        config.random_seed = 46

        return config

    def get_all_configs(self) -> dict:
        """
        获取所有区域配置

        Returns:
            dict: {region_name: config}
        """
        return {
            'Region_A_Standard': self.base_config,
            'Region_B_Weather': self.create_region_b_weather(),
            'Region_C_Traffic': self.create_region_c_traffic(),
            'Region_D_Regulation': self.create_region_d_regulation(),
            'Region_E_EnergyCost': self.create_region_e_energy_cost()
        }

    def print_config_summary(self):
        """打印所有配置的对比摘要"""
        configs = self.get_all_configs()

        print("\n" + "="*80)
        print("异质性区域配置对比 (Heterogeneous Region Configurations)")
        print("="*80)

        print(f"\n{'区域':<25} {'到达率':<12} {'容量':<20} {'服务率范围':<20}")
        print("-"*80)

        for name, config in configs.items():
            arrival_rate = config.base_arrival_rate
            total_capacity = sum(config.layer_capacities)
            service_rates = config.layer_service_rates
            service_range = f"{min(service_rates):.2f}-{max(service_rates):.2f}"

            print(f"{name:<25} {arrival_rate:<12.3f} {total_capacity:<20} {service_range:<20}")

        print("\n" + "="*80)
        print("关键差异分析:")
        print("-"*80)

        base = self.base_config

        print(f"\n📍 Region B (气象异质性):")
        print(f"   - 服务率: {base.layer_service_rates[0]:.2f} → {configs['Region_B_Weather'].layer_service_rates[0]:.2f} (-20%)")
        print(f"   - 影响: 恶劣天气导致无人机飞行速度变慢")

        print(f"\n📍 Region C (流量模式异质性):")
        print(f"   - 到达率: {base.base_arrival_rate:.3f} → {configs['Region_C_Traffic'].base_arrival_rate:.3f} (+50%)")
        print(f"   - 影响: 高峰时段订单激增")

        print(f"\n📍 Region D (监管政策异质性):")
        base_capacity = sum(base.layer_capacities)
        new_capacity = sum(configs['Region_D_Regulation'].layer_capacities)
        print(f"   - 容量: {base_capacity} → {new_capacity} ({(new_capacity-base_capacity)/base_capacity*100:.1f}%)")
        print(f"   - 影响: 限高政策禁止使用部分空域")

        print(f"\n📍 Region E (能耗成本异质性):")
        print(f"   - 高层服务率: {base.layer_service_rates[0]:.2f} → {configs['Region_E_EnergyCost'].layer_service_rates[0]:.2f} (-25%)")
        print(f"   - 影响: 能源成本高，倾向使用低层空域")

        print("\n" + "="*80)
        print("✅ 配置生成完成！")
        print("="*80 + "\n")

    def export_config_table_latex(self) -> str:
        """
        导出LaTeX表格

        用于论文第3.4节
        """
        latex = r"""
\begin{table}[htbp]
\centering
\caption{跨场景异质性配置对比}
\label{tab:heterogeneous_configs}
\begin{tabular}{lcccc}
\hline
\textbf{区域} & \textbf{异质性类型} & \textbf{关键参数修改} & \textbf{到达率} & \textbf{总容量} \\
\hline
"""

        configs = self.get_all_configs()
        config_details = [
            ('Region A', '基线（标准）', '-', 0.25, 23),
            ('Region B', '气象差异', '服务率 $\\times 0.8$', 0.25, 23),
            ('Region C', '流量模式差异', '到达率 $\\times 1.5$', 0.375, 23),
            ('Region D', '监管政策差异', '容量 $-22\\%$', 0.25, 18),
            ('Region E', '能耗成本差异', '高层服务率 $-25\\%$', 0.25, 23),
        ]

        for region, het_type, param_change, arrival, capacity in config_details:
            latex += f"{region} & {het_type} & {param_change} & {arrival:.3f} & {capacity} \\\\\n"

        latex += r"""\hline
\end{tabular}
\end{table}
"""

        return latex


if __name__ == "__main__":
    print("\n🔬 创建异质性区域配置...")

    generator = HeterogeneousRegionConfigs()

    # 打印配置摘要
    generator.print_config_summary()

    # 验证所有配置
    print("\n🔍 验证配置有效性...")
    configs = generator.get_all_configs()

    for name, config in configs.items():
        try:
            config._validate_config()
            print(f"   ✅ {name}: 验证通过")
        except AssertionError as e:
            print(f"   ❌ {name}: 验证失败 - {e}")

    # 导出LaTeX表格
    print("\n📄 LaTeX表格:")
    print(generator.export_config_table_latex())
