"""
Heterogeneous Region Configurations

Creates 4 test regions for Section 3.4 "Cross-Scenario Generalization Analysis":
- Region B: Weather heterogeneity (adverse weather conditions)
- Region C: Traffic pattern heterogeneity (peak hours)
- Region D: Regulatory policy heterogeneity (altitude restrictions)
- Region E: Energy cost heterogeneity (energy-sensitive operations)
"""

import copy
from env.config import VerticalQueueConfig


class HeterogeneousRegionConfigs:
    """Heterogeneous region configuration generator"""

    def __init__(self):
        """Initialize baseline configuration"""
        self.base_config = VerticalQueueConfig()  # Region A (Standard)

    def create_region_b_weather(self) -> VerticalQueueConfig:
        """
        Region B: Weather heterogeneity

        Scenario: Adverse weather (strong wind/rain) reduces service rates

        Modifications:
        - Service rate × 0.8 (20% reduction across all layers)
        - Minimum wait time × 1.25 (slower transfers)

        Expected impact: Drones affected by weather, slower ascent/descent speeds
        """
        config = copy.deepcopy(self.base_config)

        # Service rate reduced by 20% (adverse weather impact)
        config.layer_service_rates = [rate * 0.8 for rate in self.base_config.layer_service_rates]

        # Minimum wait time increased by 25%
        config.min_wait_times = [int(t * 1.25) for t in self.base_config.min_wait_times]

        # Update random seed for reproducibility
        config.random_seed = 43

        return config

    def create_region_c_traffic(self) -> VerticalQueueConfig:
        """
        Region C: Traffic pattern heterogeneity

        Scenario: Peak hours with significantly increased order arrival rate

        Modifications:
        - Base arrival rate × 1.5 (0.25 → 0.375)
        - Arrival weight redistribution (more orders to mid-low layers)

        Expected impact: Increased system load, approaching stability boundary
        """
        config = copy.deepcopy(self.base_config)

        # Arrival rate increased by 50%
        config.base_arrival_rate = self.base_config.base_arrival_rate * 1.5  # 0.25 → 0.375

        # Arrival weight redistribution (peak hours allocate more orders to mid-low layers)
        # Original: [0.3, 0.25, 0.2, 0.15, 0.1] (L5→L1)
        # New distribution: [0.2, 0.2, 0.25, 0.2, 0.15] (more uniform, increased pressure on mid-low layers)
        config.arrival_weights = [0.2, 0.2, 0.25, 0.2, 0.15]

        config.random_seed = 44

        return config

    def create_region_d_regulation(self) -> VerticalQueueConfig:
        """
        Region D: Regulatory policy heterogeneity

        Scenario: Altitude restriction policy prohibiting use of top 2 layers (100m, 80m)

        Modifications:
        - Upper layer capacity reduced (L5: 8→5, L4: 6→4)
        - Arrival weights redistributed to lower layers

        Expected impact: 20% reduction in available airspace, increased congestion in lower layers
        """
        config = copy.deepcopy(self.base_config)

        # Capacity constraints (upper layers restricted)
        # Original: [8, 6, 4, 3, 2] (L5→L1)
        # New configuration: [5, 4, 4, 3, 2] (upper layers reduced, total capacity 23→18, -22%)
        config.layer_capacities = [5, 4, 4, 3, 2]

        # Arrival weight redistribution (drones prefer lower layers)
        # Original: [0.3, 0.25, 0.2, 0.15, 0.1]
        # New configuration: [0.15, 0.20, 0.25, 0.25, 0.15] (pressure shifted to mid-low layers)
        config.arrival_weights = [0.15, 0.20, 0.25, 0.25, 0.15]

        config.random_seed = 45

        return config

    def create_region_e_energy_cost(self) -> VerticalQueueConfig:
        """
        Region E: Energy cost heterogeneity

        Scenario: Energy price-sensitive region where high-altitude flight costs more

        Modifications:
        - Implemented implicitly through reward function modification (different reward wrapper used during testing)
        - For configuration consistency, we modify service rates to simulate "reluctance to use upper layers" effect

        Expected impact: Policy prefers lower layers, leading to lower layer overload
        """
        config = copy.deepcopy(self.base_config)

        # Simulate energy sensitivity: reduce "attractiveness" of upper layers
        # Method: Reduce upper layer service rates (simulating "can serve, but high cost makes it undesirable")
        # Original: [1.2, 1.0, 0.8, 0.6, 0.4] (L5→L1)
        # New configuration: [0.9, 0.9, 0.8, 0.7, 0.5] (upper layer service rates reduced more)
        config.layer_service_rates = [0.9, 0.9, 0.8, 0.7, 0.5]

        # Arrival weights also skewed toward lower layers
        config.arrival_weights = [0.15, 0.18, 0.22, 0.25, 0.20]

        config.random_seed = 46

        return config

    def get_all_configs(self) -> dict:
        """
        Get all region configurations

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
        """Print comparison summary of all configurations"""
        configs = self.get_all_configs()

        print("\n" + "="*80)
        print("Heterogeneous Region Configuration Comparison")
        print("="*80)

        print(f"\n{'Region':<25} {'Arrival Rate':<12} {'Capacity':<20} {'Service Rate Range':<20}")
        print("-"*80)

        for name, config in configs.items():
            arrival_rate = config.base_arrival_rate
            total_capacity = sum(config.layer_capacities)
            service_rates = config.layer_service_rates
            service_range = f"{min(service_rates):.2f}-{max(service_rates):.2f}"

            print(f"{name:<25} {arrival_rate:<12.3f} {total_capacity:<20} {service_range:<20}")

        print("\n" + "="*80)
        print("Key Difference Analysis:")
        print("-"*80)

        base = self.base_config

        print(f"\n📍 Region B (Weather Heterogeneity):")
        print(f"   - Service rate: {base.layer_service_rates[0]:.2f} → {configs['Region_B_Weather'].layer_service_rates[0]:.2f} (-20%)")
        print(f"   - Impact: Adverse weather causes slower drone flight speeds")

        print(f"\n📍 Region C (Traffic Pattern Heterogeneity):")
        print(f"   - Arrival rate: {base.base_arrival_rate:.3f} → {configs['Region_C_Traffic'].base_arrival_rate:.3f} (+50%)")
        print(f"   - Impact: Peak hour order surge")

        print(f"\n📍 Region D (Regulatory Policy Heterogeneity):")
        base_capacity = sum(base.layer_capacities)
        new_capacity = sum(configs['Region_D_Regulation'].layer_capacities)
        print(f"   - Capacity: {base_capacity} → {new_capacity} ({(new_capacity-base_capacity)/base_capacity*100:.1f}%)")
        print(f"   - Impact: Altitude restriction policy prohibits use of some airspace")

        print(f"\n📍 Region E (Energy Cost Heterogeneity):")
        print(f"   - Upper layer service rate: {base.layer_service_rates[0]:.2f} → {configs['Region_E_EnergyCost'].layer_service_rates[0]:.2f} (-25%)")
        print(f"   - Impact: High energy costs, preference for lower altitude airspace")

        print("\n" + "="*80)
        print("✅ Configuration generation complete!")
        print("="*80 + "\n")

    def export_config_table_latex(self) -> str:
        """
        Export LaTeX table

        For use in paper Section 3.4
        """
        latex = r"""
\begin{table}[htbp]
\centering
\caption{Cross-Scenario Heterogeneous Configuration Comparison}
\label{tab:heterogeneous_configs}
\begin{tabular}{lcccc}
\hline
\textbf{Region} & \textbf{Heterogeneity Type} & \textbf{Key Parameter Modification} & \textbf{Arrival Rate} & \textbf{Total Capacity} \\
\hline
"""

        configs = self.get_all_configs()
        config_details = [
            ('Region A', 'Baseline (Standard)', '-', 0.25, 23),
            ('Region B', 'Weather Variation', 'Service rate $\\times 0.8$', 0.25, 23),
            ('Region C', 'Traffic Pattern Variation', 'Arrival rate $\\times 1.5$', 0.375, 23),
            ('Region D', 'Regulatory Policy Variation', 'Capacity $-22\\%$', 0.25, 18),
            ('Region E', 'Energy Cost Variation', 'Upper layer service rate $-25\\%$', 0.25, 23),
        ]

        for region, het_type, param_change, arrival, capacity in config_details:
            latex += f"{region} & {het_type} & {param_change} & {arrival:.3f} & {capacity} \\\\\n"

        latex += r"""\hline
\end{tabular}
\end{table}
"""

        return latex


if __name__ == "__main__":
    print("\n🔬 Creating heterogeneous region configurations...")

    generator = HeterogeneousRegionConfigs()

    # Print configuration summary
    generator.print_config_summary()

    # Validate all configurations
    print("\n🔍 Validating configuration validity...")
    configs = generator.get_all_configs()

    for name, config in configs.items():
        try:
            config._validate_config()
            print(f"   ✅ {name}: Validation passed")
        except AssertionError as e:
            print(f"   ❌ {name}: Validation failed - {e}")

    # Export LaTeX table
    print("\n📄 LaTeX table:")
    print(generator.export_config_table_latex())
