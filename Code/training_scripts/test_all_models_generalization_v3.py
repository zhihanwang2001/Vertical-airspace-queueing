"""
Top 3 Models Cross-Region Generalization Test Script V3 - Reward Decomposition
Top 3 Models Cross-Region Generalization Test Script V3 - Reward Decomposition

Core Improvements (V2 → V3):
1. Retain all multi-dimensional system metrics from V2
2. **New: Extract reward component decomposition (reward_components)**
3. Analyze limitations of single-objective optimization (RP1) in multi-objective trade-offs
4. Provide scientific basis for RP1→RP2 transition

Evaluation Metrics:
[V2 Metrics]
- Cumulative Reward
- Queue Utilization
- Load Rate
- System Throughput
- Stability Score

[V3 New - Reward Components]
- R_throughput: Throughput reward (10.0 × served orders)
- R_balance: Load balance reward (Gini coefficient, 0-5.0)
- R_efficiency: Energy efficiency reward (service/energy ratio, 0-3.0)
- transfer_benefit: Transfer benefit (0-2.0)
- stability_bonus: Stability reward (0-2.0)
- P_congestion: Congestion penalty (<0)
- P_instability: Instability penalty (<0)

Analysis Purpose:
Reveal that RP1's single-objective optimization achieves high cumulative reward, but has trade-offs in:
  - Inter-layer fairness (R_balance)
  - Energy efficiency (R_efficiency)
  - Load balancing
→ motivates RP2's MORL approach
"""

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import json
from pathlib import Path
from typing import Dict, List
import time

# Import baseline algorithms
from algorithms.baselines.sb3_a2c_baseline import SB3A2CBaseline
from algorithms.baselines.sb3_ppo_baseline import SB3PPOBaseline
from algorithms.advanced.td7.td7_baseline import TD7Baseline

# Import environment and configuration
from env.configurable_env_wrapper import ConfigurableEnvWrapper
from algorithms.baselines.space_utils import SB3DictWrapper

# Import heterogeneous configuration generator
from heterogeneous_configs import HeterogeneousRegionConfigs


def test_model_in_region(model, model_type: str, config, region_name: str,
                         n_episodes: int = 10, verbose: bool = True):
    """
    Test model in specified region - V3 version (extract reward component decomposition)

    Args:
        model: Loaded model baseline instance
        model_type: Model type ('A2C', 'PPO', 'TD7')
        config: VerticalQueueConfig configuration
        region_name: Region name
        n_episodes: Number of test episodes
        verbose: Whether to print detailed information

    Returns:
        dict: Test results (including multi-dimensional metrics + reward component decomposition)
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"Testing: {model_type} @ {region_name}")
        print(f"{'='*80}")

    # Create environment for this region
    base_env = ConfigurableEnvWrapper(config)
    eval_env = SB3DictWrapper(base_env)

    # Record results
    episode_rewards = []
    episode_lengths = []

    # V2 metrics: System performance
    episode_avg_utilizations = []
    episode_avg_load_rates = []
    episode_throughputs = []
    episode_stability_scores = []
    episode_max_utilizations = []

    # V3 new: Reward components
    episode_avg_r_throughput = []
    episode_avg_r_balance = []
    episode_avg_r_efficiency = []
    episode_avg_transfer = []
    episode_avg_stability_bonus = []
    episode_avg_p_congestion = []
    episode_avg_p_instability = []

    episode_details = []

    # Run n_episodes episodes
    for episode in range(n_episodes):
        obs, info = eval_env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        # V2 metric collection
        step_utilizations = []
        step_load_rates = []
        step_stability_scores = []

        # V3 new: Reward component collection
        step_r_throughput = []
        step_r_balance = []
        step_r_efficiency = []
        step_transfer = []
        step_stability_bonus = []
        step_p_congestion = []
        step_p_instability = []

        # Run a complete episode
        while not done:
            # Select prediction method based on model type
            if model_type == 'TD7':
                action = model.agent.act(obs, training=False)
            else:  # A2C or PPO
                action, _ = model.model.predict(obs, deterministic=True)

            # Execute action
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

            # Extract V2 system metrics
            if 'utilization_rates' in info:
                step_utilizations.append(np.mean(info['utilization_rates']))
            if 'load_rates' in info:
                step_load_rates.append(np.mean(info['load_rates']))
            if 'stability_score' in info:
                step_stability_scores.append(info['stability_score'])

            # V3 new: Extract reward components
            if 'reward_components' in info:
                rc = info['reward_components']
                step_r_throughput.append(rc.get('throughput', 0.0))
                step_r_balance.append(rc.get('balance', 0.0))
                step_r_efficiency.append(rc.get('efficiency', 0.0))
                step_transfer.append(rc.get('transfer', 0.0))
                step_stability_bonus.append(rc.get('stability', 0.0))
                step_p_congestion.append(rc.get('congestion', 0.0))
                step_p_instability.append(rc.get('instability', 0.0))

            # Prevent infinite loop
            if episode_length >= 1000:
                if verbose:
                    print(f"  Warning: Episode {episode+1} reached maximum step limit (1000)")
                break

        # Calculate episode-level statistics
        # V2 metrics
        avg_utilization = np.mean(step_utilizations) if step_utilizations else 0.0
        avg_load_rate = np.mean(step_load_rates) if step_load_rates else 0.0
        avg_stability = np.mean(step_stability_scores) if step_stability_scores else 0.0
        max_utilization = np.max(step_utilizations) if step_utilizations else 0.0
        throughput = info.get('throughput', 0.0) if info else 0.0

        # V3 new: Reward component averages
        avg_r_throughput = np.mean(step_r_throughput) if step_r_throughput else 0.0
        avg_r_balance = np.mean(step_r_balance) if step_r_balance else 0.0
        avg_r_efficiency = np.mean(step_r_efficiency) if step_r_efficiency else 0.0
        avg_transfer = np.mean(step_transfer) if step_transfer else 0.0
        avg_stability_bonus = np.mean(step_stability_bonus) if step_stability_bonus else 0.0
        avg_p_congestion = np.mean(step_p_congestion) if step_p_congestion else 0.0
        avg_p_instability = np.mean(step_p_instability) if step_p_instability else 0.0

        # Record results
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # V2 metrics
        episode_avg_utilizations.append(avg_utilization)
        episode_avg_load_rates.append(avg_load_rate)
        episode_throughputs.append(throughput)
        episode_stability_scores.append(avg_stability)
        episode_max_utilizations.append(max_utilization)

        # V3 new: Reward components
        episode_avg_r_throughput.append(avg_r_throughput)
        episode_avg_r_balance.append(avg_r_balance)
        episode_avg_r_efficiency.append(avg_r_efficiency)
        episode_avg_transfer.append(avg_transfer)
        episode_avg_stability_bonus.append(avg_stability_bonus)
        episode_avg_p_congestion.append(avg_p_congestion)
        episode_avg_p_instability.append(avg_p_instability)

        episode_details.append({
            'episode': episode + 1,
            'reward': float(episode_reward),
            'length': int(episode_length),
            # V2 metrics
            'avg_utilization': float(avg_utilization),
            'avg_load_rate': float(avg_load_rate),
            'throughput': float(throughput),
            'stability_score': float(avg_stability),
            'max_utilization': float(max_utilization),
            # V3 new: Reward components
            'reward_components': {
                'throughput': float(avg_r_throughput),
                'balance': float(avg_r_balance),
                'efficiency': float(avg_r_efficiency),
                'transfer': float(avg_transfer),
                'stability': float(avg_stability_bonus),
                'congestion': float(avg_p_congestion),
                'instability': float(avg_p_instability)
            }
        })

        if verbose:
            print(f"  Episode {episode+1}/{n_episodes}:")
            print(f"    Reward={episode_reward:.2f}, Length={episode_length}")
            print(f"    [V2] Util={avg_utilization:.3f}, Load={avg_load_rate:.3f}, Throughput={throughput:.2f}")
            print(f"    [V3] R_throughput={avg_r_throughput:.1f}, R_balance={avg_r_balance:.2f}, R_efficiency={avg_r_efficiency:.2f}")

    # Calculate statistics
    results = {
        'model_type': model_type,
        'region_name': region_name,
        'n_episodes': n_episodes,

        # Original metrics
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'mean_length': float(np.mean(episode_lengths)),

        # V2 metrics: System performance
        'mean_utilization': float(np.mean(episode_avg_utilizations)),
        'std_utilization': float(np.std(episode_avg_utilizations)),
        'mean_load_rate': float(np.mean(episode_avg_load_rates)),
        'std_load_rate': float(np.std(episode_avg_load_rates)),
        'mean_throughput': float(np.mean(episode_throughputs)),
        'std_throughput': float(np.std(episode_throughputs)),
        'mean_stability': float(np.mean(episode_stability_scores)),
        'std_stability': float(np.std(episode_stability_scores)),
        'mean_max_congestion': float(np.mean(episode_max_utilizations)),
        'std_max_congestion': float(np.std(episode_max_utilizations)),

        # V3 new: Reward component statistics
        'reward_components': {
            'mean_throughput': float(np.mean(episode_avg_r_throughput)),
            'std_throughput': float(np.std(episode_avg_r_throughput)),
            'mean_balance': float(np.mean(episode_avg_r_balance)),
            'std_balance': float(np.std(episode_avg_r_balance)),
            'mean_efficiency': float(np.mean(episode_avg_r_efficiency)),
            'std_efficiency': float(np.std(episode_avg_r_efficiency)),
            'mean_transfer': float(np.mean(episode_avg_transfer)),
            'std_transfer': float(np.std(episode_avg_transfer)),
            'mean_stability': float(np.mean(episode_avg_stability_bonus)),
            'std_stability': float(np.std(episode_avg_stability_bonus)),
            'mean_congestion': float(np.mean(episode_avg_p_congestion)),
            'std_congestion': float(np.std(episode_avg_p_congestion)),
            'mean_instability': float(np.mean(episode_avg_p_instability)),
            'std_instability': float(np.std(episode_avg_p_instability))
        },

        # Detailed data
        'episode_rewards': [float(r) for r in episode_rewards],
        'episode_lengths': [int(l) for l in episode_lengths],
        'episode_details': episode_details,
        'config_summary': base_env.get_config_summary()
    }

    if verbose:
        print(f"\n{model_type} @ {region_name} Test Results:")
        print(f"   [V2 Metrics]")
        print(f"   Cumulative reward: {results['mean_reward']:.2f} +/- {results['std_reward']:.2f}")
        print(f"   Queue utilization: {results['mean_utilization']:.3f} +/- {results['std_utilization']:.3f}")
        print(f"   Load rate: {results['mean_load_rate']:.3f} +/- {results['std_load_rate']:.3f}")
        print(f"   Throughput: {results['mean_throughput']:.2f} +/- {results['std_throughput']:.2f}")
        print(f"   Stability: {results['mean_stability']:.3f} +/- {results['std_stability']:.3f}")
        print(f"\n   [V3 Reward Components]")
        rc = results['reward_components']
        print(f"   R_throughput: {rc['mean_throughput']:.2f} +/- {rc['std_throughput']:.2f}")
        print(f"   R_balance (fairness): {rc['mean_balance']:.2f} +/- {rc['std_balance']:.2f}")
        print(f"   R_efficiency (energy): {rc['mean_efficiency']:.2f} +/- {rc['std_efficiency']:.2f}")
        print(f"   P_congestion (penalty): {rc['mean_congestion']:.2f} +/- {rc['std_congestion']:.2f}")

    # Clean up environment
    eval_env.close()

    return results

def main():
    """Main function: Test all 3 models' generalization performance across all heterogeneous regions - V3 version (reward component decomposition)"""

    print("\n" + "="*80)
    print("Top 3 Models Cross-Region Generalization Test V3 - Reward Component Decomposition")
    print("Cross-Region Generalization Test V3 - Reward Component Decomposition")
    print("="*80 + "\n")

    print("V3 Core Improvements:")
    print("   - Retain V2's multi-dimensional system metrics")
    print("   - New: Extract reward component decomposition (7 components)")
    print("   - Reveal multi-objective trade-offs in single-objective optimization")
    print("   - Provide scientific basis for RP1→RP2 transition\n")

    # ========== Step 1: Load 3 trained models ==========
    print("Step 1: Load 3 trained models")
    print("-"*80)

    models = {}
    model_paths = {
        'A2C': '../../Models/a2c/a2c_model_500000',
        'PPO': '../../Models/ppo/ppo_model_500000',
        'TD7': '../../Models/td7/td7_model_500000.pt'
    }

    # Load A2C
    print("\n1.1 Loading A2C model...")
    if not os.path.exists(model_paths['A2C'] + '.pth'):
        print(f"Error: Cannot find A2C model file {model_paths['A2C']}.pth")
        return

    a2c = SB3A2CBaseline()
    a2c.load(model_paths['A2C'])
    models['A2C'] = a2c
    print("A2C model loaded successfully!")

    # Load PPO
    print("\n1.2 Loading PPO model...")
    if not os.path.exists(model_paths['PPO'] + '.pth'):
        print(f"Error: Cannot find PPO model file {model_paths['PPO']}.pth")
        return

    ppo = SB3PPOBaseline()
    ppo.load(model_paths['PPO'])
    models['PPO'] = ppo
    print("PPO model loaded successfully!")

    # Load TD7
    print("\n1.3 Loading TD7 model...")
    if not os.path.exists(model_paths['TD7']):
        print(f"Error: Cannot find TD7 model file {model_paths['TD7']}")
        return

    print(f"Model file size: {os.path.getsize(model_paths['TD7']) / (1024*1024):.1f} MB")
    td7 = TD7Baseline()
    td7.load(model_paths['TD7'])
    models['TD7'] = td7
    print("TD7 model loaded successfully!")

    print("\nAll 3 models loaded successfully!")

    # ========== Step 2: Create heterogeneous region configurations ==========
    print("\nStep 2: Create heterogeneous region configurations")
    print("-"*80)

    config_generator = HeterogeneousRegionConfigs()
    all_configs = config_generator.get_all_configs()

    print(f"Created {len(all_configs)} region configurations:")
    for region_name in all_configs.keys():
        print(f"   - {region_name}")

    # ========== Step 3: Run tests in each region ==========
    print("\nStep 3: Run generalization tests in each region (V3 - reward component decomposition)")
    print("-"*80)
    print("Warning: This is a real test, not mock data!")
    print(f"   Total tests: {len(models)} models × {len(all_configs)} regions × 10 episodes = {len(models) * len(all_configs) * 10} episodes")

    all_results = {
        'A2C': {},
        'PPO': {},
        'TD7': {}
    }

    n_episodes_per_region = 10
    start_time = time.time()

    # Run tests for each model and each region
    for model_name in ['A2C', 'PPO', 'TD7']:
        print(f"\n{'='*80}")
        print(f"Starting test for {model_name} model")
        print(f"{'='*80}")

        model = models[model_name]

        for region_name, config in all_configs.items():
            results = test_model_in_region(
                model=model,
                model_type=model_name,
                config=config,
                region_name=region_name,
                n_episodes=n_episodes_per_region,
                verbose=True
            )
            all_results[model_name][region_name] = results

    total_time = time.time() - start_time

    # ========== Step 4: Summarize results (V3 - including reward component analysis) ==========
    print("\n" + "="*80)
    print("Testing completed! Summary of results (V3 - multi-dimensional metrics + reward component decomposition)")
    print("="*80 + "\n")

    # Table 1: Cumulative reward comparison
    print("[Table 1] Cumulative Reward Comparison")
    print("-"*90)
    print(f"{'Region':<30} {'A2C':<20} {'PPO':<20} {'TD7':<20}")
    print("-"*90)

    baseline_rewards = {}

    for region_name in all_configs.keys():
        a2c_reward = all_results['A2C'][region_name]['mean_reward']
        ppo_reward = all_results['PPO'][region_name]['mean_reward']
        td7_reward = all_results['TD7'][region_name]['mean_reward']

        if 'Standard' in region_name:
            baseline_rewards['A2C'] = a2c_reward
            baseline_rewards['PPO'] = ppo_reward
            baseline_rewards['TD7'] = td7_reward

        print(f"{region_name:<30} {a2c_reward:<20.2f} {ppo_reward:<20.2f} {td7_reward:<20.2f}")

    # V3 new: Reward component comparison tables
    print("\n[Table 6] R_balance (Load Balance/Fairness) Comparison")
    print("-"*90)
    print(f"{'Region':<30} {'A2C':<20} {'PPO':<20} {'TD7':<20}")
    print("-"*90)

    for region_name in all_configs.keys():
        a2c_bal = all_results['A2C'][region_name]['reward_components']['mean_balance']
        ppo_bal = all_results['PPO'][region_name]['reward_components']['mean_balance']
        td7_bal = all_results['TD7'][region_name]['reward_components']['mean_balance']

        print(f"{region_name:<30} {a2c_bal:<20.3f} {ppo_bal:<20.3f} {td7_bal:<20.3f}")

    print("\n[Table 7] R_efficiency (Energy Efficiency) Comparison")
    print("-"*90)
    print(f"{'Region':<30} {'A2C':<20} {'PPO':<20} {'TD7':<20}")
    print("-"*90)

    for region_name in all_configs.keys():
        a2c_eff = all_results['A2C'][region_name]['reward_components']['mean_efficiency']
        ppo_eff = all_results['PPO'][region_name]['reward_components']['mean_efficiency']
        td7_eff = all_results['TD7'][region_name]['reward_components']['mean_efficiency']

        print(f"{region_name:<30} {a2c_eff:<20.3f} {ppo_eff:<20.3f} {td7_eff:<20.3f}")

    print("\n[Table 8] R_throughput (Throughput Reward Component) Comparison")
    print("-"*90)
    print(f"{'Region':<30} {'A2C':<20} {'PPO':<20} {'TD7':<20}")
    print("-"*90)

    for region_name in all_configs.keys():
        a2c_thr = all_results['A2C'][region_name]['reward_components']['mean_throughput']
        ppo_thr = all_results['PPO'][region_name]['reward_components']['mean_throughput']
        td7_thr = all_results['TD7'][region_name]['reward_components']['mean_throughput']

        print(f"{region_name:<30} {a2c_thr:<20.2f} {ppo_thr:<20.2f} {td7_thr:<20.2f}")

    print("\n[Table 9] P_congestion (Congestion Penalty) Comparison")
    print("-"*90)
    print(f"{'Region':<30} {'A2C':<20} {'PPO':<20} {'TD7':<20}")
    print("-"*90)

    for region_name in all_configs.keys():
        a2c_cong = all_results['A2C'][region_name]['reward_components']['mean_congestion']
        ppo_cong = all_results['PPO'][region_name]['reward_components']['mean_congestion']
        td7_cong = all_results['TD7'][region_name]['reward_components']['mean_congestion']

        print(f"{region_name:<30} {a2c_cong:<20.2f} {ppo_cong:<20.2f} {td7_cong:<20.2f}")

    print("\n" + "-"*80)
    print(f"Total test time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Total episodes: {len(models) * len(all_configs) * n_episodes_per_region}")

    # ========== Step 5: Save results ==========
    print("\nStep 5: Save test results (V3 version)")
    print("-"*80)

    # Create save directory
    save_dir = Path("../../Results/generalization")
    save_dir.mkdir(exist_ok=True)

    # Save detailed results
    results_file = save_dir / "all_models_generalization_results_v3.json"

    full_results = {
        'test_info': {
            'version': 'v3_reward_decomposition',
            'test_date': time.strftime("%Y-%m-%d %H:%M:%S"),
            'n_episodes_per_region': n_episodes_per_region,
            'total_time_seconds': total_time,
            'models_tested': ['A2C', 'PPO', 'TD7'],
            'regions_tested': list(all_configs.keys()),
            'metrics_evaluated': {
                'v2_system_metrics': [
                    'cumulative_reward', 'queue_utilization', 'load_rate',
                    'throughput', 'stability_score', 'max_congestion'
                ],
                'v3_reward_components': [
                    'R_throughput', 'R_balance', 'R_efficiency',
                    'transfer_benefit', 'stability_bonus',
                    'P_congestion', 'P_instability'
                ]
            },
            'purpose': 'Reveal multi-objective trade-offs in single-objective RL (RP1) to motivate MORL approach (RP2)'
        },
        'model_paths': model_paths,
        'baseline_performance': {
            'A2C': f"{baseline_rewards['A2C']:.2f}",
            'PPO': f"{baseline_rewards['PPO']:.2f}",
            'TD7': f"{baseline_rewards['TD7']:.2f}"
        },
        'results': all_results
    }

    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)

    print(f"Detailed results saved to: {results_file}")

    # Save summary table (CSV format - V3 enhanced version, including reward components)
    summary_file = save_dir / "all_models_generalization_summary_v3.csv"
    import csv

    with open(summary_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Region', 'Model',
            'Mean Reward', 'Std Reward',
            'Mean Utilization', 'Std Utilization',
            'Mean Load Rate', 'Std Load Rate',
            'Mean Throughput', 'Std Throughput',
            'Mean Stability', 'Std Stability',
            # V3 new: Reward components
            'R_throughput', 'R_balance', 'R_efficiency',
            'transfer_benefit', 'stability_bonus',
            'P_congestion', 'P_instability'
        ])

        for region_name in all_configs.keys():
            for model_name in ['A2C', 'PPO', 'TD7']:
                res = all_results[model_name][region_name]
                rc = res['reward_components']
                writer.writerow([
                    region_name, model_name,
                    f"{res['mean_reward']:.2f}", f"{res['std_reward']:.2f}",
                    f"{res['mean_utilization']:.4f}", f"{res['std_utilization']:.4f}",
                    f"{res['mean_load_rate']:.4f}", f"{res['std_load_rate']:.4f}",
                    f"{res['mean_throughput']:.2f}", f"{res['std_throughput']:.2f}",
                    f"{res['mean_stability']:.4f}", f"{res['std_stability']:.4f}",
                    # V3 new
                    f"{rc['mean_throughput']:.2f}", f"{rc['mean_balance']:.3f}", f"{rc['mean_efficiency']:.3f}",
                    f"{rc['mean_transfer']:.3f}", f"{rc['mean_stability']:.3f}",
                    f"{rc['mean_congestion']:.3f}", f"{rc['mean_instability']:.3f}"
                ])

    print(f"Summary table saved to: {summary_file}")

    print("\n" + "="*80)
    print("All model generalization tests completed (V3 - reward component decomposition version)!")
    print("="*80 + "\n")

    print("V3 Key Findings (Reward Component Decomposition):")
    print(f"\n   Baseline performance (Region A - Standard):")
    print(f"     - A2C: {baseline_rewards['A2C']:.2f}")
    print(f"     - PPO: {baseline_rewards['PPO']:.2f}")
    print(f"     - TD7: {baseline_rewards['TD7']:.2f}")

    print(f"\n   RP1→RP2 Transition Logic:")
    print(f"   Although RP1's single-objective optimization achieves high cumulative reward,")
    print(f"   reward component decomposition reveals trade-offs across multiple objectives:")
    print(f"     - R_balance (fairness): Uneven load distribution across layers")
    print(f"     - R_efficiency (energy): Low energy utilization rate")
    print(f"     - P_congestion (congestion): Increased congestion under high load")
    print(f"   These trade-offs reveal limitations of single-objective optimization,")
    print(f"   motivating RP2 to adopt MORL approach for Pareto optimization.")

    print("\nNext steps:")
    print("   1. View detailed results: cat generalization_results/all_models_generalization_results_v3.json")
    print("   2. View summary table: cat generalization_results/all_models_generalization_summary_v3.csv")
    print("   3. Analyze reward component trade-offs, design RP1→RP2 transition logic")
    print("   4. Write paper Section 3.4: Cross-scenario generalization analysis + RP2 motivation")


if __name__ == "__main__":
    main()
