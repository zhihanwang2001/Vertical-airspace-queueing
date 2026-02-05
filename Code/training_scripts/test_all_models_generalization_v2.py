"""
Top 3 Models Cross-Region Generalization Test Script V2 - Enhanced

Core Improvements:
1. Not only cumulative reward, but also multi-dimensional system performance metrics
2. Extract key metrics like queue saturation, load rate, stability, throughput
3. More accurately reflect model's real performance in heterogeneous environments

Evaluation Metrics:
- Cumulative Reward
- Average Queue Utilization
- Average Load Rate
- System Throughput
- Stability Score
- Congestion Level
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'rpTransition'))

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
import importlib.util
spec = importlib.util.spec_from_file_location(
    "heterogeneous_configs",
    os.path.join(os.path.dirname(__file__), '..', 'rpTransition', 'heterogeneous_configs.py')
)
heterogeneous_configs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(heterogeneous_configs)

HeterogeneousRegionConfigs = heterogeneous_configs.HeterogeneousRegionConfigs


def test_model_in_region(model, model_type: str, config, region_name: str,
                         n_episodes: int = 10, verbose: bool = True):
    """
    Test model in specified region - Enhanced version (extract multi-dimensional metrics)

    Args:
        model: Loaded model baseline instance
        model_type: Model type ('A2C', 'PPO', 'TD7')
        config: VerticalQueueConfig configuration
        region_name: Region name
        n_episodes: Number of test episodes
        verbose: Whether to print detailed information

    Returns:
        dict: Test results (including multi-dimensional metrics)
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

    # New: System performance metrics
    episode_avg_utilizations = []  # Average queue utilization
    episode_avg_load_rates = []     # Average load rate
    episode_throughputs = []        # Throughput
    episode_stability_scores = []   # Stability score
    episode_max_utilizations = []   # Maximum queue utilization (congestion level)

    episode_details = []

    # Run n_episodes episodes
    for episode in range(n_episodes):
        obs, info = eval_env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        # Collect system metrics within episode
        step_utilizations = []
        step_load_rates = []
        step_stability_scores = []

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

            # Extract system metrics (from info)
            if 'utilization_rates' in info:
                step_utilizations.append(np.mean(info['utilization_rates']))
            if 'load_rates' in info:
                step_load_rates.append(np.mean(info['load_rates']))
            if 'stability_score' in info:
                step_stability_scores.append(info['stability_score'])

            # Prevent infinite loop
            if episode_length >= 1000:
                if verbose:
                    print(f"  Warning: Episode {episode+1} reached maximum step limit (1000)")
                break

        # Calculate episode-level system metrics
        avg_utilization = np.mean(step_utilizations) if step_utilizations else 0.0
        avg_load_rate = np.mean(step_load_rates) if step_load_rates else 0.0
        avg_stability = np.mean(step_stability_scores) if step_stability_scores else 0.0
        max_utilization = np.max(step_utilizations) if step_utilizations else 0.0
        throughput = info.get('throughput', 0.0) if info else 0.0

        # Record results
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_avg_utilizations.append(avg_utilization)
        episode_avg_load_rates.append(avg_load_rate)
        episode_throughputs.append(throughput)
        episode_stability_scores.append(avg_stability)
        episode_max_utilizations.append(max_utilization)

        episode_details.append({
            'episode': episode + 1,
            'reward': float(episode_reward),
            'length': int(episode_length),
            'avg_utilization': float(avg_utilization),
            'avg_load_rate': float(avg_load_rate),
            'throughput': float(throughput),
            'stability_score': float(avg_stability),
            'max_utilization': float(max_utilization)
        })

        if verbose:
            print(f"  Episode {episode+1}/{n_episodes}:")
            print(f"    Reward={episode_reward:.2f}, Length={episode_length}")
            print(f"    Utilization={avg_utilization:.3f}, LoadRate={avg_load_rate:.3f}, Throughput={throughput:.2f}")

    # Calculate statistics
    results = {
        'model_type': model_type,
        'region_name': region_name,
        'n_episodes': n_episodes,

        # Original metrics
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'mean_length': float(np.mean(episode_lengths)),

        # New: System performance metrics
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

        # Detailed data
        'episode_rewards': [float(r) for r in episode_rewards],
        'episode_lengths': [int(l) for l in episode_lengths],
        'episode_details': episode_details,
        'config_summary': base_env.get_config_summary()
    }

    if verbose:
        print(f"\n{model_type} @ {region_name} Test Results:")
        print(f"   Cumulative reward: {results['mean_reward']:.2f} +/- {results['std_reward']:.2f}")
        print(f"   Queue utilization: {results['mean_utilization']:.3f} +/- {results['std_utilization']:.3f}")
        print(f"   Load rate: {results['mean_load_rate']:.3f} +/- {results['std_load_rate']:.3f}")
        print(f"   Throughput: {results['mean_throughput']:.2f} +/- {results['std_throughput']:.2f}")
        print(f"   Stability: {results['mean_stability']:.3f} +/- {results['std_stability']:.3f}")
        print(f"   Max congestion: {results['mean_max_congestion']:.3f} +/- {results['std_max_congestion']:.3f}")

    # Clean up environment
    eval_env.close()

    return results


def main():
    """Main function: Test all 3 models' generalization performance across all heterogeneous regions - Enhanced version"""

    print("\n" + "="*80)
    print("Top 3 Models Cross-Region Generalization Test V2 - Enhanced")
    print("Cross-Region Generalization Test V2 - Enhanced with Multi-Dimensional Metrics")
    print("="*80 + "\n")

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
    print("\nStep 3: Run generalization tests in each region (Enhanced - extract multi-dimensional metrics)")
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

    # ========== Step 4: Summarize results (multi-dimensional) ==========
    print("\n" + "="*80)
    print("Testing completed! Summary of results (multi-dimensional metrics)")
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

    # Table 2: Queue utilization comparison
    print("\n[Table 2] Average Queue Utilization")
    print("-"*90)
    print(f"{'Region':<30} {'A2C':<20} {'PPO':<20} {'TD7':<20}")
    print("-"*90)

    for region_name in all_configs.keys():
        a2c_util = all_results['A2C'][region_name]['mean_utilization']
        ppo_util = all_results['PPO'][region_name]['mean_utilization']
        td7_util = all_results['TD7'][region_name]['mean_utilization']

        print(f"{region_name:<30} {a2c_util:<20.3f} {ppo_util:<20.3f} {td7_util:<20.3f}")

    # Table 3: Load rate comparison
    print("\n[Table 3] Average Load Rate (closer to 1 is better)")
    print("-"*90)
    print(f"{'Region':<30} {'A2C':<20} {'PPO':<20} {'TD7':<20}")
    print("-"*90)

    for region_name in all_configs.keys():
        a2c_load = all_results['A2C'][region_name]['mean_load_rate']
        ppo_load = all_results['PPO'][region_name]['mean_load_rate']
        td7_load = all_results['TD7'][region_name]['mean_load_rate']

        print(f"{region_name:<30} {a2c_load:<20.3f} {ppo_load:<20.3f} {td7_load:<20.3f}")

    # Table 4: System throughput comparison
    print("\n[Table 4] System Throughput (orders/step)")
    print("-"*90)
    print(f"{'Region':<30} {'A2C':<20} {'PPO':<20} {'TD7':<20}")
    print("-"*90)

    for region_name in all_configs.keys():
        a2c_thru = all_results['A2C'][region_name]['mean_throughput']
        ppo_thru = all_results['PPO'][region_name]['mean_throughput']
        td7_thru = all_results['TD7'][region_name]['mean_throughput']

        print(f"{region_name:<30} {a2c_thru:<20.2f} {ppo_thru:<20.2f} {td7_thru:<20.2f}")

    # Table 5: Stability score comparison
    print("\n[Table 5] Stability Score (higher is better)")
    print("-"*90)
    print(f"{'Region':<30} {'A2C':<20} {'PPO':<20} {'TD7':<20}")
    print("-"*90)

    for region_name in all_configs.keys():
        a2c_stab = all_results['A2C'][region_name]['mean_stability']
        ppo_stab = all_results['PPO'][region_name]['mean_stability']
        td7_stab = all_results['TD7'][region_name]['mean_stability']

        print(f"{region_name:<30} {a2c_stab:<20.3f} {ppo_stab:<20.3f} {td7_stab:<20.3f}")

    print("\n" + "-"*80)
    print(f"Total test time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Total episodes: {len(models) * len(all_configs) * n_episodes_per_region}")

    # ========== Step 5: Save results ==========
    print("\nStep 5: Save test results")
    print("-"*80)

    # Create save directory
    save_dir = Path("../../Results/generalization")
    save_dir.mkdir(exist_ok=True)

    # Save detailed results
    results_file = save_dir / "all_models_generalization_results_v2.json"

    full_results = {
        'test_info': {
            'version': 'v2_enhanced',
            'test_date': time.strftime("%Y-%m-%d %H:%M:%S"),
            'n_episodes_per_region': n_episodes_per_region,
            'total_time_seconds': total_time,
            'models_tested': ['A2C', 'PPO', 'TD7'],
            'regions_tested': list(all_configs.keys()),
            'metrics_evaluated': [
                'cumulative_reward', 'queue_utilization', 'load_rate',
                'throughput', 'stability_score', 'max_congestion'
            ]
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

    # Save summary table (CSV format - enhanced version)
    summary_file = save_dir / "all_models_generalization_summary_v2.csv"
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
            'Mean Max Congestion', 'Std Max Congestion'
        ])

        for region_name in all_configs.keys():
            for model_name in ['A2C', 'PPO', 'TD7']:
                res = all_results[model_name][region_name]
                writer.writerow([
                    region_name, model_name,
                    f"{res['mean_reward']:.2f}", f"{res['std_reward']:.2f}",
                    f"{res['mean_utilization']:.4f}", f"{res['std_utilization']:.4f}",
                    f"{res['mean_load_rate']:.4f}", f"{res['std_load_rate']:.4f}",
                    f"{res['mean_throughput']:.2f}", f"{res['std_throughput']:.2f}",
                    f"{res['mean_stability']:.4f}", f"{res['std_stability']:.4f}",
                    f"{res['mean_max_congestion']:.4f}", f"{res['std_max_congestion']:.4f}"
                ])

    print(f"Summary table saved to: {summary_file}")

    print("\n" + "="*80)
    print("All model generalization tests completed (enhanced version)!")
    print("="*80 + "\n")

    print("Key findings (multi-dimensional evaluation):")
    print(f"   Baseline performance (Region A):")
    print(f"     - A2C: {baseline_rewards['A2C']:.2f}")
    print(f"     - PPO: {baseline_rewards['PPO']:.2f}")
    print(f"     - TD7: {baseline_rewards['TD7']:.2f}")

    print("\nNext steps:")
    print("   1. View detailed results: cat generalization_results/all_models_generalization_results_v2.json")
    print("   2. View summary table: cat generalization_results/all_models_generalization_summary_v2.csv")
    print("   3. Analyze multi-dimensional metrics, write paper")


if __name__ == "__main__":
    main()
