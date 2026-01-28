"""
Top 3 Models Cross-Region Generalization Test Script
Top 3 Models Cross-Region Generalization Test Script

Core Objective: Verify the generalization ability of Top 3 models (A2C, PPO, TD7) across different heterogeneous regions
Important: This is not a mock test, uses real environment execution and model inference!

Test Logic:
1. Load 3 trained models
   - A2C: ./models/a2c/a2c_model_500000.pth (4392.86 ± 145.42)
   - PPO: ./models/ppo/ppo_model_500000.pth (4419.98 ± 135.71)
   - TD7: ./models/td7/td7_model_500000.pt  (4351.84 from RP1)
2. Test in 5 different heterogeneous regions
3. Run 10 episodes per region to obtain real performance data
4. Record detailed performance metrics and environment configurations
5. Compare generalization ability of the 3 models
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
    Test model in specified region

    Args:
        model: Loaded model baseline instance
        model_type: Model type ('A2C', 'PPO', 'TD7')
        config: VerticalQueueConfig configuration
        region_name: Region name
        n_episodes: Number of test episodes
        verbose: Whether to print detailed information

    Returns:
        dict: Test results
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
    episode_details = []

    # Run n_episodes episodes
    for episode in range(n_episodes):
        obs, info = eval_env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

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

            # Prevent infinite loop
            if episode_length >= 1000:
                if verbose:
                    print(f"  Warning: Episode {episode+1} reached maximum step limit (1000)")
                break

        # Record results
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_details.append({
            'episode': episode + 1,
            'reward': float(episode_reward),
            'length': int(episode_length)
        })

        if verbose:
            print(f"  Episode {episode+1}/{n_episodes}: Reward = {episode_reward:.2f}, Length = {episode_length}")

    # Calculate statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)

    results = {
        'model_type': model_type,
        'region_name': region_name,
        'n_episodes': n_episodes,
        'mean_reward': float(mean_reward),
        'std_reward': float(std_reward),
        'mean_length': float(mean_length),
        'episode_rewards': [float(r) for r in episode_rewards],
        'episode_lengths': [int(l) for l in episode_lengths],
        'episode_details': episode_details,
        'config_summary': base_env.get_config_summary()
    }

    if verbose:
        print(f"\n{model_type} @ {region_name} Test Results:")
        print(f"   Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        print(f"   Mean length: {mean_length:.1f}")

    # Clean up environment
    eval_env.close()

    return results


def main():
    """Main function: Test all 3 models' generalization performance across all heterogeneous regions"""

    print("\n" + "="*80)
    print("Top 3 Models Cross-Region Generalization Test")
    print("Cross-Region Generalization Test for Top 3 Models (A2C, PPO, TD7)")
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
    print("\nStep 3: Run generalization tests in each region")
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

    # ========== Step 4: Summarize results ==========
    print("\n" + "="*80)
    print("Testing completed! Summary of results")
    print("="*80 + "\n")

    # Print performance comparison across models and regions
    print(f"{'Region':<30} {'A2C':<20} {'PPO':<20} {'TD7':<20}")
    print("-"*90)

    baseline_rewards = {}  # Record each model's performance in baseline region

    for region_name in all_configs.keys():
        a2c_reward = all_results['A2C'][region_name]['mean_reward']
        ppo_reward = all_results['PPO'][region_name]['mean_reward']
        td7_reward = all_results['TD7'][region_name]['mean_reward']

        # Record baseline performance
        if 'Standard' in region_name:
            baseline_rewards['A2C'] = a2c_reward
            baseline_rewards['PPO'] = ppo_reward
            baseline_rewards['TD7'] = td7_reward

        print(f"{region_name:<30} {a2c_reward:<20.2f} {ppo_reward:<20.2f} {td7_reward:<20.2f}")

    # Print performance degradation percentage
    print("\n" + "="*80)
    print("Performance Degradation Percentage (relative to Region A Baseline)")
    print("="*80 + "\n")

    print(f"{'Region':<30} {'A2C':<20} {'PPO':<20} {'TD7':<20}")
    print("-"*90)

    for region_name in all_configs.keys():
        if 'Standard' in region_name:
            print(f"{region_name:<30} {'0.0%':<20} {'0.0%':<20} {'0.0%':<20}")
        else:
            a2c_diff = ((all_results['A2C'][region_name]['mean_reward'] - baseline_rewards['A2C'])
                       / baseline_rewards['A2C'] * 100)
            ppo_diff = ((all_results['PPO'][region_name]['mean_reward'] - baseline_rewards['PPO'])
                       / baseline_rewards['PPO'] * 100)
            td7_diff = ((all_results['TD7'][region_name]['mean_reward'] - baseline_rewards['TD7'])
                       / baseline_rewards['TD7'] * 100)

            print(f"{region_name:<30} {a2c_diff:+.1f}%{' ':<15} {ppo_diff:+.1f}%{' ':<15} {td7_diff:+.1f}%")

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
    results_file = save_dir / "all_models_generalization_results.json"

    full_results = {
        'test_info': {
            'test_date': time.strftime("%Y-%m-%d %H:%M:%S"),
            'n_episodes_per_region': n_episodes_per_region,
            'total_time_seconds': total_time,
            'models_tested': ['A2C', 'PPO', 'TD7'],
            'regions_tested': list(all_configs.keys())
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

    # Save summary table (CSV format)
    summary_file = save_dir / "all_models_generalization_summary.csv"
    import csv

    with open(summary_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Region', 'A2C Mean', 'A2C Std', 'PPO Mean', 'PPO Std',
                        'TD7 Mean', 'TD7 Std', 'Best Model'])

        for region_name in all_configs.keys():
            a2c_res = all_results['A2C'][region_name]
            ppo_res = all_results['PPO'][region_name]
            td7_res = all_results['TD7'][region_name]

            # Find best model
            best_reward = max(a2c_res['mean_reward'], ppo_res['mean_reward'], td7_res['mean_reward'])
            if a2c_res['mean_reward'] == best_reward:
                best_model = 'A2C'
            elif ppo_res['mean_reward'] == best_reward:
                best_model = 'PPO'
            else:
                best_model = 'TD7'

            writer.writerow([
                region_name,
                f"{a2c_res['mean_reward']:.2f}",
                f"{a2c_res['std_reward']:.2f}",
                f"{ppo_res['mean_reward']:.2f}",
                f"{ppo_res['std_reward']:.2f}",
                f"{td7_res['mean_reward']:.2f}",
                f"{td7_res['std_reward']:.2f}",
                best_model
            ])

    print(f"Summary table saved to: {summary_file}")

    # Save generalization scores for each model
    generalization_file = save_dir / "generalization_ranking.txt"

    with open(generalization_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("Cross-Region Generalization Performance Ranking\n")
        f.write("Cross-Region Generalization Performance Ranking\n")
        f.write("="*80 + "\n\n")

        # Calculate average performance degradation
        avg_drop = {}
        for model_name in ['A2C', 'PPO', 'TD7']:
            drops = []
            for region_name in all_configs.keys():
                if 'Standard' not in region_name:
                    reward = all_results[model_name][region_name]['mean_reward']
                    baseline = baseline_rewards[model_name]
                    drop_pct = ((reward - baseline) / baseline) * 100
                    drops.append(drop_pct)
            avg_drop[model_name] = np.mean(drops)

        # Sort (smaller drop is better)
        ranking = sorted(avg_drop.items(), key=lambda x: x[1], reverse=True)

        f.write("Average performance degradation (smaller is better, indicates stronger generalization):\n")
        f.write("-"*80 + "\n")
        for rank, (model_name, drop) in enumerate(ranking, 1):
            f.write(f"{rank}. {model_name}: {drop:+.2f}%\n")

        f.write("\n" + "="*80 + "\n")
        f.write("Best model per region:\n")
        f.write("-"*80 + "\n")

        for region_name in all_configs.keys():
            rewards = {
                'A2C': all_results['A2C'][region_name]['mean_reward'],
                'PPO': all_results['PPO'][region_name]['mean_reward'],
                'TD7': all_results['TD7'][region_name]['mean_reward']
            }
            best = max(rewards.items(), key=lambda x: x[1])
            f.write(f"{region_name}: {best[0]} ({best[1]:.2f})\n")

    print(f"Generalization ranking saved to: {generalization_file}")

    print("\n" + "="*80)
    print("All model generalization tests completed!")
    print("="*80 + "\n")

    print("Key findings:")
    print(f"   Baseline performance (Region A):")
    print(f"     - A2C: {baseline_rewards['A2C']:.2f}")
    print(f"     - PPO: {baseline_rewards['PPO']:.2f}")
    print(f"     - TD7: {baseline_rewards['TD7']:.2f}")

    print(f"\n   Average performance degradation:")
    for rank, (model_name, drop) in enumerate(ranking, 1):
        print(f"     {rank}. {model_name}: {drop:+.2f}% {'(strongest generalization)' if rank == 1 else ''}")

    print("\nNext steps:")
    print("   1. View detailed results: cat generalization_results/all_models_generalization_results.json")
    print("   2. View summary table: cat generalization_results/all_models_generalization_summary.csv")
    print("   3. View generalization ranking: cat generalization_results/generalization_ranking.txt")
    print("   4. Create visualization charts")


if __name__ == "__main__":
    main()
