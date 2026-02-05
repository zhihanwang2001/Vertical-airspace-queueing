"""
Run missing HCA2C experiments locally
Missing: seed45 and seed46 with load 7.0
"""
import sys
import os
import json
import time
from pathlib import Path

# Add Code directory to path
sys.path.insert(0, str(Path(__file__).parent / "Code"))

from algorithms.hca2c.hca2c_baseline import HCA2CBaseline
from algorithms.baselines.sb3_a2c_baseline import SB3A2CBaseline
from algorithms.baselines.sb3_ppo_baseline import SB3PPOBaseline

# Configuration
CONFIG = {
    "hca2c_timesteps": 500000,
    "baseline_timesteps": 100000,
    "seeds": [45, 46],  # Missing seeds
    "load_multipliers": [7.0],  # Missing load
    "eval_episodes": 30,
}

def run_experiment(algo_name, algo_class, timesteps, seed, load, output_dir):
    """Run single experiment"""
    print(f"\n{'='*70}")
    print(f"Running: {algo_name} | Seed={seed} | Load={load}x | Steps={timesteps}")
    print(f"{'='*70}")
    
    # Create model
    if algo_name == "HCA2C":
        model = algo_class({
            "seed": seed,
            "verbose": 1,
            "load_multiplier": load,
        })
        model.setup_env(load_multiplier=load)
    else:
        model = algo_class({"seed": seed, "verbose": 1})
        model.setup_env(load_multiplier=load)
    
    # Train
    print(f"\n[Training] {algo_name} for {timesteps:,} steps...")
    start_time = time.time()
    model.train(total_timesteps=timesteps)
    train_time = time.time() - start_time
    print(f"Training completed in {train_time:.1f}s ({train_time/60:.1f} min)")
    
    # Evaluate
    print(f"\n[Evaluation] Running {CONFIG['eval_episodes']} episodes...")
    eval_results = model.evaluate(n_episodes=CONFIG['eval_episodes'], verbose=False)
    
    # Save results
    results = {
        "algorithm": algo_name,
        "seed": seed,
        "load_multiplier": load,
        "timesteps": timesteps,
        "train_time": train_time,
        "mean_reward": eval_results["mean_reward"],
        "std_reward": eval_results["std_reward"],
        "mean_length": eval_results["mean_length"],
        "crash_rate": eval_results.get("crash_rate", 0.0),
        "episode_rewards": eval_results.get("episode_rewards", []),
        "episode_lengths": eval_results.get("episode_lengths", []),
    }
    
    # Save JSON
    output_file = output_dir / f"{algo_name}_seed{seed}_load{load}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save model
    model_file = output_dir / f"{algo_name}_seed{seed}_load{load}_model.zip"
    model.save(str(model_file))
    
    print(f"\n[Results] {algo_name}:")
    print(f"  Mean Reward: {results['mean_reward']:.1f} ± {results['std_reward']:.1f}")
    print(f"  Mean Length: {results['mean_length']:.1f}")
    print(f"  Crash Rate: {results['crash_rate']:.2%}")
    print(f"  Saved to: {output_file}")
    
    return results

def main():
    """Run missing experiments"""
    print("\n" + "="*70)
    print("Running Missing HCA2C Experiments Locally")
    print("="*70)
    print(f"Seeds: {CONFIG['seeds']}")
    print(f"Load: {CONFIG['load_multipliers']}")
    print(f"Total runs: {3 * len(CONFIG['seeds']) * len(CONFIG['load_multipliers'])}")
    print("="*70)
    
    # Create output directory
    output_dir = Path("Data/hca2c_final_comparison_local")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Algorithm configurations
    algorithms = [
        ("HCA2C", HCA2CBaseline, CONFIG["hca2c_timesteps"]),
        ("A2C", SB3A2CBaseline, CONFIG["baseline_timesteps"]),
        ("PPO", SB3PPOBaseline, CONFIG["baseline_timesteps"]),
    ]
    
    # Run experiments
    all_results = []
    total_runs = len(algorithms) * len(CONFIG["seeds"]) * len(CONFIG["load_multipliers"])
    current_run = 0
    
    for seed in CONFIG["seeds"]:
        for load in CONFIG["load_multipliers"]:
            for algo_name, algo_class, timesteps in algorithms:
                current_run += 1
                print(f"\n{'#'*70}")
                print(f"Progress: {current_run}/{total_runs}")
                print(f"{'#'*70}")
                
                try:
                    results = run_experiment(
                        algo_name, algo_class, timesteps, 
                        seed, load, output_dir
                    )
                    all_results.append(results)
                except Exception as e:
                    print(f"\n[ERROR] Failed to run {algo_name} seed {seed}: {e}")
                    import traceback
                    traceback.print_exc()
    
    # Save summary
    summary_file = output_dir / "experiment_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*70)
    print("EXPERIMENTS COMPLETE")
    print("="*70)
    print(f"Total experiments: {len(all_results)}/{total_runs}")
    print(f"Results saved to: {output_dir}")
    print("="*70)

if __name__ == "__main__":
    main()
