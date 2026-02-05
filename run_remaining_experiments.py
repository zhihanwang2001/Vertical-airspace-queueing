"""
Run remaining 5 experiments after HCA2C seed45 completes
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "Code"))

from algorithms.baselines.sb3_a2c_baseline import SB3A2CBaseline
from algorithms.baselines.sb3_ppo_baseline import SB3PPOBaseline
from algorithms.hca2c.hca2c_baseline import HCA2CBaseline

CONFIG = {
    "hca2c_timesteps": 500000,
    "baseline_timesteps": 100000,
    "eval_episodes": 30,
}

def run_experiment(algo_name, algo_class, timesteps, seed, load, output_dir):
    """Run single experiment"""
    print(f"\n{'='*70}")
    print(f"Running: {algo_name} | Seed={seed} | Load={load}x | Steps={timesteps}")
    print(f"{'='*70}")
    
    # Create model
    if algo_name == "HCA2C":
        model = algo_class({"seed": seed, "verbose": 1, "load_multiplier": load})
        model.setup_env(load_multiplier=load)
    else:
        model = algo_class({"seed": seed, "verbose": 1})
        model.setup_env(load_multiplier=load)
    
    # Train
    print(f"\n[Training] {algo_name} for {timesteps:,} steps...")
    start_time = time.time()
    model.train(total_timesteps=timesteps)
    train_time = time.time() - start_time
    
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
    
    output_file = output_dir / f"{algo_name}_seed{seed}_load{load}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    model_file = output_dir / f"{algo_name}_seed{seed}_load{load}_model.zip"
    model.save(str(model_file))
    
    print(f"\n[Results] {algo_name}:")
    print(f"  Mean Reward: {results['mean_reward']:.1f} ± {results['std_reward']:.1f}")
    print(f"  Saved to: {output_file}")
    
    return results

def main():
    output_dir = Path("Data/hca2c_final_comparison_local")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    experiments = [
        ("A2C", SB3A2CBaseline, CONFIG["baseline_timesteps"], 45, 7.0),
        ("PPO", SB3PPOBaseline, CONFIG["baseline_timesteps"], 45, 7.0),
        ("HCA2C", HCA2CBaseline, CONFIG["hca2c_timesteps"], 46, 7.0),
        ("A2C", SB3A2CBaseline, CONFIG["baseline_timesteps"], 46, 7.0),
        ("PPO", SB3PPOBaseline, CONFIG["baseline_timesteps"], 46, 7.0),
    ]
    
    print("\n" + "="*70)
    print("Running Remaining 5 Experiments")
    print("="*70)
    
    for i, (algo_name, algo_class, timesteps, seed, load) in enumerate(experiments, 1):
        print(f"\n{'#'*70}")
        print(f"Progress: {i}/5")
        print(f"{'#'*70}")
        
        try:
            run_experiment(algo_name, algo_class, timesteps, seed, load, output_dir)
        except Exception as e:
            print(f"\n[ERROR] Failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()
