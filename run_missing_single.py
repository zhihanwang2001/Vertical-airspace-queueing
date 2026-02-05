"""
Run a single missing experiment: HCA2C seed45 load7.0
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "Code"))

from algorithms.hca2c.hca2c_baseline import HCA2CBaseline

def main():
    print("\n" + "="*70)
    print("Running: HCA2C | Seed=45 | Load=7.0x | Steps=500,000")
    print("="*70)
    
    # Create output directory
    output_dir = Path("Data/hca2c_final_comparison_local")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model
    print("\n[Setup] Creating HCA2C model...")
    model = HCA2CBaseline({
        "seed": 45,
        "verbose": 1,
        "load_multiplier": 7.0,
    })
    model.setup_env(load_multiplier=7.0)
    print("✓ Model created")
    
    # Train
    print(f"\n[Training] Starting training for 500,000 steps...")
    start_time = time.time()
    model.train(total_timesteps=500000)
    train_time = time.time() - start_time
    print(f"\n✓ Training completed in {train_time:.1f}s ({train_time/60:.1f} min)")
    
    # Evaluate
    print(f"\n[Evaluation] Running 30 episodes...")
    eval_results = model.evaluate(n_episodes=30, verbose=False)
    
    # Save results
    results = {
        "algorithm": "HCA2C",
        "seed": 45,
        "load_multiplier": 7.0,
        "timesteps": 500000,
        "train_time": train_time,
        "mean_reward": eval_results["mean_reward"],
        "std_reward": eval_results["std_reward"],
        "mean_length": eval_results["mean_length"],
        "crash_rate": eval_results.get("crash_rate", 0.0),
        "episode_rewards": eval_results.get("episode_rewards", []),
        "episode_lengths": eval_results.get("episode_lengths", []),
    }
    
    # Save JSON
    output_file = output_dir / "HCA2C_seed45_load7.0.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save model
    model_file = output_dir / "HCA2C_seed45_load7.0_model.zip"
    model.save(str(model_file))
    
    print(f"\n{'='*70}")
    print("[Results] HCA2C seed45 load7.0:")
    print(f"  Mean Reward: {results['mean_reward']:.1f} ± {results['std_reward']:.1f}")
    print(f"  Mean Length: {results['mean_length']:.1f}")
    print(f"  Crash Rate: {results['crash_rate']:.2%}")
    print(f"  Training Time: {train_time/60:.1f} min")
    print(f"  Saved to: {output_file}")
    print("="*70)

if __name__ == "__main__":
    main()
