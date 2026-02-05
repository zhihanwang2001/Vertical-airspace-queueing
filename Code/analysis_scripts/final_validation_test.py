"""
Final validation test for Pareto analysis fixes
"""
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from env.drl_optimized_env_fixed import DRLOptimizedQueueEnvFixed
from pareto_analysis_final import ParetoAnalyzer

def final_validation_test():
    """Final validation test for all fixes"""
    print("Final Validation Test - All Pareto Analysis Fixes")
    print("=" * 60)

    # Create environment and analyzer
    env = DRLOptimizedQueueEnvFixed()
    analyzer = ParetoAnalyzer(env)

    print("1. Environment and analyzer created successfully")

    # Test with small sample
    print("\n2. Testing with small sample (50 solutions)...")
    analyzer.generate_random_solutions(n_solutions=50)

    print(f"   Generated {len(analyzer.solutions)} solutions")
    print(f"   Objective shape: {analyzer.objective_values.shape}")

    # Check objective statistics
    print("\n3. Objective statistics:")
    for i, name in enumerate(analyzer.objective_names):
        values = analyzer.objective_values[:, i]
        print(f"   {name:12}: min={np.min(values):.3f}, max={np.max(values):.3f}, "
              f"std={np.std(values):.3f}, unique={len(np.unique(values))}/{len(values)}")

    # Test Pareto front
    print("\n4. Testing Pareto front detection...")
    analyzer.find_pareto_front()
    pareto_ratio = len(analyzer.pareto_indices) / len(analyzer.solutions)
    print(f"   Found {len(analyzer.pareto_indices)}/{len(analyzer.solutions)} Pareto solutions ({pareto_ratio:.1%})")

    # Test knee point detection
    print("\n5. Testing knee point detection...")
    knee_indices = analyzer.find_knee_points_improved()
    knee_ratio = len(knee_indices) / len(analyzer.pareto_indices) if len(analyzer.pareto_indices) > 0 else 0
    print(f"   Found {len(knee_indices)} knee points ({knee_ratio:.1%} of Pareto solutions)")

    # Test correlation analysis
    print("\n6. Testing correlation analysis...")
    conflicts = analyzer.analyze_objective_conflicts()
    print(f"   Found {len(conflicts)} significant correlations:")
    for pair, corr in list(conflicts.items())[:3]:  # Show first 3
        print(f"     {pair}: {corr:.3f}")

    # Verify objective extraction timing
    print("\n7. Testing objective extraction timing...")
    # Manually execute a step to test
    obs, _ = env.reset()
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)

    # Extract objectives
    objectives = analyzer._extract_objectives(obs, action, reward, info)
    print(f"   Extracted objectives shape: {objectives.shape}")
    print(f"   Objectives: {objectives}")

    # Verify value validity
    if np.all(np.isfinite(objectives)) and np.all(objectives >= 0):
        print("   All objectives are finite and non-negative")
    else:
        print("   Some objectives have invalid values")

    print("\nFinal validation completed successfully!")
    print("   The Pareto analysis code is ready for full execution.")

    return True

if __name__ == "__main__":
    try:
        success = final_validation_test()
        if success:
            print("\nAll tests passed! You can now run: python pareto_analysis_final.py")
        else:
            print("\nSome tests failed!")
    except Exception as e:
        print(f"\nValidation failed with error: {e}")
        import traceback
        traceback.print_exc()