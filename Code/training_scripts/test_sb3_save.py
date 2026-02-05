"""
Test SB3 Model Saving - Find the root cause of pickle issues
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from algorithms.baselines.sb3_a2c_baseline import SB3A2CBaseline
import pickle

print("Creating A2C baseline...")
a2c = SB3A2CBaseline()

# Train for a few steps
print("Quick training for 100 steps...")
a2c.train(total_timesteps=100)

# Test different exclude combinations
print("\n" + "="*80)
print("Testing different exclude combinations")
print("="*80)

test_cases = [
    ([], "No exclude"),
    (['env'], "exclude=['env']"),
    (['env', 'ep_info_buffer', 'ep_success_buffer'], "Exclude environment and buffers"),
    (['env', '_vec_normalize_env'], "Exclude environment and normalize"),
    (['env', '_vec_normalize_env', 'ep_info_buffer', 'ep_success_buffer'], "Exclude all common objects"),
]

for exclude_list, desc in test_cases:
    print(f"\nTest: {desc}")
    print(f"Exclude: {exclude_list}")
    try:
        test_path = f"/tmp/test_a2c_save_{len(exclude_list)}"
        a2c.model.save(test_path, exclude=exclude_list if exclude_list else None)

        # Check file size
        if os.path.exists(test_path + ".zip"):
            size = os.path.getsize(test_path + ".zip")
            print(f"Success! File size: {size} bytes")

            # Try loading to verify
            from stable_baselines3 import A2C
            loaded_model = A2C.load(test_path)
            print(f"Successfully loaded model")
        else:
            print(f"Warning: File not created")
    except Exception as e:
        print(f"Failed: {type(e).__name__}: {str(e)[:100]}")

print("\n" + "="*80)
print("Testing completed")
print("="*80)

# Try to directly check which attribute has issues
print("\nChecking model object attributes...")
for attr_name in dir(a2c.model):
    if attr_name.startswith('_'):
        continue
    try:
        attr_value = getattr(a2c.model, attr_name)
        if callable(attr_value):
            continue
        # Try to pickle this attribute
        pickle.dumps(attr_value)
    except Exception as e:
        print(f"Warning: Attribute '{attr_name}' cannot be pickled: {type(e).__name__}")
