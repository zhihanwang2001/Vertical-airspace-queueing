"""
测试SB3模型保存 - 找出pickle问题的根源
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from algorithms.baselines.sb3_a2c_baseline import SB3A2CBaseline
import pickle

print("创建A2C baseline...")
a2c = SB3A2CBaseline()

# 训练几步
print("快速训练100步...")
a2c.train(total_timesteps=100)

# 测试不同的exclude组合
print("\n" + "="*80)
print("测试不同的exclude组合")
print("="*80)

test_cases = [
    ([], "无exclude"),
    (['env'], "exclude=['env']"),
    (['env', 'ep_info_buffer', 'ep_success_buffer'], "exclude环境和buffer"),
    (['env', '_vec_normalize_env'], "exclude环境和normalize"),
    (['env', '_vec_normalize_env', 'ep_info_buffer', 'ep_success_buffer'], "exclude所有常见对象"),
]

for exclude_list, desc in test_cases:
    print(f"\n测试: {desc}")
    print(f"Exclude: {exclude_list}")
    try:
        test_path = f"/tmp/test_a2c_save_{len(exclude_list)}"
        a2c.model.save(test_path, exclude=exclude_list if exclude_list else None)

        # 检查文件大小
        if os.path.exists(test_path + ".zip"):
            size = os.path.getsize(test_path + ".zip")
            print(f"✅ 成功! 文件大小: {size} bytes")

            # 尝试加载验证
            from stable_baselines3 import A2C
            loaded_model = A2C.load(test_path)
            print(f"✅ 成功加载模型")
        else:
            print(f"⚠️  文件未创建")
    except Exception as e:
        print(f"❌ 失败: {type(e).__name__}: {str(e)[:100]}")

print("\n" + "="*80)
print("测试完成")
print("="*80)

# 尝试直接检查哪个属性有问题
print("\n检查model对象的属性...")
for attr_name in dir(a2c.model):
    if attr_name.startswith('_'):
        continue
    try:
        attr_value = getattr(a2c.model, attr_name)
        if callable(attr_value):
            continue
        # 尝试pickle这个属性
        pickle.dumps(attr_value)
    except Exception as e:
        print(f"⚠️  属性 '{attr_name}' 无法pickle: {type(e).__name__}")
