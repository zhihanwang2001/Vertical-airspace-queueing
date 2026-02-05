"""
验证所有45个实验是否完整
"""
import json
from pathlib import Path

def verify_experiments():
    """验证实验完整性"""
    
    data_dir = Path("Data/hca2c_final_comparison")
    
    print("="*70)
    print("实验完整性验证")
    print("="*70)
    print()
    
    # 预期的实验配置
    algorithms = ["HCA2C", "A2C", "PPO"]
    seeds = [42, 43, 44, 45, 46]
    loads = [3.0, 5.0, 7.0]
    
    total_expected = len(algorithms) * len(seeds) * len(loads)
    print(f"预期实验总数: {total_expected}")
    print(f"  算法: {algorithms}")
    print(f"  Seeds: {seeds}")
    print(f"  Loads: {loads}")
    print()
    
    # 检查每个实验
    missing = []
    found = []
    
    for algo in algorithms:
        for seed in seeds:
            for load in loads:
                filename = f"{algo}_seed{seed}_load{load}.json"
                filepath = data_dir / filename
                
                if filepath.exists():
                    found.append(filename)
                else:
                    missing.append(filename)
    
    print("="*70)
    print("检查结果")
    print("="*70)
    print()
    
    print(f"✅ 已完成: {len(found)}/{total_expected} 个实验")
    print(f"❌ 缺失: {len(missing)}/{total_expected} 个实验")
    print()
    
    if missing:
        print("缺失的实验:")
        for fname in sorted(missing):
            print(f"  - {fname}")
        print()
    
    # 按算法统计
    print("="*70)
    print("按算法统计")
    print("="*70)
    print()
    
    for algo in algorithms:
        algo_files = [f for f in found if f.startswith(algo)]
        expected_per_algo = len(seeds) * len(loads)
        print(f"{algo}: {len(algo_files)}/{expected_per_algo}")
        
        # 检查缺失的配置
        algo_missing = [f for f in missing if f.startswith(algo)]
        if algo_missing:
            print(f"  缺失:")
            for fname in algo_missing:
                print(f"    - {fname}")
    
    print()
    
    # 验证JSON文件内容
    print("="*70)
    print("验证JSON文件内容")
    print("="*70)
    print()
    
    valid_count = 0
    invalid_count = 0
    
    for fname in found[:5]:  # 只检查前5个作为示例
        filepath = data_dir / fname
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                # 检查必要字段
                required_fields = ["algorithm", "seed", "load_multiplier", "mean_reward"]
                if all(field in data for field in required_fields):
                    valid_count += 1
                    print(f"✓ {fname}: mean_reward={data['mean_reward']:.1f}")
                else:
                    invalid_count += 1
                    print(f"✗ {fname}: 缺少必要字段")
        except Exception as e:
            invalid_count += 1
            print(f"✗ {fname}: 读取错误 - {e}")
    
    print(f"\n(检查了前5个文件)")
    print()
    
    return {
        "total_expected": total_expected,
        "found": len(found),
        "missing": len(missing),
        "missing_files": missing,
        "complete": len(missing) == 0
    }

if __name__ == "__main__":
    result = verify_experiments()
    
    print("="*70)
    print("总结")
    print("="*70)
    
    if result["complete"]:
        print("🎉 所有45个实验已完成!")
    else:
        print(f"⚠️  还有 {result['missing']} 个实验未完成")
        print("需要运行本地实验来补充缺失的数据")
