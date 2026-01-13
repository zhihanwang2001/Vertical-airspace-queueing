#!/usr/bin/env python3
"""
验证IMPALA是否已成功回滚到原始配置
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from algorithms.advanced.impala.impala_baseline import IMPALABaseline

def verify_impala_rollback():
    """验证IMPALA回滚配置"""
    print("🔄 验证IMPALA是否已回滚到原始配置...")
    
    # 创建IMPALA实例
    baseline = IMPALABaseline()
    config = baseline.config
    
    print("\n📋 当前配置:")
    
    # 检查原始配置参数
    original_config = {
        "学习率": (config['learning_rate'], 1e-4),
        "熵系数": (config['entropy_coeff'], 0.01),
        "价值损失权重": (config['value_loss_coeff'], 0.5),
        "梯度裁剪": (config['gradient_clip'], 40.0),
        "V-trace ρ_bar": (config['rho_bar'], 1.0),
        "V-trace c_bar": (config['c_bar'], 1.0),
        "缓冲区大小": (config['buffer_size'], 10000),
        "序列长度": (config['sequence_length'], 20),
        "批次大小": (config['batch_size'], 16),
        "学习启动": (config['learning_starts'], 1000),
        "训练频率": (config['train_freq'], 4),
        "更新频率": (config['update_freq'], 100)
    }
    
    all_correct = True
    for param_name, (actual, expected) in original_config.items():
        status = "✅" if actual == expected else "❌"
        if actual != expected:
            all_correct = False
        print(f"   {status} {param_name}: {actual} (期望: {expected})")
    
    print(f"\n🔄 回滚结果:")
    if all_correct:
        print("✅ IMPALA已成功回滚到原始配置！")
        print("✅ 所有参数都恢复到初始值")
        print("✅ 可以重新开始优化策略")
    else:
        print("❌ 回滚不完整，需要检查")
        return False
    
    return True

if __name__ == "__main__":
    print("🚀 开始验证IMPALA回滚...")
    
    success = verify_impala_rollback()
    
    if success:
        print("\n🎯 回滚完成！IMPALA已恢复到原始配置。")
        print("现在可以重新设计优化策略。")
    else:
        print("\n❌ 回滚失败，请检查配置。")