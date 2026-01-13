"""
优化的算法配置
Optimized algorithm configurations for better convergence
"""

# TD3 优化配置
TD3_OPTIMIZED_CONFIG = {
    'learning_rate': 1e-4,      # 降低学习率
    'buffer_size': 1000000,
    'batch_size': 256,
    'gamma': 0.99,
    'tau': 0.005,
    'policy_noise': 0.1,        # 减小策略噪声
    'noise_clip': 0.3,          # 减小噪声裁剪
    'policy_freq': 2,
    'expl_noise': 0.05,         # 减小探索噪声
    'max_action': 1.0,
    'min_action': -1.0
}

# A2C 优化配置
A2C_OPTIMIZED_CONFIG = {
    'learning_rate': 1e-4,      # 降低学习率
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'value_loss_coef': 0.5,
    'entropy_coef': 0.01,
    'max_grad_norm': 0.5,
    'n_steps': 128,             # 减少rollout长度
    'normalize_advantage': True
}

# SAC 优化配置
SAC_OPTIMIZED_CONFIG = {
    'learning_rate': 1e-4,      # 降低学习率
    'buffer_size': 1000000,
    'batch_size': 256,
    'gamma': 0.99,
    'tau': 0.005,
    'alpha': 0.1,               # 降低熵系数
    'automatic_entropy_tuning': True,
    'learning_starts': 5000,    # 增加学习开始步数
    'train_freq': 1,
    'gradient_steps': 1
}

# DQN 优化配置
DQN_OPTIMIZED_CONFIG = {
    'learning_rate': 5e-5,      # 降低学习率
    'buffer_size': 100000,
    'batch_size': 32,
    'gamma': 0.99,
    'epsilon_start': 0.9,       # 降低初始探索
    'epsilon_end': 0.02,        # 降低最终探索
    'epsilon_decay': 0.9995,    # 更慢的衰减
    'target_update_freq': 2000, # 增加更新频率
    'learning_starts': 5000,
    'n_bins_per_dim': 5
}

# Heuristic 优化配置
HEURISTIC_OPTIMIZED_CONFIG = {
    'load_balance_threshold': 0.75,  # 更保守的阈值
    'utilization_target': 0.65,     # 更保守的目标
    'emergency_threshold': 0.85,    # 更保守的紧急阈值
    'adaptive_factor': 0.05         # 更慢的自适应
}