"""
Optimized Algorithm Configurations
Optimized algorithm configurations for better convergence
"""

# TD3 optimized configuration
TD3_OPTIMIZED_CONFIG = {
    'learning_rate': 1e-4,      # Reduced learning rate
    'buffer_size': 1000000,
    'batch_size': 256,
    'gamma': 0.99,
    'tau': 0.005,
    'policy_noise': 0.1,        # Reduced policy noise
    'noise_clip': 0.3,          # Reduced noise clip
    'policy_freq': 2,
    'expl_noise': 0.05,         # Reduced exploration noise
    'max_action': 1.0,
    'min_action': -1.0
}

# A2C optimized configuration
A2C_OPTIMIZED_CONFIG = {
    'learning_rate': 1e-4,      # Reduced learning rate
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'value_loss_coef': 0.5,
    'entropy_coef': 0.01,
    'max_grad_norm': 0.5,
    'n_steps': 128,             # Reduced rollout length
    'normalize_advantage': True
}

# SAC optimized configuration
SAC_OPTIMIZED_CONFIG = {
    'learning_rate': 1e-4,      # Reduced learning rate
    'buffer_size': 1000000,
    'batch_size': 256,
    'gamma': 0.99,
    'tau': 0.005,
    'alpha': 0.1,               # Reduced entropy coefficient
    'automatic_entropy_tuning': True,
    'learning_starts': 5000,    # Increased learning start steps
    'train_freq': 1,
    'gradient_steps': 1
}

# DQN optimized configuration
DQN_OPTIMIZED_CONFIG = {
    'learning_rate': 5e-5,      # Reduced learning rate
    'buffer_size': 100000,
    'batch_size': 32,
    'gamma': 0.99,
    'epsilon_start': 0.9,       # Reduced initial exploration
    'epsilon_end': 0.02,        # Reduced final exploration
    'epsilon_decay': 0.9995,    # Slower decay
    'target_update_freq': 2000, # Increased update frequency
    'learning_starts': 5000,
    'n_bins_per_dim': 5
}

# Heuristic optimized configuration
HEURISTIC_OPTIMIZED_CONFIG = {
    'load_balance_threshold': 0.75,  # More conservative threshold
    'utilization_target': 0.65,     # More conservative target
    'emergency_threshold': 0.85,    # More conservative emergency threshold
    'adaptive_factor': 0.05         # Slower adaptation
}