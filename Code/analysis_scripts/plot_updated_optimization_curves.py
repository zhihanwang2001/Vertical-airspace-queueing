"""
Updated Optimization Training Curves Comparison with Latest A2C-v3
Updated Optimization Training Curves Comparison with Latest A2C-v3
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Set font for international characters
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

# Read training data
a2c_data = pd.read_csv('result_excel/SB3_A2C.csv')
rainbow_data = pd.read_csv('result_excel/Rainbow_DQN.csv')
impala_data = pd.read_csv('result_excel/IMPALA.csv')

# Create 4 subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# ====== Subplot 1: A2C-v3 Training Curve ======
ax1 = axes[0, 0]
ax1.plot(a2c_data['Step'], a2c_data['Value'], linewidth=1.5, alpha=0.8, color='#FFD700', label='A2C-v3')
ax1.axvline(x=300000, color='red', linestyle='--', linewidth=2, alpha=0.7, label='300k: Cosine annealing starts')
ax1.axhline(y=4437.86, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Final: 4437.86±128.41')
ax1.fill_between([0, 300000], 0, 5000, alpha=0.1, color='blue', label='Fixed lr=7e-4')
ax1.fill_between([300000, 500000], 0, 5000, alpha=0.1, color='orange', label='Cosine annealing→1e-5')

ax1.set_xlabel('Training Steps', fontsize=11, fontweight='bold')
ax1.set_ylabel('Reward', fontsize=11, fontweight='bold')
ax1.set_title('A2C-v3: Delayed Cosine Annealing Learning Rate Schedule\n🏆 Champion Algorithm (4437.86±128.41, 6.9 minutes)', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9, loc='lower right')
ax1.set_xlim([0, 500000])
ax1.set_ylim([-500, 5000])

# ====== Subplot 2: Rainbow DQN-v2 Training Curve ======
ax2 = axes[0, 1]
ax2.plot(rainbow_data['Step'], rainbow_data['Value'], linewidth=1.5, alpha=0.8, color='#ff7f0e', label='Rainbow DQN-v2')
ax2.axhline(y=2360.53, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Final: 2360.53±45.50')
ax2.set_xlabel('Training Steps', fontsize=11, fontweight='bold')
ax2.set_ylabel('Reward', fontsize=11, fontweight='bold')
ax2.set_title('Rainbow DQN-v2: Stability Optimization\nStd Dev Reduced 73% (2360±46, 10.9 hours)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=9, loc='lower right')
ax2.set_xlim([0, 500000])
ax2.set_ylim([0, 3000])

# ====== Subplot 3: IMPALA-v2 Training Curve ======
ax3 = axes[1, 0]
ax3.plot(impala_data['Step'], impala_data['Value'], linewidth=1.5, alpha=0.8, color='#d62728', label='IMPALA-v2')
ax3.axhline(y=1682.19, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Final: 1682.19±73.85')
ax3.set_xlabel('Training Steps', fontsize=11, fontweight='bold')
ax3.set_ylabel('Reward', fontsize=11, fontweight='bold')
ax3.set_title('IMPALA-v2: Conservative V-trace Policy\nEliminated Crashes, Stable Convergence (1682±74, 1.0 hour)', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=9, loc='lower right')
ax3.set_xlim([0, 500000])
ax3.set_ylim([0, 2500])

# ====== Subplot 4: Three Algorithm Comparison ======
ax4 = axes[1, 1]

# Smooth data (moving average) for better comparison
def smooth(data, window=50):
    return pd.Series(data).rolling(window=window, min_periods=1).mean()

# Draw smoothed curves
ax4.plot(a2c_data['Step'], smooth(a2c_data['Value'], 30), linewidth=2.5, alpha=0.9,
         color='#FFD700', label='A2C-v3 (4437±128)', linestyle='-')
ax4.plot(rainbow_data['Step'], smooth(rainbow_data['Value'], 30), linewidth=2.5, alpha=0.9,
         color='#ff7f0e', label='Rainbow DQN-v2 (2361±46)', linestyle='-')
ax4.plot(impala_data['Step'], smooth(impala_data['Value'], 30), linewidth=2.5, alpha=0.9,
         color='#d62728', label='IMPALA-v2 (1682±74)', linestyle='-')

# Mark 300k turning point
ax4.axvline(x=300000, color='red', linestyle='--', linewidth=2, alpha=0.5, label='300k: A2C annealing starts')

ax4.set_xlabel('Training Steps', fontsize=11, fontweight='bold')
ax4.set_ylabel('Reward', fontsize=11, fontweight='bold')
ax4.set_title('Optimization Algorithm Performance Comparison\nDecisive Role of Hyperparameter Optimization', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=10, loc='lower right')
ax4.set_xlim([0, 500000])
ax4.set_ylim([0, 5000])

# Add performance tier annotations
ax4.axhspan(4200, 5000, alpha=0.1, color='gold', label='Top Tier')
ax4.axhspan(2000, 4200, alpha=0.1, color='silver')
ax4.axhspan(0, 2000, alpha=0.1, color='#CD7F32')
ax4.text(450000, 4600, 'Top Tier\nA2C-v3', fontsize=9, ha='center',
         bbox=dict(boxstyle='round', facecolor='gold', alpha=0.5))
ax4.text(450000, 2800, 'Mid Tier\nRainbow', fontsize=9, ha='center',
         bbox=dict(boxstyle='round', facecolor='silver', alpha=0.5))
ax4.text(450000, 1200, 'Base Tier\nIMPALA', fontsize=9, ha='center',
         bbox=dict(boxstyle='round', facecolor='#CD7F32', alpha=0.5))

# Overall title
fig.suptitle('Optimization Training Curves Comparison\n'
             'A2C-v3 Delayed Cosine Annealing | Rainbow DQN-v2 Stability Optimization | IMPALA-v2 Conservative V-trace',
             fontsize=15, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('../../Figures/analysis/optimization_training_curves.png', dpi=300, bbox_inches='tight')
print("✅ Optimization training curves comparison figure updated: optimization_training_curves.png")

plt.show()
