"""
更新优化算法训练曲线对比图（包含最新A2C-v3数据）
Updated Optimization Training Curves Comparison with Latest A2C-v3
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

# 读取训练数据
a2c_data = pd.read_csv('result_excel/SB3_A2C.csv')
rainbow_data = pd.read_csv('result_excel/Rainbow_DQN.csv')
impala_data = pd.read_csv('result_excel/IMPALA.csv')

# 创建4个子图
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# ====== 子图1: A2C-v3 训练曲线 ======
ax1 = axes[0, 0]
ax1.plot(a2c_data['Step'], a2c_data['Value'], linewidth=1.5, alpha=0.8, color='#FFD700', label='A2C-v3')
ax1.axvline(x=300000, color='red', linestyle='--', linewidth=2, alpha=0.7, label='300k: 余弦退火开始')
ax1.axhline(y=4437.86, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='最终: 4437.86±128.41')
ax1.fill_between([0, 300000], 0, 5000, alpha=0.1, color='blue', label='固定lr=7e-4')
ax1.fill_between([300000, 500000], 0, 5000, alpha=0.1, color='orange', label='余弦退火→1e-5')

ax1.set_xlabel('训练步数 (Training Steps)', fontsize=11, fontweight='bold')
ax1.set_ylabel('奖励 (Reward)', fontsize=11, fontweight='bold')
ax1.set_title('A2C-v3: 延迟余弦退火学习率调度\n🏆 冠军算法 (4437.86±128.41, 6.9分钟)', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9, loc='lower right')
ax1.set_xlim([0, 500000])
ax1.set_ylim([-500, 5000])

# ====== 子图2: Rainbow DQN-v2 训练曲线 ======
ax2 = axes[0, 1]
ax2.plot(rainbow_data['Step'], rainbow_data['Value'], linewidth=1.5, alpha=0.8, color='#ff7f0e', label='Rainbow DQN-v2')
ax2.axhline(y=2360.53, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='最终: 2360.53±45.50')
ax2.set_xlabel('训练步数 (Training Steps)', fontsize=11, fontweight='bold')
ax2.set_ylabel('奖励 (Reward)', fontsize=11, fontweight='bold')
ax2.set_title('Rainbow DQN-v2: 稳定性优化\n标准差降低73% (2360±46, 10.9小时)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=9, loc='lower right')
ax2.set_xlim([0, 500000])
ax2.set_ylim([0, 3000])

# ====== 子图3: IMPALA-v2 训练曲线 ======
ax3 = axes[1, 0]
ax3.plot(impala_data['Step'], impala_data['Value'], linewidth=1.5, alpha=0.8, color='#d62728', label='IMPALA-v2')
ax3.axhline(y=1682.19, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='最终: 1682.19±73.85')
ax3.set_xlabel('训练步数 (Training Steps)', fontsize=11, fontweight='bold')
ax3.set_ylabel('奖励 (Reward)', fontsize=11, fontweight='bold')
ax3.set_title('IMPALA-v2: 保守V-trace策略\n消除崩溃，稳定收敛 (1682±74, 1.0小时)', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=9, loc='lower right')
ax3.set_xlim([0, 500000])
ax3.set_ylim([0, 2500])

# ====== 子图4: 三算法对比 ======
ax4 = axes[1, 1]

# 对数据进行平滑处理（移动平均）以便更好对比
def smooth(data, window=50):
    return pd.Series(data).rolling(window=window, min_periods=1).mean()

# 绘制平滑曲线
ax4.plot(a2c_data['Step'], smooth(a2c_data['Value'], 30), linewidth=2.5, alpha=0.9,
         color='#FFD700', label='A2C-v3 (4437±128)', linestyle='-')
ax4.plot(rainbow_data['Step'], smooth(rainbow_data['Value'], 30), linewidth=2.5, alpha=0.9,
         color='#ff7f0e', label='Rainbow DQN-v2 (2361±46)', linestyle='-')
ax4.plot(impala_data['Step'], smooth(impala_data['Value'], 30), linewidth=2.5, alpha=0.9,
         color='#d62728', label='IMPALA-v2 (1682±74)', linestyle='-')

# 标注300k转折点
ax4.axvline(x=300000, color='red', linestyle='--', linewidth=2, alpha=0.5, label='300k: A2C退火开始')

ax4.set_xlabel('训练步数 (Training Steps)', fontsize=11, fontweight='bold')
ax4.set_ylabel('奖励 (Reward)', fontsize=11, fontweight='bold')
ax4.set_title('优化算法性能对比\n超参数优化的决定性作用', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=10, loc='lower right')
ax4.set_xlim([0, 500000])
ax4.set_ylim([0, 5000])

# 添加性能层级标注
ax4.axhspan(4200, 5000, alpha=0.1, color='gold', label='顶级层')
ax4.axhspan(2000, 4200, alpha=0.1, color='silver')
ax4.axhspan(0, 2000, alpha=0.1, color='#CD7F32')
ax4.text(450000, 4600, '顶级层\nA2C-v3', fontsize=9, ha='center',
         bbox=dict(boxstyle='round', facecolor='gold', alpha=0.5))
ax4.text(450000, 2800, '中级层\nRainbow', fontsize=9, ha='center',
         bbox=dict(boxstyle='round', facecolor='silver', alpha=0.5))
ax4.text(450000, 1200, '基础层\nIMPALA', fontsize=9, ha='center',
         bbox=dict(boxstyle='round', facecolor='#CD7F32', alpha=0.5))

# 总标题
fig.suptitle('优化算法训练曲线对比 (Optimization Training Curves Comparison)\n'
             'A2C-v3延迟余弦退火 | Rainbow DQN-v2稳定性优化 | IMPALA-v2保守V-trace',
             fontsize=15, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('../../Figures/analysis/optimization_training_curves.png', dpi=300, bbox_inches='tight')
print("✅ 优化算法训练曲线对比图已更新: optimization_training_curves.png")

plt.show()
