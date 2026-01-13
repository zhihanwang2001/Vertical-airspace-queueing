#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCRPS/D/K框架架构图生成器
Generate Framework Architecture Diagram for MCRPS/D/K
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib import font_manager
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建图形
fig, ax = plt.subplots(figsize=(16, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# 颜色方案
COLOR_QUEUE = '#E8F4F8'      # 浅蓝 - 排队系统
COLOR_DRL = '#FFF4E6'         # 浅橙 - DRL系统
COLOR_PARETO = '#F0E6FF'      # 浅紫 - Pareto优化
COLOR_CONTROL = '#E8F8E8'     # 浅绿 - 控制系统
COLOR_ARROW = '#4A90E2'       # 蓝色箭头
COLOR_HIGHLIGHT = '#FF6B6B'   # 红色高亮

# ============================================================================
# 第1层：垂直分层排队系统 (顶部)
# ============================================================================
y_queue = 8.5
queue_height = 1.2

# 标题
ax.text(5, y_queue + 0.8, 'MCRPS/D/K 垂直分层排队系统',
        ha='center', va='center', fontsize=16, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_QUEUE, edgecolor='black', linewidth=2))

# 5个队列层（倒金字塔）
layer_info = [
    {'name': 'L5 (100m)', 'capacity': 'C=8', 'service': 'μ=1.2', 'width': 3.0, 'y': y_queue},
    {'name': 'L4 (80m)', 'capacity': 'C=6', 'service': 'μ=1.0', 'width': 2.6, 'y': y_queue - 0.3},
    {'name': 'L3 (60m)', 'capacity': 'C=4', 'service': 'μ=0.8', 'width': 2.2, 'y': y_queue - 0.6},
    {'name': 'L2 (40m)', 'capacity': 'C=3', 'service': 'μ=0.6', 'width': 1.8, 'y': y_queue - 0.9},
    {'name': 'L1 (20m)', 'capacity': 'C=2', 'service': 'μ=0.4', 'width': 1.4, 'y': y_queue - 1.2},
]

for i, layer in enumerate(layer_info):
    x_center = 2.5
    # 绘制队列层
    rect = FancyBboxPatch((x_center - layer['width']/2, layer['y'] - 0.12),
                           layer['width'], 0.24,
                           boxstyle="round,pad=0.02",
                           facecolor=COLOR_QUEUE,
                           edgecolor='#2C5F7F', linewidth=2)
    ax.add_patch(rect)

    # 层标签
    ax.text(x_center - layer['width']/2 - 0.3, layer['y'], layer['name'],
            ha='right', va='center', fontsize=10, fontweight='bold')

    # 容量和服务率
    ax.text(x_center, layer['y'] + 0.05, f"{layer['capacity']}, {layer['service']}",
            ha='center', va='center', fontsize=9)

# 到达流（箭头从上方）
arrival_x = 2.5
for i in range(5):
    y_arr = layer_info[0]['y'] + 0.3 + i * 0.05
    ax.annotate('', xy=(arrival_x + i*0.3 - 0.6, layer_info[i]['y'] + 0.13),
                xytext=(arrival_x + i*0.3 - 0.6, y_arr),
                arrowprops=dict(arrowstyle='->', lw=1.5, color=COLOR_ARROW))

# 到达权重标注
ax.text(arrival_x - 0.6, layer_info[0]['y'] + 0.5, 'α=[0.3, 0.25, 0.2, 0.15, 0.1]',
        ha='center', va='bottom', fontsize=9, style='italic', color='#2C5F7F')

# 层间传输箭头（向下）
for i in range(4):
    y_from = layer_info[i]['y'] - 0.13
    y_to = layer_info[i+1]['y'] + 0.13
    x_transfer = 2.5 + layer_info[i]['width']/2 + 0.15

    ax.annotate('', xy=(x_transfer, y_to),
                xytext=(x_transfer, y_from),
                arrowprops=dict(arrowstyle='<->', lw=1.2, color=COLOR_HIGHLIGHT, linestyle='--'))

ax.text(2.5 + 2.0, y_queue - 0.6, '压力触发\n层间传输',
        ha='center', va='center', fontsize=9, color=COLOR_HIGHLIGHT, fontweight='bold')

# ============================================================================
# 第2层：压力度量与控制决策 (中上部)
# ============================================================================
y_pressure = 6.2

# 压力度量模块
pressure_box = FancyBboxPatch((0.3, y_pressure - 0.3), 1.8, 0.6,
                              boxstyle="round,pad=0.1",
                              facecolor=COLOR_CONTROL,
                              edgecolor='#2F5F2F', linewidth=2)
ax.add_patch(pressure_box)

ax.text(1.2, y_pressure + 0.15, '压力度量 P',
        ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(1.2, y_pressure - 0.05, r'$P_\ell = \beta_1 \frac{Q_\ell}{C_\ell} + \beta_2(1-\frac{\mu_\ell}{\mu_{max}}) + \beta_3 G_\ell$',
        ha='center', va='center', fontsize=8)

# 连接到队列系统
ax.annotate('', xy=(1.2, y_pressure + 0.35),
            xytext=(1.2, y_queue - 1.35),
            arrowprops=dict(arrowstyle='<-', lw=2, color=COLOR_ARROW))

# 公平性控制（基尼系数）
gini_box = FancyBboxPatch((2.5, y_pressure - 0.3), 1.8, 0.6,
                          boxstyle="round,pad=0.1",
                          facecolor=COLOR_CONTROL,
                          edgecolor='#2F5F2F', linewidth=2)
ax.add_patch(gini_box)

ax.text(3.4, y_pressure + 0.15, '公平性控制',
        ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(3.4, y_pressure - 0.05, r'Gini: $G_t \leq G_{target} + \epsilon$',
        ha='center', va='center', fontsize=8)

# ============================================================================
# 第3层：深度强化学习模块 (中部)
# ============================================================================
y_drl = 4.0

# DRL主模块
drl_main = FancyBboxPatch((4.8, y_drl - 0.8), 4.5, 1.6,
                          boxstyle="round,pad=0.15",
                          facecolor=COLOR_DRL,
                          edgecolor='#D97706', linewidth=3)
ax.add_patch(drl_main)

ax.text(7.05, y_drl + 0.65, '深度强化学习 (DRL)',
        ha='center', va='center', fontsize=13, fontweight='bold')

# 状态空间
state_box = FancyBboxPatch((5.0, y_drl + 0.1), 1.9, 0.45,
                           boxstyle="round,pad=0.05",
                           facecolor='white',
                           edgecolor='#D97706', linewidth=1.5)
ax.add_patch(state_box)
ax.text(5.95, y_drl + 0.38, '状态空间 (29D)', ha='center', va='center', fontsize=9, fontweight='bold')
ax.text(5.95, y_drl + 0.22, 'Queue, Util, Load,', ha='center', va='center', fontsize=7)
ax.text(5.95, y_drl + 0.12, 'Service, Metrics', ha='center', va='center', fontsize=7)

# 动作空间
action_box = FancyBboxPatch((7.2, y_drl + 0.1), 1.9, 0.45,
                            boxstyle="round,pad=0.05",
                            facecolor='white',
                            edgecolor='#D97706', linewidth=1.5)
ax.add_patch(action_box)
ax.text(8.15, y_drl + 0.38, '动作空间 (11D)', ha='center', va='center', fontsize=9, fontweight='bold')
ax.text(8.15, y_drl + 0.22, 'Service×5, Arrival×1,', ha='center', va='center', fontsize=7)
ax.text(8.15, y_drl + 0.12, 'Transfer×5', ha='center', va='center', fontsize=7)

# 奖励函数
reward_box = FancyBboxPatch((5.0, y_drl - 0.45), 4.1, 0.45,
                            boxstyle="round,pad=0.05",
                            facecolor='white',
                            edgecolor='#D97706', linewidth=1.5)
ax.add_patch(reward_box)
ax.text(7.05, y_drl - 0.12, '奖励函数 R = R_吞吐 + R_公平(基尼) + R_效率 + R_转移 + R_稳定 + P_拥塞 + P_不稳定',
        ha='center', va='center', fontsize=7.5, fontweight='bold')
ax.text(7.05, y_drl - 0.28, '权重: [10.0, 5.0, 3.0, 2.0, 稳定奖励, -20.0, -15.0]',
        ha='center', va='center', fontsize=6.5, style='italic')
ax.text(7.05, y_drl - 0.40, '基于排队论稳定性约束与多目标优化原则设计',
        ha='center', va='center', fontsize=6, color='#D97706', style='italic')

# 算法模块（下方3个box）
algo_y = y_drl - 0.75
algos = [
    {'name': 'A2C/PPO', 'x': 5.2, 'desc': '策略梯度\n并列第一'},
    {'name': 'TD7', 'x': 7.05, 'desc': 'SALE表示\n跳跃学习'},
    {'name': 'R2D2', 'x': 8.9, 'desc': 'LSTM记忆\nTop-4'}
]

for algo in algos:
    algo_box = FancyBboxPatch((algo['x'] - 0.5, algo_y - 0.25), 1.0, 0.5,
                              boxstyle="round,pad=0.05",
                              facecolor='#FFE8CC',
                              edgecolor='#D97706', linewidth=1.5)
    ax.add_patch(algo_box)
    lines = algo['desc'].split('\n')
    ax.text(algo['x'], algo_y + 0.1, algo['name'],
            ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(algo['x'], algo_y - 0.08, lines[0],
            ha='center', va='center', fontsize=7)
    ax.text(algo['x'], algo_y - 0.18, lines[1],
            ha='center', va='center', fontsize=7)

# 连接DRL到队列系统
ax.annotate('', xy=(4.5, y_queue - 0.6),
            xytext=(4.9, y_drl + 0.3),
            arrowprops=dict(arrowstyle='<-', lw=2.5, color=COLOR_ARROW,
                          connectionstyle="arc3,rad=0.3"))

ax.text(4.5, y_queue - 0.25, '状态观测', ha='center', va='center',
        fontsize=9, color=COLOR_ARROW, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=COLOR_ARROW))

ax.annotate('', xy=(4.9, y_drl - 0.2),
            xytext=(4.5, y_queue - 1.0),
            arrowprops=dict(arrowstyle='<-', lw=2.5, color=COLOR_HIGHLIGHT,
                          connectionstyle="arc3,rad=-0.3"))

ax.text(4.5, y_queue - 1.35, '控制动作', ha='center', va='center',
        fontsize=9, color=COLOR_HIGHLIGHT, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=COLOR_HIGHLIGHT))

# ============================================================================
# 第4层：Pareto多目标优化 (底部)
# ============================================================================
y_pareto = 1.5

# Pareto主框
pareto_main = FancyBboxPatch((0.3, y_pareto - 0.5), 4.0, 1.0,
                             boxstyle="round,pad=0.15",
                             facecolor=COLOR_PARETO,
                             edgecolor='#7C3AED', linewidth=3)
ax.add_patch(pareto_main)

ax.text(2.3, y_pareto + 0.35, 'Pareto 多目标优化',
        ha='center', va='center', fontsize=12, fontweight='bold')

# 6个目标
objectives = ['吞吐量↑', '延迟↓', '公平性↑', '稳定性↑', '安全性↑', '效率↑']
obj_x_start = 0.5
for i, obj in enumerate(objectives):
    obj_box = FancyBboxPatch((obj_x_start + i*0.63, y_pareto - 0.15), 0.58, 0.25,
                             boxstyle="round,pad=0.03",
                             facecolor='white',
                             edgecolor='#7C3AED', linewidth=1)
    ax.add_patch(obj_box)
    ax.text(obj_x_start + i*0.63 + 0.29, y_pareto - 0.025, obj,
            ha='center', va='center', fontsize=7)

# Pareto结果
ax.text(2.3, y_pareto - 0.42, '10,000解 → 262帕累托解 → 13膝点',
        ha='center', va='center', fontsize=8, style='italic', color='#7C3AED')

# 连接DRL到Pareto
ax.annotate('', xy=(4.4, y_pareto + 0.1),
            xytext=(5.5, y_drl - 1.1),
            arrowprops=dict(arrowstyle='->', lw=2, color='#7C3AED',
                          connectionstyle="arc3,rad=0.2"))

ax.text(4.8, 2.8, '策略评估', ha='center', va='center',
        fontsize=9, color='#7C3AED', fontweight='bold')

# ============================================================================
# 第5层：部署与应用 (最底部)
# ============================================================================
y_deploy = 0.3

deploy_boxes = [
    {'name': '实时调度', 'x': 1.5},
    {'name': '城市空域', 'x': 3.5},
    {'name': '配送服务', 'x': 5.5},
    {'name': '性能监控', 'x': 7.5}
]

for box in deploy_boxes:
    deploy_rect = FancyBboxPatch((box['x'] - 0.6, y_deploy - 0.15), 1.2, 0.3,
                                 boxstyle="round,pad=0.05",
                                 facecolor='#D1FAE5',
                                 edgecolor='#059669', linewidth=2)
    ax.add_patch(deploy_rect)
    ax.text(box['x'], y_deploy, box['name'],
            ha='center', va='center', fontsize=9, fontweight='bold')

# 从Pareto到部署的箭头
ax.annotate('', xy=(2.3, y_deploy + 0.2),
            xytext=(2.3, y_pareto - 0.55),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='#059669'))

ax.text(2.3, 0.85, '部署配置', ha='center', va='center',
        fontsize=9, color='#059669', fontweight='bold')

# ============================================================================
# 添加图例
# ============================================================================
legend_elements = [
    mpatches.Patch(facecolor=COLOR_QUEUE, edgecolor='#2C5F7F', label='垂直分层排队系统'),
    mpatches.Patch(facecolor=COLOR_DRL, edgecolor='#D97706', label='深度强化学习模块'),
    mpatches.Patch(facecolor=COLOR_PARETO, edgecolor='#7C3AED', label='Pareto多目标优化'),
    mpatches.Patch(facecolor=COLOR_CONTROL, edgecolor='#2F5F2F', label='压力度量与控制'),
]

ax.legend(handles=legend_elements, loc='upper right', fontsize=9,
          frameon=True, fancybox=True, shadow=True)

# 添加标题和说明
fig.suptitle('MCRPS/D/K 框架架构图', fontsize=18, fontweight='bold', y=0.98)
ax.text(5, 0.05, 'Multi-Class Related Poisson arrival, Random batch Service, Poisson splitting, '
                 'state-Dependent control, Dynamic transfer, finite Capacity K',
        ha='center', va='bottom', fontsize=8, style='italic', color='gray')

# 调整布局
plt.tight_layout(rect=[0, 0.02, 1, 0.96])

# 保存图形
output_path = 'figure4_architecture.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ 框架架构图已生成: {output_path}")

# 显示图形
plt.show()
