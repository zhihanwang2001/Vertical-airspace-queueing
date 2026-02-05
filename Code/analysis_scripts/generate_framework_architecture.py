#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCRPS/D/K Framework Architecture Diagram Generator
Generate Framework Architecture Diagram for MCRPS/D/K
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib import font_manager
import numpy as np

# Set font for international characters
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Create figure
fig, ax = plt.subplots(figsize=(16, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Color scheme
COLOR_QUEUE = '#E8F4F8'      # Light blue - Queue system
COLOR_DRL = '#FFF4E6'         # Light orange - DRL system
COLOR_PARETO = '#F0E6FF'      # Light purple - Pareto optimization
COLOR_CONTROL = '#E8F8E8'     # Light green - Control system
COLOR_ARROW = '#4A90E2'       # Blue arrow
COLOR_HIGHLIGHT = '#FF6B6B'   # Red highlight

# ============================================================================
# Layer 1: Vertical Stratified Queue System (Top)
# ============================================================================
y_queue = 8.5
queue_height = 1.2

# Title
ax.text(5, y_queue + 0.8, 'MCRPS/D/K Vertical Stratified Queue System',
        ha='center', va='center', fontsize=16, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_QUEUE, edgecolor='black', linewidth=2))

# 5 queue layers (inverted pyramid)
layer_info = [
    {'name': 'L5 (100m)', 'capacity': 'C=8', 'service': 'μ=1.2', 'width': 3.0, 'y': y_queue},
    {'name': 'L4 (80m)', 'capacity': 'C=6', 'service': 'μ=1.0', 'width': 2.6, 'y': y_queue - 0.3},
    {'name': 'L3 (60m)', 'capacity': 'C=4', 'service': 'μ=0.8', 'width': 2.2, 'y': y_queue - 0.6},
    {'name': 'L2 (40m)', 'capacity': 'C=3', 'service': 'μ=0.6', 'width': 1.8, 'y': y_queue - 0.9},
    {'name': 'L1 (20m)', 'capacity': 'C=2', 'service': 'μ=0.4', 'width': 1.4, 'y': y_queue - 1.2},
]

for i, layer in enumerate(layer_info):
    x_center = 2.5
    # Draw queue layer
    rect = FancyBboxPatch((x_center - layer['width']/2, layer['y'] - 0.12),
                           layer['width'], 0.24,
                           boxstyle="round,pad=0.02",
                           facecolor=COLOR_QUEUE,
                           edgecolor='#2C5F7F', linewidth=2)
    ax.add_patch(rect)

    # Layer label
    ax.text(x_center - layer['width']/2 - 0.3, layer['y'], layer['name'],
            ha='right', va='center', fontsize=10, fontweight='bold')

    # Capacity and service rate
    ax.text(x_center, layer['y'] + 0.05, f"{layer['capacity']}, {layer['service']}",
            ha='center', va='center', fontsize=9)

# Arrival stream (arrows from above)
arrival_x = 2.5
for i in range(5):
    y_arr = layer_info[0]['y'] + 0.3 + i * 0.05
    ax.annotate('', xy=(arrival_x + i*0.3 - 0.6, layer_info[i]['y'] + 0.13),
                xytext=(arrival_x + i*0.3 - 0.6, y_arr),
                arrowprops=dict(arrowstyle='->', lw=1.5, color=COLOR_ARROW))

# Arrival weight annotation
ax.text(arrival_x - 0.6, layer_info[0]['y'] + 0.5, 'α=[0.3, 0.25, 0.2, 0.15, 0.1]',
        ha='center', va='bottom', fontsize=9, style='italic', color='#2C5F7F')

# Inter-layer transfer arrows (downward)
for i in range(4):
    y_from = layer_info[i]['y'] - 0.13
    y_to = layer_info[i+1]['y'] + 0.13
    x_transfer = 2.5 + layer_info[i]['width']/2 + 0.15

    ax.annotate('', xy=(x_transfer, y_to),
                xytext=(x_transfer, y_from),
                arrowprops=dict(arrowstyle='<->', lw=1.2, color=COLOR_HIGHLIGHT, linestyle='--'))

ax.text(2.5 + 2.0, y_queue - 0.6, 'Pressure-triggered\nInter-layer Transfer',
        ha='center', va='center', fontsize=9, color=COLOR_HIGHLIGHT, fontweight='bold')

# ============================================================================
# Layer 2: Pressure Measurement and Control Decision (Upper Middle)
# ============================================================================
y_pressure = 6.2

# Pressure measurement module
pressure_box = FancyBboxPatch((0.3, y_pressure - 0.3), 1.8, 0.6,
                              boxstyle="round,pad=0.1",
                              facecolor=COLOR_CONTROL,
                              edgecolor='#2F5F2F', linewidth=2)
ax.add_patch(pressure_box)

ax.text(1.2, y_pressure + 0.15, 'Pressure Metric P',
        ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(1.2, y_pressure - 0.05, r'$P_\ell = \beta_1 \frac{Q_\ell}{C_\ell} + \beta_2(1-\frac{\mu_\ell}{\mu_{max}}) + \beta_3 G_\ell$',
        ha='center', va='center', fontsize=8)

# Connect to queue system
ax.annotate('', xy=(1.2, y_pressure + 0.35),
            xytext=(1.2, y_queue - 1.35),
            arrowprops=dict(arrowstyle='<-', lw=2, color=COLOR_ARROW))

# Fairness control (Gini coefficient)
gini_box = FancyBboxPatch((2.5, y_pressure - 0.3), 1.8, 0.6,
                          boxstyle="round,pad=0.1",
                          facecolor=COLOR_CONTROL,
                          edgecolor='#2F5F2F', linewidth=2)
ax.add_patch(gini_box)

ax.text(3.4, y_pressure + 0.15, 'Fairness Control',
        ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(3.4, y_pressure - 0.05, r'Gini: $G_t \leq G_{target} + \epsilon$',
        ha='center', va='center', fontsize=8)

# ============================================================================
# Layer 3: Deep Reinforcement Learning Module (Middle)
# ============================================================================
y_drl = 4.0

# DRL main module
drl_main = FancyBboxPatch((4.8, y_drl - 0.8), 4.5, 1.6,
                          boxstyle="round,pad=0.15",
                          facecolor=COLOR_DRL,
                          edgecolor='#D97706', linewidth=3)
ax.add_patch(drl_main)

ax.text(7.05, y_drl + 0.65, 'Deep Reinforcement Learning (DRL)',
        ha='center', va='center', fontsize=13, fontweight='bold')

# State space
state_box = FancyBboxPatch((5.0, y_drl + 0.1), 1.9, 0.45,
                           boxstyle="round,pad=0.05",
                           facecolor='white',
                           edgecolor='#D97706', linewidth=1.5)
ax.add_patch(state_box)
ax.text(5.95, y_drl + 0.38, 'State Space (29D)', ha='center', va='center', fontsize=9, fontweight='bold')
ax.text(5.95, y_drl + 0.22, 'Queue, Util, Load,', ha='center', va='center', fontsize=7)
ax.text(5.95, y_drl + 0.12, 'Service, Metrics', ha='center', va='center', fontsize=7)

# Action space
action_box = FancyBboxPatch((7.2, y_drl + 0.1), 1.9, 0.45,
                            boxstyle="round,pad=0.05",
                            facecolor='white',
                            edgecolor='#D97706', linewidth=1.5)
ax.add_patch(action_box)
ax.text(8.15, y_drl + 0.38, 'Action Space (11D)', ha='center', va='center', fontsize=9, fontweight='bold')
ax.text(8.15, y_drl + 0.22, 'Service×5, Arrival×1,', ha='center', va='center', fontsize=7)
ax.text(8.15, y_drl + 0.12, 'Transfer×5', ha='center', va='center', fontsize=7)

# Reward function
reward_box = FancyBboxPatch((5.0, y_drl - 0.45), 4.1, 0.45,
                            boxstyle="round,pad=0.05",
                            facecolor='white',
                            edgecolor='#D97706', linewidth=1.5)
ax.add_patch(reward_box)
ax.text(7.05, y_drl - 0.12, 'Reward Function R = R_throughput + R_fairness(Gini) + R_efficiency + R_transfer + R_stability + P_congestion + P_instability',
        ha='center', va='center', fontsize=7.5, fontweight='bold')
ax.text(7.05, y_drl - 0.28, 'Weights: [10.0, 5.0, 3.0, 2.0, stability reward, -20.0, -15.0]',
        ha='center', va='center', fontsize=6.5, style='italic')
ax.text(7.05, y_drl - 0.40, 'Designed based on queueing theory stability constraints and multi-objective optimization principles',
        ha='center', va='center', fontsize=6, color='#D97706', style='italic')

# Algorithm modules (3 boxes below)
algo_y = y_drl - 0.75
algos = [
    {'name': 'A2C/PPO', 'x': 5.2, 'desc': 'Policy Gradient\nJoint First'},
    {'name': 'TD7', 'x': 7.05, 'desc': 'SALE Representation\nJump Learning'},
    {'name': 'R2D2', 'x': 8.9, 'desc': 'LSTM Memory\nTop-4'}
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

# Connect DRL to queue system
ax.annotate('', xy=(4.5, y_queue - 0.6),
            xytext=(4.9, y_drl + 0.3),
            arrowprops=dict(arrowstyle='<-', lw=2.5, color=COLOR_ARROW,
                          connectionstyle="arc3,rad=0.3"))

ax.text(4.5, y_queue - 0.25, 'State Observation', ha='center', va='center',
        fontsize=9, color=COLOR_ARROW, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=COLOR_ARROW))

ax.annotate('', xy=(4.9, y_drl - 0.2),
            xytext=(4.5, y_queue - 1.0),
            arrowprops=dict(arrowstyle='<-', lw=2.5, color=COLOR_HIGHLIGHT,
                          connectionstyle="arc3,rad=-0.3"))

ax.text(4.5, y_queue - 1.35, 'Control Action', ha='center', va='center',
        fontsize=9, color=COLOR_HIGHLIGHT, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=COLOR_HIGHLIGHT))

# ============================================================================
# Layer 4: Pareto Multi-objective Optimization (Bottom)
# ============================================================================
y_pareto = 1.5

# Pareto main frame
pareto_main = FancyBboxPatch((0.3, y_pareto - 0.5), 4.0, 1.0,
                             boxstyle="round,pad=0.15",
                             facecolor=COLOR_PARETO,
                             edgecolor='#7C3AED', linewidth=3)
ax.add_patch(pareto_main)

ax.text(2.3, y_pareto + 0.35, 'Pareto Multi-objective Optimization',
        ha='center', va='center', fontsize=12, fontweight='bold')

# 6 objectives
objectives = ['Throughput↑', 'Delay↓', 'Fairness↑', 'Stability↑', 'Safety↑', 'Efficiency↑']
obj_x_start = 0.5
for i, obj in enumerate(objectives):
    obj_box = FancyBboxPatch((obj_x_start + i*0.63, y_pareto - 0.15), 0.58, 0.25,
                             boxstyle="round,pad=0.03",
                             facecolor='white',
                             edgecolor='#7C3AED', linewidth=1)
    ax.add_patch(obj_box)
    ax.text(obj_x_start + i*0.63 + 0.29, y_pareto - 0.025, obj,
            ha='center', va='center', fontsize=7)

# Pareto results
ax.text(2.3, y_pareto - 0.42, '10,000 solutions → 262 Pareto solutions → 13 knee points',
        ha='center', va='center', fontsize=8, style='italic', color='#7C3AED')

# Connect DRL to Pareto
ax.annotate('', xy=(4.4, y_pareto + 0.1),
            xytext=(5.5, y_drl - 1.1),
            arrowprops=dict(arrowstyle='->', lw=2, color='#7C3AED',
                          connectionstyle="arc3,rad=0.2"))

ax.text(4.8, 2.8, 'Policy Evaluation', ha='center', va='center',
        fontsize=9, color='#7C3AED', fontweight='bold')

# ============================================================================
# Layer 5: Deployment and Application (Bottom)
# ============================================================================
y_deploy = 0.3

deploy_boxes = [
    {'name': 'Real-time Scheduling', 'x': 1.5},
    {'name': 'Urban Airspace', 'x': 3.5},
    {'name': 'Delivery Service', 'x': 5.5},
    {'name': 'Performance Monitoring', 'x': 7.5}
]

for box in deploy_boxes:
    deploy_rect = FancyBboxPatch((box['x'] - 0.6, y_deploy - 0.15), 1.2, 0.3,
                                 boxstyle="round,pad=0.05",
                                 facecolor='#D1FAE5',
                                 edgecolor='#059669', linewidth=2)
    ax.add_patch(deploy_rect)
    ax.text(box['x'], y_deploy, box['name'],
            ha='center', va='center', fontsize=9, fontweight='bold')

# Arrow from Pareto to deployment
ax.annotate('', xy=(2.3, y_deploy + 0.2),
            xytext=(2.3, y_pareto - 0.55),
            arrowprops=dict(arrowstyle='->', lw=2.5, color='#059669'))

ax.text(2.3, 0.85, 'Deployment Configuration', ha='center', va='center',
        fontsize=9, color='#059669', fontweight='bold')

# ============================================================================
# Add legend
# ============================================================================
legend_elements = [
    mpatches.Patch(facecolor=COLOR_QUEUE, edgecolor='#2C5F7F', label='Vertical Stratified Queue System'),
    mpatches.Patch(facecolor=COLOR_DRL, edgecolor='#D97706', label='Deep Reinforcement Learning Module'),
    mpatches.Patch(facecolor=COLOR_PARETO, edgecolor='#7C3AED', label='Pareto Multi-objective Optimization'),
    mpatches.Patch(facecolor=COLOR_CONTROL, edgecolor='#2F5F2F', label='Pressure Measurement and Control'),
]

ax.legend(handles=legend_elements, loc='upper right', fontsize=9,
          frameon=True, fancybox=True, shadow=True)

# Add title and description
fig.suptitle('MCRPS/D/K Framework Architecture Diagram', fontsize=18, fontweight='bold', y=0.98)
ax.text(5, 0.05, 'Multi-Class Related Poisson arrival, Random batch Service, Poisson splitting, '
                 'state-Dependent control, Dynamic transfer, finite Capacity K',
        ha='center', va='bottom', fontsize=8, style='italic', color='gray')

# Adjust layout
plt.tight_layout(rect=[0, 0.02, 1, 0.96])

# Save figure
output_path = 'figure4_architecture.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Framework architecture diagram generated: {output_path}")

# Display figure
plt.show()
