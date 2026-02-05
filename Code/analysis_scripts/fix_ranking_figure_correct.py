#!/usr/bin/env python3
"""
Fix Algorithm Performance Ranking Figure (Correct Order)
Fix Algorithm Performance Ranking Figure (Correct Order)
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set Chinese font and style
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def create_ranking_figure():
    """Create correctly sorted ranking figure based on result.md"""
    
    # Based on final ranking data from result.md (sorted from high to low)
    algorithms = [
        'PPO', 'TD7', 'R2D2', 'SAC v2', 'TD3', 'SAC', 'Heuristic', 
        'Rainbow DQN', 'Priority', 'FCFS', 'SJF', 'DDPG', 'A2C', 'IMPALA', 'Random'
    ]
    
    mean_rewards = [
        4419.98, 4392.52, 4289.22, 4282.94, 3972.69, 3659.63, 2860.69,
        2413.46, 2040.04, 2024.75, 2011.16, 1889.25, 1724.72, 1705.13, 294.75
    ]
    
    std_rewards = [
        135.71, 84.60, 82.23, 80.70, 168.56, 1386.03, 87.96,
        166.43, 67.63, 66.64, 66.58, 119.34, 52.68, 25.24, 308.75
    ]
    
    # Color classification: Top tier(>4200) red, Mid tier(2000-4000) orange, Low tier(1000-2000) blue, Bottom green
    colors = []
    for reward in mean_rewards:
        if reward > 4200:
            colors.append('#e74c3c')  # 红色 - 顶级层
        elif reward >= 2000:
            colors.append('#f39c12')  # 橙色 - 中级层  
        elif reward >= 1000:
            colors.append('#3498db')  # 蓝色 - 低级层
        else:
            colors.append('#2ecc71')  # green - 最低层
    
    # Create horizontal bar chart (highest score at top)
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # yy-axis position (highest score at top, so reverse)
    y_pos = np.arange(len(algorithms))[::-1]  # Reverse y-axis position
    
    # Draw bar chart
    bars = ax.barh(y_pos, mean_rewards, xerr=std_rewards, 
                   color=colors, alpha=0.8, capsize=5,
                   error_kw={'elinewidth': 2, 'capthick': 2})
    
    # Set algorithm labels (highest score at top)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(algorithms, fontsize=11)
    
    # Set x-axis
    ax.set_xlabel('Mean Reward', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 5000)
    
    # Add title
    ax.set_title('Algorithm Performance Ranking in MCRPS/D/K Framework', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Add performance tier lines
    ax.axvline(x=4000, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax.axvline(x=2000, color='orange', linestyle='--', alpha=0.7, linewidth=2)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', alpha=0.8, label='Top-Tier (>4000)'),
        Patch(facecolor='#f39c12', alpha=0.8, label='Mid-Tier (2000-4000)'),
        Patch(facecolor='#3498db', alpha=0.8, label='Low-Tier (1000-2000)'),
        Patch(facecolor='#2ecc71', alpha=0.8, label='Bottom-Tier (<1000)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    # Add values on each bar
    for i, (bar, mean, std) in enumerate(zip(bars, mean_rewards, std_rewards)):
        width = bar.get_width()
        ax.text(width + 50, bar.get_y() + bar.get_height()/2, 
                f'{mean:.0f}±{std:.0f}', 
                ha='left', va='center', fontsize=9, fontweight='bold')
    
    # Add ranking indicators (top 3)
    for i in range(3):
        medals = ['🥇', '🥈', '🥉']
        ax.text(100, y_pos[i], medals[i], 
                ha='left', va='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('../../Figures/analysis/figure1_performance_ranking_corrected.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Corrected ranking figure generated: figure1_performance_ranking_corrected.png")
    print("Correct order: PPO (highest) at top, Random (lowest) at bottom")

if __name__ == "__main__":
    create_ranking_figure()