#!/usr/bin/env python3
"""
Create System Architecture Diagram for MCRPS/D/K Environment
Shows the interaction between DRL agent and queueing system
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

# Configuration
OUTPUT_FILE = 'figures/system_architecture.pdf'
DPI = 300

def create_architecture_diagram():
    """Create system architecture diagram"""

    fig, ax = plt.subplots(figsize=(12, 10), dpi=DPI)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(6, 9.5, 'DRL System Architecture for MCRPS/D/K',
            ha='center', va='center', fontsize=16, fontweight='bold')

    # ===== Left Side: Environment Components =====

    # Environment box (large container)
    env_box = FancyBboxPatch((0.5, 1.5), 4.5, 6.5,
                             boxstyle="round,pad=0.1",
                             edgecolor='darkblue', facecolor='lightblue',
                             linewidth=2, alpha=0.3)
    ax.add_patch(env_box)
    ax.text(2.75, 7.7, 'MCRPS/D/K Environment',
            ha='center', fontweight='bold', fontsize=12)

    # Arrival Process
    arrival_box = FancyBboxPatch((0.8, 6.5), 1.8, 0.8,
                                 boxstyle="round,pad=0.05",
                                 edgecolor='green', facecolor='lightgreen',
                                 linewidth=1.5)
    ax.add_patch(arrival_box)
    ax.text(1.7, 6.9, 'Arrival\nProcess', ha='center', va='center', fontsize=9)

    # Queue Layers (5 layers)
    layer_colors = ['#FFE5B4', '#FFD700', '#FFA500', '#FF8C00', '#FF6347']
    layer_names = ['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'Layer 5']

    for i, (color, name) in enumerate(zip(layer_colors, layer_names)):
        y_pos = 5.3 - i * 0.7
        layer_box = FancyBboxPatch((0.8, y_pos), 1.8, 0.5,
                                   boxstyle="round,pad=0.03",
                                   edgecolor='brown', facecolor=color,
                                   linewidth=1.2)
        ax.add_patch(layer_box)
        ax.text(1.7, y_pos + 0.25, name, ha='center', va='center', fontsize=8)

    # Servers
    server_box = FancyBboxPatch((3.2, 2.0), 1.5, 4.8,
                                boxstyle="round,pad=0.05",
                                edgecolor='purple', facecolor='lavender',
                                linewidth=1.5)
    ax.add_patch(server_box)
    ax.text(3.95, 6.5, 'Servers\n(K total)', ha='center', va='center', fontsize=9, fontweight='bold')

    # Server icons
    for i in range(4):
        y_pos = 5.8 - i * 0.9
        server = FancyBboxPatch((3.4, y_pos), 1.1, 0.6,
                                boxstyle="round,pad=0.02",
                                edgecolor='darkviolet', facecolor='plum',
                                linewidth=1)
        ax.add_patch(server)
        ax.text(3.95, y_pos + 0.3, f'S{i+1}', ha='center', va='center', fontsize=7)

    # ===== Right Side: DRL Agent Components =====

    # Agent box (large container)
    agent_box = FancyBboxPatch((7.0, 1.5), 4.5, 6.5,
                               boxstyle="round,pad=0.1",
                               edgecolor='darkred', facecolor='mistyrose',
                               linewidth=2, alpha=0.3)
    ax.add_patch(agent_box)
    ax.text(9.25, 7.7, 'DRL Agent',
            ha='center', fontweight='bold', fontsize=12)

    # Neural Network (Actor)
    actor_box = FancyBboxPatch((7.5, 5.8), 3.5, 1.4,
                               boxstyle="round,pad=0.05",
                               edgecolor='red', facecolor='lightcoral',
                               linewidth=1.5)
    ax.add_patch(actor_box)
    ax.text(9.25, 6.7, 'Actor Network', ha='center', fontweight='bold', fontsize=10)
    ax.text(9.25, 6.3, '(Policy π)', ha='center', fontsize=8, style='italic')

    # Neural Network (Critic)
    critic_box = FancyBboxPatch((7.5, 4.0), 3.5, 1.4,
                                boxstyle="round,pad=0.05",
                                edgecolor='red', facecolor='lightcoral',
                                linewidth=1.5)
    ax.add_patch(critic_box)
    ax.text(9.25, 4.9, 'Critic Network', ha='center', fontweight='bold', fontsize=10)
    ax.text(9.25, 4.5, '(Value V)', ha='center', fontsize=8, style='italic')

    # Replay Buffer
    buffer_box = FancyBboxPatch((7.5, 2.2), 3.5, 1.3,
                                boxstyle="round,pad=0.05",
                                edgecolor='orange', facecolor='wheat',
                                linewidth=1.5)
    ax.add_patch(buffer_box)
    ax.text(9.25, 3.0, 'Replay Buffer', ha='center', fontweight='bold', fontsize=10)
    ax.text(9.25, 2.6, '(s, a, r, s')', ha='center', fontsize=8, style='italic')

    # ===== Middle: Information Flow =====

    # State box
    state_box = FancyBboxPatch((5.3, 5.8), 1.2, 0.8,
                               boxstyle="round,pad=0.05",
                               edgecolor='blue', facecolor='lightblue',
                               linewidth=1.5)
    ax.add_patch(state_box)
    ax.text(5.9, 6.2, 'State', ha='center', fontweight='bold', fontsize=9)
    ax.text(5.9, 5.95, 's_t', ha='center', fontsize=7, style='italic')

    # Action box
    action_box = FancyBboxPatch((5.3, 4.3), 1.2, 0.8,
                                boxstyle="round,pad=0.05",
                                edgecolor='red', facecolor='lightyellow',
                                linewidth=1.5)
    ax.add_patch(action_box)
    ax.text(5.9, 4.7, 'Action', ha='center', fontweight='bold', fontsize=9)
    ax.text(5.9, 4.45, 'a_t', ha='center', fontsize=7, style='italic')

    # Reward box
    reward_box = FancyBboxPatch((5.3, 2.8), 1.2, 0.8,
                                boxstyle="round,pad=0.05",
                                edgecolor='green', facecolor='lightgreen',
                                linewidth=1.5)
    ax.add_patch(reward_box)
    ax.text(5.9, 3.2, 'Reward', ha='center', fontweight='bold', fontsize=9)
    ax.text(5.9, 2.95, 'r_t', ha='center', fontsize=7, style='italic')

    # ===== Arrows: Information Flow =====

    # Environment -> State
    arrow1 = FancyArrowPatch((4.7, 4.5), (5.3, 6.0),
                             arrowstyle='->', lw=2, color='blue',
                             connectionstyle="arc3,rad=0.2")
    ax.add_patch(arrow1)

    # State -> Actor
    arrow2 = FancyArrowPatch((6.5, 6.2), (7.5, 6.5),
                             arrowstyle='->', lw=2, color='blue')
    ax.add_patch(arrow2)

    # Actor -> Action
    arrow3 = FancyArrowPatch((7.5, 6.3), (6.5, 4.7),
                             arrowstyle='->', lw=2, color='red',
                             connectionstyle="arc3,rad=0.3")
    ax.add_patch(arrow3)

    # Action -> Environment
    arrow4 = FancyArrowPatch((5.3, 4.5), (4.7, 4.0),
                             arrowstyle='->', lw=2, color='red')
    ax.add_patch(arrow4)

    # Environment -> Reward
    arrow5 = FancyArrowPatch((4.7, 3.5), (5.3, 3.2),
                             arrowstyle='->', lw=2, color='green')
    ax.add_patch(arrow5)

    # Reward -> Critic
    arrow6 = FancyArrowPatch((6.5, 3.2), (7.5, 4.7),
                             arrowstyle='->', lw=2, color='green',
                             connectionstyle="arc3,rad=-0.2")
    ax.add_patch(arrow6)

    # Cycle annotation
    ax.text(6.0, 8.5, 'Training Loop', ha='center', fontsize=11,
            fontweight='bold', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    return fig, ax

if __name__ == '__main__':
    fig, ax = create_architecture_diagram()
    plt.savefig(OUTPUT_FILE, dpi=DPI, bbox_inches='tight', facecolor='white')
    print(f"System architecture diagram saved to {OUTPUT_FILE}")
