"""
Elegant Vertical Stratified Queue System Structure (图1)
Elegant Vertical Stratified Queue System Structure
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.patches import Circle, Rectangle, Wedge
import numpy as np
import seaborn as sns

# Set plotting style
sns.set_style("white")
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

# Create figure (2D side view, clearer)
fig, ax = plt.subplots(figsize=(18, 12))

# Layer configuration (inverted pyramid)
layers_data = [
    {'name': 'Layer 5', 'altitude': 100, 'capacity': 8, 'service_rate': 1.20,
     'color': '#3498DB', 'y_pos': 5},
    {'name': 'Layer 4', 'altitude': 80, 'capacity': 6, 'service_rate': 1.00,
     'color': '#1ABC9C', 'y_pos': 4},
    {'name': 'Layer 3', 'altitude': 60, 'capacity': 4, 'service_rate': 0.80,
     'color': '#F39C12', 'y_pos': 3},
    {'name': 'Layer 2', 'altitude': 40, 'capacity': 3, 'service_rate': 0.60,
     'color': '#E74C3C', 'y_pos': 2},
    {'name': 'Layer 1', 'altitude': 20, 'capacity': 2, 'service_rate': 0.40,
     'color': '#9B59B6', 'y_pos': 1},
]

# Draw each layer (inverted pyramid: width proportional to capacity)
base_width = 3
for i, layer in enumerate(layers_data):
    y = layer['y_pos']
    capacity = layer['capacity']

    # Calculate width based on capacity (inverted pyramid)
    width = base_width + capacity * 0.8
    height = 0.6
    x_center = 10

    # Draw layer rectangle
    rect = FancyBboxPatch((x_center - width/2, y - height/2), width, height,
                          boxstyle="round,pad=0.05",
                          facecolor=layer['color'],
                          edgecolor='#2C3E50',
                          linewidth=3.5, alpha=0.7, zorder=2)
    ax.add_patch(rect)

    # Add layer information text (left side)
    info_text = (f"{layer['name']}\n"
                f"Altitude: {layer['altitude']}m\n"
                f"Capacity: C={capacity}\n"
                f"Service: μ={layer['service_rate']}")

    ax.text(x_center - width/2 - 2.5, y, info_text,
           fontsize=11, fontweight='bold', color='#2C3E50',
           verticalalignment='center',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                    edgecolor=layer['color'], linewidth=2.5, alpha=0.95))

    # Draw UAV icons inside layer (small drones)
    num_uavs = min(capacity // 2 + 1, capacity)
    spacing = width / (num_uavs + 1)
    for j in range(num_uavs):
        x_uav = x_center - width/2 + spacing * (j + 1)
        # Draw simplified drone icon
        ax.scatter([x_uav], [y], s=400, c='white', marker='o',
                  edgecolors='#2C3E50', linewidths=2.5, zorder=3, alpha=0.95)
        ax.scatter([x_uav], [y], s=150, c=layer['color'], marker='s',
                  edgecolors='#2C3E50', linewidths=1.5, zorder=4, alpha=1)

    # Add capacity indicator bar (right side)
    bar_x = x_center + width/2 + 1.5
    bar_width = 0.3
    bar_height = capacity * 0.08
    capacity_bar = Rectangle((bar_x, y - bar_height/2), bar_width, bar_height,
                             facecolor=layer['color'], edgecolor='#2C3E50',
                             linewidth=2, alpha=0.8)
    ax.add_patch(capacity_bar)
    ax.text(bar_x + bar_width + 0.3, y, f'{capacity}',
           fontsize=12, fontweight='bold', color='#2C3E50',
           verticalalignment='center')

# Draw inter-layer transfer arrows
for i in range(len(layers_data) - 1):
    y_from = layers_data[i]['y_pos']
    y_to = layers_data[i + 1]['y_pos']

    # Upward transfer arrow (right side)
    arrow_up = FancyArrowPatch((15.5, y_from + 0.2), (15.5, y_to - 0.2),
                              arrowstyle='->,head_width=0.6,head_length=0.4',
                              color='#27AE60', linewidth=3.5, alpha=0.85,
                              zorder=1)
    ax.add_patch(arrow_up)

    # Downward transfer arrow (left side)
    arrow_down = FancyArrowPatch((4.5, y_to - 0.2), (4.5, y_from + 0.2),
                                arrowstyle='->,head_width=0.6,head_length=0.4',
                                color='#E67E22', linewidth=3.5, alpha=0.85,
                                zorder=1)
    ax.add_patch(arrow_down)

# Add transfer labels
ax.text(15.5, 3.5, 'Upward\nTransfer', fontsize=11, fontweight='bold',
       color='#27AE60', horizontalalignment='center',
       bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8F8F5',
                edgecolor='#27AE60', linewidth=2))

ax.text(4.5, 3.5, 'Downward\nTransfer', fontsize=11, fontweight='bold',
       color='#E67E22', horizontalalignment='center',
       bbox=dict(boxstyle='round,pad=0.4', facecolor='#FEF5E7',
                edgecolor='#E67E22', linewidth=2))

# Add arrival stream (bottom)
arrival_arrow = FancyArrowPatch((10, -0.5), (10, 0.7),
                               arrowstyle='->,head_width=0.8,head_length=0.5',
                               color='#8E44AD', linewidth=4.5, alpha=0.9,
                               zorder=1)
ax.add_patch(arrival_arrow)

ax.text(10, -1, 'Arrival Stream\n(3 Classes: Standard/Priority/Emergency)',
       fontsize=12, fontweight='bold', color='#8E44AD',
       horizontalalignment='center',
       bbox=dict(boxstyle='round,pad=0.6', facecolor='#F4ECF7',
                edgecolor='#8E44AD', linewidth=3))

# Add departure stream (top)
departure_arrow = FancyArrowPatch((10, 5.3), (10, 6.5),
                                 arrowstyle='->,head_width=0.8,head_length=0.5',
                                 color='#16A085', linewidth=4.5, alpha=0.9,
                                 zorder=1)
ax.add_patch(departure_arrow)

ax.text(10, 7, 'Departure Stream\n(Service Completed)',
       fontsize=12, fontweight='bold', color='#16A085',
       horizontalalignment='center',
       bbox=dict(boxstyle='round,pad=0.6', facecolor='#E8F8F5',
                edgecolor='#16A085', linewidth=3))

# Add inverted pyramid illustration (top right explanation)
pyramid_text = ("Inverted Pyramid Capacity:\n"
               "• Higher altitude → Larger capacity\n"
               "• Lower altitude → Tighter constraints\n"
               "• C = {2, 3, 4, 6, 8} (Low to High)")

ax.text(19.5, 5.5, pyramid_text,
       fontsize=11, fontweight='bold', color='#2C3E50',
       verticalalignment='top',
       bbox=dict(boxstyle='round,pad=0.7', facecolor='#FEF9E7',
                edgecolor='#D4AC0D', linewidth=3, alpha=0.95))

# Add system features description (top left)
system_text = ("MCRPS/D/K Framework:\n"
              "• Multi-Class Related Arrivals\n"
              "• Random Batch Service\n"
              "• Poisson Splitting\n"
              "• State-Dependent Control\n"
              "• Dynamic Transfer\n"
              "• Finite Capacity K")

ax.text(0.5, 5.5, system_text,
       fontsize=11, fontweight='bold', color='#2C3E50',
       verticalalignment='top',
       bbox=dict(boxstyle='round,pad=0.7', facecolor='#EBF5FB',
                edgecolor='#2874A6', linewidth=3, alpha=0.95))

# Set title
ax.set_title('Vertical Stratified Queue System Architecture\n'
            'Inverted Pyramid Capacity Profile for UAV Airspace Management',
            fontsize=20, fontweight='bold', pad=25, color='#2C3E50')

# Set axes
ax.set_xlim([0, 20])
ax.set_ylim([-2, 8])
ax.set_aspect('equal')

# Hide axes
ax.axis('off')

# Add grid background (faded)
ax.grid(False)

plt.tight_layout()
plt.savefig('figure3_3d_structure.png', dpi=400, bbox_inches='tight',
           facecolor='white', edgecolor='none')
print("✅ Elegant 3D Structure Figure Saved: figure3_3d_structure.png")
plt.close()
