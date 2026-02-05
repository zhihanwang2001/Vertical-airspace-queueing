"""
Beautiful 3D Vertical Stratified Queue System Structure (Figure 1)
Beautiful 3D Vertical Stratified Queue System Structure
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import seaborn as sns

# 设置绘图风格
sns.set_style("white")
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

# 创建图形
fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(111, projection='3d')

# 层配置（倒金字塔）
layers = [
    {'name': 'Layer 5', 'height': 100, 'capacity': 8, 'color': '#3498DB', 'alpha': 0.25},
    {'name': 'Layer 4', 'height': 80, 'capacity': 6, 'color': '#2ECC71', 'alpha': 0.3},
    {'name': 'Layer 3', 'height': 60, 'capacity': 4, 'color': '#F39C12', 'alpha': 0.35},
    {'name': 'Layer 2', 'height': 40, 'capacity': 3, 'color': '#E74C3C', 'alpha': 0.4},
    {'name': 'Layer 1', 'height': 20, 'capacity': 2, 'color': '#9B59B6', 'alpha': 0.45},
]

# 绘制每一层（倒金字塔形状：容量越大，层越宽）
for i, layer in enumerate(layers):
    z = layer['height']
    capacity = layer['capacity']

    # 根据容量计算层的大小（倒金字塔：容量大→面积大）
    size = capacity * 1.5  # 缩放因子

    # 定义层的四个角点
    x_range = [-size, size, size, -size, -size]
    y_range = [-size, -size, size, size, -size]
    z_range = [z, z, z, z, z]

    # 绘制层的底面
    verts = [list(zip(x_range[:-1], y_range[:-1], z_range[:-1]))]
    poly = Poly3DCollection(verts, alpha=layer['alpha'],
                           facecolor=layer['color'],
                           edgecolor='#2C3E50', linewidth=2.5)
    ax.add_collection3d(poly)

    # 绘制层的边框（加粗）
    ax.plot(x_range, y_range, z_range, color='#2C3E50', linewidth=3, alpha=0.9)

    # 添加层标签
    ax.text(size + 2, 0, z,
            f'{layer["name"]}\n{z}m\nC={capacity}',
            fontsize=13, fontweight='bold', color='#2C3E50',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                     edgecolor=layer['color'], linewidth=2.5, alpha=0.95))

    # 在层中心绘制UAV示意（小球）
    num_uavs = min(capacity // 2 + 1, 4)  # 示意性显示UAV数量
    for j in range(num_uavs):
        theta = 2 * np.pi * j / num_uavs
        r = size * 0.5
        x_uav = r * np.cos(theta)
        y_uav = r * np.sin(theta)
        ax.scatter([x_uav], [y_uav], [z + 0.5],
                  c=layer['color'], s=300, marker='o',
                  edgecolors='#2C3E50', linewidths=2, alpha=0.9)

# 绘制层间传输箭头
for i in range(len(layers) - 1):
    z_from = layers[i]['height']
    z_to = layers[i + 1]['height']

    # 向上传输箭头
    ax.quiver(8, 8, z_from + 1, 0, 0, z_to - z_from - 2,
             color='#27AE60', arrow_length_ratio=0.15, linewidth=3, alpha=0.8)

    # 向下传输箭头
    ax.quiver(-8, -8, z_to - 1, 0, 0, -(z_to - z_from - 2),
             color='#E67E22', arrow_length_ratio=0.15, linewidth=3, alpha=0.8)

# 添加到达流标注
ax.text(-15, -15, 0, 'Arrival\nStream', fontsize=14, fontweight='bold',
        color='#8E44AD',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='#F4ECF7',
                 edgecolor='#8E44AD', linewidth=2.5))
ax.quiver(-13, -13, 0, 6, 6, 5, color='#8E44AD', arrow_length_ratio=0.2,
         linewidth=3.5, alpha=0.9)

# 添加离开流标注
ax.text(15, 15, 105, 'Departure\nStream', fontsize=14, fontweight='bold',
        color='#16A085',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='#E8F8F5',
                 edgecolor='#16A085', linewidth=2.5))

# 添加图例
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w',
              markerfacecolor='#3498DB', markersize=12,
              markeredgecolor='#2C3E50', markeredgewidth=2,
              label='UAV (Drone)'),
    plt.Line2D([0], [0], color='#27AE60', linewidth=3,
              label='Upward Transfer'),
    plt.Line2D([0], [0], color='#E67E22', linewidth=3,
              label='Downward Transfer'),
]

ax.legend(handles=legend_elements, loc='upper left', fontsize=13,
         framealpha=0.95, edgecolor='#2C3E50', fancybox=True, shadow=True)

# 设置轴标签和标题
ax.set_xlabel('X-axis (m)', fontsize=14, fontweight='bold', color='#2C3E50', labelpad=10)
ax.set_ylabel('Y-axis (m)', fontsize=14, fontweight='bold', color='#2C3E50', labelpad=10)
ax.set_zlabel('Altitude (m)', fontsize=14, fontweight='bold', color='#2C3E50', labelpad=10)

ax.set_title('Vertical Stratified Queue System: Inverted Pyramid Capacity Profile\n'
             'MCRPS/D/K Framework for UAV Airspace Management',
             fontsize=18, fontweight='bold', pad=30, color='#2C3E50')

# 设置视角
ax.view_init(elev=25, azim=45)

# 设置轴范围
ax.set_xlim([-18, 18])
ax.set_ylim([-18, 18])
ax.set_zlim([0, 120])

# 背景颜色
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('#BDC3C7')
ax.yaxis.pane.set_edgecolor('#BDC3C7')
ax.zaxis.pane.set_edgecolor('#BDC3C7')

# 网格
ax.grid(True, alpha=0.2, linestyle='--', linewidth=1)

plt.tight_layout()
plt.savefig('../../Figures/analysis/figure3_3d_structure.png', dpi=400, bbox_inches='tight', facecolor='white')
print("✅ Beautiful 3D Structure Figure Saved: figure3_3d_structure.png")
plt.close()
