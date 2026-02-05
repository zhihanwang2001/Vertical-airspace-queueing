#!/bin/bash
# 监控当前实验并在完成后自动运行剩余实验

echo "=== 实验监控和自动继续脚本 ==="
echo "开始时间: $(date '+%H:%M:%S')"
echo ""

# 等待当前实验完成
echo "等待 HCA2C seed45 load7.0 完成..."
while ps aux | grep "69608.*python" | grep -v grep > /dev/null; do
    sleep 30
done

echo ""
echo "✓ HCA2C seed45 load7.0 已完成!"
echo "完成时间: $(date '+%H:%M:%S')"
echo ""

# 检查结果文件
if [ -f "Data/hca2c_final_comparison_local/HCA2C_seed45_load7.0.json" ]; then
    echo "✓ 结果文件已生成"
    ls -lh Data/hca2c_final_comparison_local/HCA2C_seed45_load7.0.json
else
    echo "⚠️  结果文件未找到"
fi

echo ""
echo "=== 开始运行剩余5个实验 ==="
echo ""

# 运行剩余实验
python run_remaining_experiments.py

echo ""
echo "=== 所有实验完成! ==="
echo "完成时间: $(date '+%H:%M:%S')"
echo ""

# 显示结果
echo "本地实验结果:"
ls -lh Data/hca2c_final_comparison_local/

echo ""
echo "下一步: 运行 ./move_local_results.sh 移动结果到主目录"
