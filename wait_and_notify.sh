#!/bin/bash
# 等待实验完成并通知

echo "等待 HCA2C seed45 load7.0 完成..."
echo "预计剩余时间: ~3分钟"
echo ""

# 等待进程结束
while ps aux | grep "69608.*python" | grep -v grep > /dev/null; do
    sleep 10
done

echo ""
echo "🎉 HCA2C seed45 load7.0 已完成!"
echo "完成时间: $(date '+%H:%M:%S')"
echo ""

# 检查结果
if [ -f "Data/hca2c_final_comparison_local/HCA2C_seed45_load7.0.json" ]; then
    echo "✓ 结果文件已生成"
    ls -lh Data/hca2c_final_comparison_local/
    echo ""
    echo "查看结果:"
    cat Data/hca2c_final_comparison_local/HCA2C_seed45_load7.0.json | python -m json.tool | head -20
else
    echo "⚠️  结果文件未找到，检查日志:"
    tail -50 hca2c_seed45_load7.log
fi

echo ""
echo "下一步: 运行剩余5个实验"
echo "命令: python run_remaining_experiments.py"
