#!/bin/bash
# 实验完成后的清理脚本
# 使用方法: ./cleanup_after_completion.sh

echo "=== 实验完成后清理脚本 ==="
echo ""

# 检查是否所有实验都完成
echo "1. 检查实验完整性..."
python verify_experiments.py > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "✓ 实验验证通过"
else
    echo "⚠️  请先确保所有实验完成"
    exit 1
fi

echo ""
echo "2. 准备清理临时文件..."
echo ""

# 显示将要删除的内容
echo "将要删除的文件夹:"
echo "  - server_backup_20260128/ (108MB)"
echo ""

# 显示保留的内容
echo "保留的数据:"
echo "  - Data/hca2c_final_comparison/ (所有45个实验)"
echo "  - Data/hca2c_final_comparison_local/ (本地实验结果)"
echo ""

read -p "确认删除 server_backup_20260128/? (y/N): " confirm

if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
    echo ""
    echo "正在删除..."
    rm -rf server_backup_20260128/
    echo "✓ 清理完成"
    echo ""
    echo "剩余数据:"
    du -sh Data/hca2c_final_comparison*
else
    echo "取消清理"
fi
