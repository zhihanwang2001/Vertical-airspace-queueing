#!/bin/bash
# 将本地实验结果移动到主数据目录
# 使用方法: ./move_local_results.sh

echo "=== 移动本地实验结果到主数据目录 ==="
echo ""

LOCAL_DIR="Data/hca2c_final_comparison_local"
MAIN_DIR="Data/hca2c_final_comparison"

# 检查本地实验目录
if [ ! -d "$LOCAL_DIR" ]; then
    echo "❌ 本地实验目录不存在: $LOCAL_DIR"
    exit 1
fi

# 统计本地实验文件
LOCAL_FILES=$(ls -1 "$LOCAL_DIR"/*.json 2>/dev/null | wc -l)
echo "本地实验结果: $LOCAL_FILES 个JSON文件"
echo ""

if [ $LOCAL_FILES -eq 0 ]; then
    echo "⚠️  没有找到本地实验结果"
    exit 1
fi

# 显示将要移动的文件
echo "将要移动的文件:"
ls -1 "$LOCAL_DIR"/*.json 2>/dev/null | xargs -n1 basename
echo ""

read -p "确认移动这些文件到 $MAIN_DIR? (y/N): " confirm

if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
    echo ""
    echo "正在移动文件..."
    
    # 移动所有文件
    mv "$LOCAL_DIR"/* "$MAIN_DIR"/ 2>/dev/null
    
    if [ $? -eq 0 ]; then
        echo "✓ 文件移动完成"
        echo ""
        echo "验证结果:"
        python verify_experiments.py
    else
        echo "❌ 移动失败"
        exit 1
    fi
else
    echo "取消移动"
fi
