#!/bin/bash
# 实验A监控脚本
# Experiment A Monitoring Script

PASSWORD='uNBRd68Bzc5hhDZF2ZpCdZKF6pMXeK83'
HOST='i-1.gpushare.com'
PORT=60899

echo "========================================="
echo "实验A监控 - Experiment A Monitor"
echo "时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================="

# 1. 检查进程状态
echo ""
echo "1. 进程状态:"
sshpass -p "$PASSWORD" ssh -p $PORT root@$HOST \
    "ps aux | grep 'run_structural_comparison_5x_load' | grep -v grep" 2>/dev/null || echo "   进程未运行"

# 2. 统计完成数量
echo ""
echo "2. 完成进度:"
COMPLETED=$(sshpass -p "$PASSWORD" ssh -p $PORT root@$HOST \
    "grep -c '✅ \[' /root/RP1/logs/experiment_a_5x_load.log 2>/dev/null" || echo "0")
echo "   已完成: $COMPLETED / 12"

# 3. 统计结果文件
echo ""
echo "3. 结果文件:"
FILE_COUNT=$(sshpass -p "$PASSWORD" ssh -p $PORT root@$HOST \
    "ls /root/RP1/Data/ablation_studies/structural_5x_load/*/*.json 2>/dev/null | wc -l" || echo "0")
echo "   文件数: $FILE_COUNT"

# 4. 查看最新进度
echo ""
echo "4. 最新进度 (最后20行):"
sshpass -p "$PASSWORD" ssh -p $PORT root@$HOST \
    "tail -20 /root/RP1/logs/experiment_a_5x_load.log" 2>/dev/null

# 5. GPU使用情况
echo ""
echo "========================================="
echo "5. GPU状态:"
sshpass -p "$PASSWORD" ssh -p $PORT root@$HOST "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader" 2>/dev/null

echo ""
echo "========================================="
echo "监控完成 - $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================="
