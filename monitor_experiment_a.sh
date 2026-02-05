#!/bin/bash
# Experiment A Monitoring Script

PASSWORD='uNBRd68Bzc5hhDZF2ZpCdZKF6pMXeK83'
HOST='i-1.gpushare.com'
PORT=60899

echo "========================================="
echo "Experiment A Monitor"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================="

# 1. Check process status
echo ""
echo "1. Process status:"
sshpass -p "$PASSWORD" ssh -p $PORT root@$HOST \
    "ps aux | grep 'run_structural_comparison_5x_load' | grep -v grep" 2>/dev/null || echo "   Process not running"

# 2. Count completed experiments
echo ""
echo "2. Completion progress:"
COMPLETED=$(sshpass -p "$PASSWORD" ssh -p $PORT root@$HOST \
    "grep -c '✅ \[' /root/RP1/logs/experiment_a_5x_load.log 2>/dev/null" || echo "0")
echo "   Completed: $COMPLETED / 12"

# 3. Count result files
echo ""
echo "3. Result files:"
FILE_COUNT=$(sshpass -p "$PASSWORD" ssh -p $PORT root@$HOST \
    "ls /root/RP1/Data/ablation_studies/structural_5x_load/*/*.json 2>/dev/null | wc -l" || echo "0")
echo "   File count: $FILE_COUNT"

# 4. View latest progress
echo ""
echo "4. Latest progress (last 20 lines):"
sshpass -p "$PASSWORD" ssh -p $PORT root@$HOST \
    "tail -20 /root/RP1/logs/experiment_a_5x_load.log" 2>/dev/null

# 5. GPU usage
echo ""
echo "========================================="
echo "5. GPU status:"
sshpass -p "$PASSWORD" ssh -p $PORT root@$HOST "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader" 2>/dev/null

echo ""
echo "========================================="
echo "Monitoring complete - $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================="
