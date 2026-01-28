#!/bin/bash
# HCA2C实验监控脚本
# 用法: ./scripts/check_hca2c_experiment.sh

SERVER="root@i-2.gpushare.com"
PORT="23937"
PASSWORD="Wtrp2NWaaqcW7merrFR3v6H6MXcQ9cgP"

echo "=========================================="
echo "HCA2C实验进度监控"
echo "=========================================="
echo ""

# 检查进程状态
echo "【进程状态】"
sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no -p $PORT $SERVER "
ps aux | grep 'python3 -u run_final' | grep -v grep | awk '{print \"  PID: \" \$2 \", CPU: \" \$3 \"%, MEM: \" \$4 \"%, 运行时间: \" \$10}'
" || echo "  ✗ 进程未运行"

echo ""

# 检查训练进度
echo "【训练进度】"
sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no -p $PORT $SERVER "
tail -3 /root/hca2c_experiment.log
"

echo ""

# 检查完成情况
echo "【完成情况】"
result_count=$(sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no -p $PORT $SERVER "ls /root/Data/hca2c_final_comparison/*.json 2>/dev/null | wc -l")
echo "  已完成: $result_count / 45 runs"

if [ "$result_count" -gt 0 ]; then
    echo ""
    echo "【最新结果】"
    sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no -p $PORT $SERVER "
    ls -lht /root/Data/hca2c_final_comparison/*.json 2>/dev/null | head -5 | awk '{print \"  \" \$9}'
    "
fi

echo ""
echo "=========================================="
echo "提示: 实验预计需要46小时完成"
echo "=========================================="
