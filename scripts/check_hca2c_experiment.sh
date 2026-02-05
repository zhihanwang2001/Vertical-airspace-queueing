#!/bin/bash
# HCA2C experiment monitoring script
# Usage: ./scripts/check_hca2c_experiment.sh

SERVER="root@i-2.gpushare.com"
PORT="23937"
PASSWORD="Wtrp2NWaaqcW7merrFR3v6H6MXcQ9cgP"

echo "=========================================="
echo "HCA2C Experiment Progress Monitor"
echo "=========================================="
echo ""

# Check process status
echo "【Process Status】"
sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no -p $PORT $SERVER "
ps aux | grep 'python3 -u run_final' | grep -v grep | awk '{print \"  PID: \" \$2 \", CPU: \" \$3 \"%, MEM: \" \$4 \"%, Runtime: \" \$10}'
" || echo "  ✗ Process not running"

echo ""

# Check training progress
echo "【Training Progress】"
sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no -p $PORT $SERVER "
tail -3 /root/hca2c_experiment.log
"

echo ""

# Check completion status
echo "【Completion Status】"
result_count=$(sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no -p $PORT $SERVER "ls /root/Data/hca2c_final_comparison/*.json 2>/dev/null | wc -l")
echo "  Completed: $result_count / 45 runs"

if [ "$result_count" -gt 0 ]; then
    echo ""
    echo "【Latest Results】"
    sshpass -p "$PASSWORD" ssh -o StrictHostKeyChecking=no -p $PORT $SERVER "
    ls -lht /root/Data/hca2c_final_comparison/*.json 2>/dev/null | head -5 | awk '{print \"  \" \$9}'
    "
fi

echo ""
echo "=========================================="
echo "Note: Experiment estimated to take 46 hours"
echo "=========================================="
