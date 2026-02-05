#!/bin/bash
echo "=== 实验进度检查 ==="
echo "时间: $(date '+%H:%M:%S')"
echo ""

# 检查进程
if ps aux | grep "python run_remaining" | grep -v grep > /dev/null; then
    echo "✓ 实验正在运行"
    ps aux | grep "python run_remaining" | grep -v grep | awk '{print "  PID: " $2 " | CPU: " $3 "% | Memory: " $4 "%"}'
else
    echo "✗ 实验未运行"
fi

echo ""
echo "=== 已完成的实验 ==="
ls -1 Data/hca2c_final_comparison_local/*.json 2>/dev/null | wc -l | xargs echo "本地完成:"
ls -1 Data/hca2c_final_comparison_local/*.json 2>/dev/null | xargs -n1 basename

echo ""
echo "=== 最新日志 (最后10行) ==="
tail -10 remaining_experiments.log 2>/dev/null || echo "无日志"
