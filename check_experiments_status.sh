#!/bin/bash
echo "=== Experiment progress check ==="
echo "Time: $(date '+%H:%M:%S')"
echo ""

# Check process
if ps aux | grep "python run_remaining" | grep -v grep > /dev/null; then
    echo "✓ Experiment is running"
    ps aux | grep "python run_remaining" | grep -v grep | awk '{print "  PID: " $2 " | CPU: " $3 "% | Memory: " $4 "%"}'
else
    echo "✗ Experiment not running"
fi

echo ""
echo "=== Completed experiments ==="
ls -1 Data/hca2c_final_comparison_local/*.json 2>/dev/null | wc -l | xargs echo "Local completed:"
ls -1 Data/hca2c_final_comparison_local/*.json 2>/dev/null | xargs -n1 basename

echo ""
echo "=== Latest log (last 10 lines) ==="
tail -10 remaining_experiments.log 2>/dev/null || echo "No log"
