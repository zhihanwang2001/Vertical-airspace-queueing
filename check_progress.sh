#!/bin/bash
echo "=== Experiment Progress Check ==="
echo "Time: $(date '+%H:%M:%S')"
echo ""

# Check if process is running
if ps aux | grep "69608.*python" | grep -v grep > /dev/null; then
    echo "✓ HCA2C seed45 load7.0 is running"
    ps aux | grep "69608.*python" | grep -v grep | awk '{print "  CPU: " $3 "% | Memory: " $4 "%"}'
    
    # Calculate elapsed time
    START_TIME="23:33:00"
    CURRENT=$(date +%s)
    START=$(date -j -f "%H:%M:%S" "$START_TIME" +%s 2>/dev/null || echo $CURRENT)
    ELAPSED=$((CURRENT - START))
    MINUTES=$((ELAPSED / 60))
    SECONDS=$((ELAPSED % 60))
    
    echo "  Elapsed: ${MINUTES}m ${SECONDS}s"
    echo "  Expected: ~17 minutes"
    
    REMAINING=$((17 * 60 - ELAPSED))
    if [ $REMAINING -gt 0 ]; then
        REM_MIN=$((REMAINING / 60))
        REM_SEC=$((REMAINING % 60))
        echo "  Remaining: ~${REM_MIN}m ${REM_SEC}s"
    else
        echo "  Should be completing soon..."
    fi
else
    echo "✗ Process not running"
fi

echo ""

# Check for results
if [ -f "Data/hca2c_final_comparison_local/HCA2C_seed45_load7.0.json" ]; then
    echo "✓ Results file created!"
    ls -lh Data/hca2c_final_comparison_local/
else
    echo "⏳ Waiting for results file..."
fi
