#!/bin/bash
# Auto-monitor experiments and run analysis when complete

echo "=========================================="
echo "Experiment Auto-Monitor"
echo "=========================================="
echo ""

LOG_FILE="rerun_experiments_fixed.log"
DATA_DIR="Data/hca2c_final_comparison"
CHECK_INTERVAL=120  # Check every 2 minutes

while true; do
    # Check if process is running
    if ps aux | grep -q "[p]ython3.*rerun_a2c_ppo"; then
        # Count completed experiments (files modified in last 4 hours)
        COMPLETED=$(find "$DATA_DIR" -name "*.json" -mmin -240 | wc -l | tr -d ' ')

        echo "[$(date '+%H:%M:%S')] Process running - $COMPLETED files updated recently"

        # Show last few lines of log
        tail -5 "$LOG_FILE" 2>/dev/null | grep -E "(Running:|Completed:|Mean reward:)" | tail -2

    else
        echo "[$(date '+%H:%M:%S')] Process completed!"

        # Check if all experiments succeeded
        COMPLETED=$(find "$DATA_DIR" -name "*.json" -mmin -240 | wc -l | tr -d ' ')
        echo "Files updated: $COMPLETED"

        if [ "$COMPLETED" -ge 20 ]; then
            echo ""
            echo "✓ All experiments completed successfully!"
            echo ""
            echo "Running analysis..."
            python3 Analysis/statistical_analysis/analyze_hca2c_ablation.py

            echo ""
            echo "=========================================="
            echo "✓ ANALYSIS COMPLETE"
            echo "=========================================="
            echo ""
            echo "Next steps:"
            echo "1. Review: Analysis/statistical_reports/hca2c_ablation_report.md"
            echo "2. Check: Analysis/figures/hca2c_ablation_comprehensive.png"
            echo "3. Update manuscript with final statistics"

        else
            echo ""
            echo "⚠️  Warning: Only $COMPLETED experiments completed"
            echo "Check $LOG_FILE for errors"
        fi

        break
    fi

    sleep $CHECK_INTERVAL
done
