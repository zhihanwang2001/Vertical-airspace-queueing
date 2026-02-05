#!/bin/bash
# Monitor current experiment and automatically run remaining experiments after completion

echo "=== Experiment monitoring and auto-continue script ==="
echo "Start time: $(date '+%H:%M:%S')"
echo ""

# Wait for current experiment to complete
echo "Waiting for HCA2C seed45 load7.0 to complete..."
while ps aux | grep "69608.*python" | grep -v grep > /dev/null; do
    sleep 30
done

echo ""
echo "✓ HCA2C seed45 load7.0 completed!"
echo "Completion time: $(date '+%H:%M:%S')"
echo ""

# Check result file
if [ -f "Data/hca2c_final_comparison_local/HCA2C_seed45_load7.0.json" ]; then
    echo "✓ Result file generated"
    ls -lh Data/hca2c_final_comparison_local/HCA2C_seed45_load7.0.json
else
    echo "⚠️  Result file not found"
fi

echo ""
echo "=== Starting remaining 5 experiments ==="
echo ""

# Run remaining experiments
python run_remaining_experiments.py

echo ""
echo "=== All experiments complete! ==="
echo "Completion time: $(date '+%H:%M:%S')"
echo ""

# Show results
echo "Local experiment results:"
ls -lh Data/hca2c_final_comparison_local/

echo ""
echo "Next step: Run ./move_local_results.sh to move results to main directory"
