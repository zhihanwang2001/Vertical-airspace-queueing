#!/bin/bash
# Wait for experiment completion and notify

echo "Waiting for HCA2C seed45 load7.0 to complete..."
echo "Estimated remaining time: ~3 minutes"
echo ""

# Wait for process to end
while ps aux | grep "69608.*python" | grep -v grep > /dev/null; do
    sleep 10
done

echo ""
echo "🎉 HCA2C seed45 load7.0 completed!"
echo "Completion time: $(date '+%H:%M:%S')"
echo ""

# Check results
if [ -f "Data/hca2c_final_comparison_local/HCA2C_seed45_load7.0.json" ]; then
    echo "✓ Result file generated"
    ls -lh Data/hca2c_final_comparison_local/
    echo ""
    echo "View results:"
    cat Data/hca2c_final_comparison_local/HCA2C_seed45_load7.0.json | python -m json.tool | head -20
else
    echo "⚠️  Result file not found, check logs:"
    tail -50 hca2c_seed45_load7.log
fi

echo ""
echo "Next step: Run remaining 5 experiments"
echo "Command: python run_remaining_experiments.py"
