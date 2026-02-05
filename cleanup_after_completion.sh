#!/bin/bash
# Cleanup script after experiment completion
# Usage: ./cleanup_after_completion.sh

echo "=== Post-experiment cleanup script ==="
echo ""

# Check if all experiments are complete
echo "1. Checking experiment integrity..."
python verify_experiments.py > /dev/null 2>&1

if [ $? -eq 0 ]; then
    echo "✓ Experiment verification passed"
else
    echo "⚠️  Please ensure all experiments are complete first"
    exit 1
fi

echo ""
echo "2. Preparing to clean temporary files..."
echo ""

# Show content to be deleted
echo "Folders to be deleted:"
echo "  - server_backup_20260128/ (108MB)"
echo ""

# Show content to be kept
echo "Data to keep:"
echo "  - Data/hca2c_final_comparison/ (all 45 experiments)"
echo "  - Data/hca2c_final_comparison_local/ (local experiment results)"
echo ""

read -p "Confirm deletion of server_backup_20260128/? (y/N): " confirm

if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
    echo ""
    echo "Deleting..."
    rm -rf server_backup_20260128/
    echo "✓ Cleanup complete"
    echo ""
    echo "Remaining data:"
    du -sh Data/hca2c_final_comparison*
else
    echo "Cleanup cancelled"
fi
