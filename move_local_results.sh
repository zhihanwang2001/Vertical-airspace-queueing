#!/bin/bash
# Move local experiment results to main data directory
# Usage: ./move_local_results.sh

echo "=== Move local experiment results to main data directory ==="
echo ""

LOCAL_DIR="Data/hca2c_final_comparison_local"
MAIN_DIR="Data/hca2c_final_comparison"

# Check local experiment directory
if [ ! -d "$LOCAL_DIR" ]; then
    echo "❌ Local experiment directory does not exist: $LOCAL_DIR"
    exit 1
fi

# Count local experiment files
LOCAL_FILES=$(ls -1 "$LOCAL_DIR"/*.json 2>/dev/null | wc -l)
echo "Local experiment results: $LOCAL_FILES JSON files"
echo ""

if [ $LOCAL_FILES -eq 0 ]; then
    echo "⚠️  No local experiment results found"
    exit 1
fi

# Show files to be moved
echo "Files to be moved:"
ls -1 "$LOCAL_DIR"/*.json 2>/dev/null | xargs -n1 basename
echo ""

read -p "Confirm moving these files to $MAIN_DIR? (y/N): " confirm

if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
    echo ""
    echo "Moving files..."

    # Move all files
    mv "$LOCAL_DIR"/* "$MAIN_DIR"/ 2>/dev/null

    if [ $? -eq 0 ]; then
        echo "✓ Files moved successfully"
        echo ""
        echo "Verify results:"
        python verify_experiments.py
    else
        echo "❌ Move failed"
        exit 1
    fi
else
    echo "Move cancelled"
fi
