#!/bin/bash
# Quick Start Script for Ablation Studies
# Usage: ./start_ablation_studies.sh

set -e

echo "=========================================="
echo "HCA2C Ablation Studies - Quick Start"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "Code/training_scripts/run_ablation_studies.py" ]; then
    echo "Error: Please run this script from the project root directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Activate virtual environment
echo "1. Activating virtual environment..."
source .venv/bin/activate

# Check dependencies
echo "2. Checking dependencies..."
python -c "import torch; import stable_baselines3; print('✓ Dependencies OK')" || {
    echo "Error: Missing dependencies. Installing..."
    pip install torch stable-baselines3 --quiet
}

# Create output directory
echo "3. Creating output directory..."
mkdir -p Data/ablation_studies

# Ask user for confirmation
echo ""
echo "=========================================="
echo "Experiment Configuration:"
echo "=========================================="
echo "Variants: hca2c_full, hca2c_flat, hca2c_wide, a2c_enhanced"
echo "Seeds: 42, 43, 44"
echo "Load: 3.0x"
echo "Timesteps: 500,000 per run"
echo "Total runs: 4 variants × 3 seeds = 12 runs"
echo "Estimated time: ~30 hours"
echo "Output: Data/ablation_studies/"
echo "=========================================="
echo ""

read -p "Start ablation studies? (y/n) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Start experiments in background
echo ""
echo "4. Starting ablation studies in background..."
nohup python Code/training_scripts/run_ablation_studies.py \
    --variants hca2c_full hca2c_flat hca2c_wide a2c_enhanced \
    --seeds 42 43 44 \
    --load 3.0 \
    --timesteps 500000 \
    --output-dir Data/ablation_studies \
    > ablation_studies.log 2>&1 &

PID=$!
echo $PID > ablation_studies.pid

echo ""
echo "=========================================="
echo "✓ Ablation studies started!"
echo "=========================================="
echo "Process ID: $PID"
echo "Log file: ablation_studies.log"
echo "PID file: ablation_studies.pid"
echo ""
echo "Monitor progress:"
echo "  tail -f ablation_studies.log"
echo ""
echo "Check status:"
echo "  python -c \"import pandas as pd; df=pd.read_csv('Data/ablation_studies/ablation_results.csv'); print(f'Completed: {len(df)}/12')\""
echo ""
echo "Stop experiments:"
echo "  kill \$(cat ablation_studies.pid)"
echo ""
echo "=========================================="
