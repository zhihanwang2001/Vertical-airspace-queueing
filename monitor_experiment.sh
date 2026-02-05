#!/bin/bash
while true; do
    clear
    echo "=== HCA2C Experiment Monitor ==="
    echo "Time: $(date +%H:%M:%S)"
    echo ""
    
    # Check process
    if ps aux | grep -q "69608.*python"; then
        echo "✓ Process running (PID: 69608)"
        ps aux | grep "69608.*python" | grep -v grep | awk '{print "  CPU: " $3 "% | Memory: " $4 "%"}'
    else
        echo "✗ Process not found"
        break
    fi
    
    echo ""
    
    # Check output files
    if [ -f "Data/hca2c_final_comparison_local/HCA2C_seed45_load7.0.json" ]; then
        echo "✓ Results file created!"
        ls -lh Data/hca2c_final_comparison_local/
        break
    else
        echo "⏳ Waiting for results..."
    fi
    
    echo ""
    echo "Log file size: $(ls -lh hca2c_seed45_load7.log 2>/dev/null | awk '{print $5}' || echo '0B')"
    echo ""
    echo "Press Ctrl+C to stop monitoring"
    
    sleep 30
done
