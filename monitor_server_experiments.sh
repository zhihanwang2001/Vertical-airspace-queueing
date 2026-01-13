#!/bin/bash
# Server Experiment Monitoring Script
# Usage: ./monitor_server_experiments.sh

PASSWORD='uNBRd68Bzc5hhDZF2ZpCdZKF6pMXeK83'
SERVER='root@i-1.gpushare.com'
PORT='60899'

echo "=================================="
echo "Server Experiment Monitor"
echo "Time: $(date)"
echo "=================================="
echo ""

echo "📋 Active Screen Sessions:"
sshpass -p "$PASSWORD" ssh -p $PORT -o StrictHostKeyChecking=no $SERVER "screen -ls"
echo ""

echo "📁 Log File Sizes:"
sshpass -p "$PASSWORD" ssh -p $PORT -o StrictHostKeyChecking=no $SERVER "cd /root/RP1 && ls -lh logs/*.log | tail -10"
echo ""

echo "=================================="
echo "🔬 K=30 Extended Training"
echo "=================================="
sshpass -p "$PASSWORD" ssh -p $PORT -o StrictHostKeyChecking=no $SERVER "cd /root/RP1 && tail -20 logs/k30_final.log"
echo ""

echo "=================================="
echo "📊 Supplementary n=3 Experiments"
echo "=================================="
sshpass -p "$PASSWORD" ssh -p $PORT -o StrictHostKeyChecking=no $SERVER "cd /root/RP1 && tail -20 logs/supplementary_n3_experiments.log"
echo ""

echo "=================================="
echo "🎯 Experiment A (5× Load)"
echo "=================================="
sshpass -p "$PASSWORD" ssh -p $PORT -o StrictHostKeyChecking=no $SERVER "cd /root/RP1 && tail -20 logs/experiment_a_5x_load.log"
echo ""

echo "=================================="
echo "✅ Monitoring Complete"
echo "=================================="
