#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"

# Params
SHAPE="${SHAPE:-uniform}"
CAPACITIES="${CAPACITIES:-10,30}"
LOADS="${LOADS:-5}"
ALGOS="${ALGOS:-A2C,PPO}"
SEEDS="${SEEDS:-5}"
TIMESTEPS="${TIMESTEPS:-100000}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"

CMD=(python "$ROOT_DIR/Code/training_scripts/run_capacity_scan.py" \
  --include-heuristics \
  --capacities "$CAPACITIES" \
  --loads "$LOADS" \
  --shape "$SHAPE" \
  --algos "$ALGOS" \
  --n-seeds "$SEEDS" \
  --timesteps "$TIMESTEPS" \
  --eval-episodes "$EVAL_EPISODES")

TS=$(date +%Y%m%d_%H%M%S)
LOGFILE="$LOG_DIR/capacity_paradox_rl_${SHAPE}_${SEEDS}s_${TIMESTEPS}t_${EVAL_EPISODES}e_${TS}.log"
echo "Launching: ${CMD[*]}" | tee -a "$LOGFILE"
nohup "${CMD[@]}" >> "$LOGFILE" 2>&1 &
PID=$!
echo $PID > "$LOGFILE.pid"
echo "PID: $PID (log: $LOGFILE)"

