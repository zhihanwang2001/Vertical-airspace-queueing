#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"

# Parameters (with defaults)
SEEDS="${SEEDS:-30}"
TIMESTEPS="${TIMESTEPS:-100000}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
LOAD_MULTIPLIER="${LOAD_MULTIPLIER:-5.0}"

CMD=(python "$ROOT_DIR/Code/training_scripts/run_structural_comparison_5x_load.py" \
  --mode all \
  --n-seeds "$SEEDS" \
  --timesteps "$TIMESTEPS" \
  --eval-episodes "$EVAL_EPISODES" \
  --load-multiplier "$LOAD_MULTIPLIER")

TS=$(date +%Y%m%d_%H%M%S)
LOGFILE="$LOG_DIR/structural_5x_${SEEDS}s_${TIMESTEPS}t_${EVAL_EPISODES}e_${TS}.log"

echo "Launching: ${CMD[*]}" | tee -a "$LOGFILE"
nohup "${CMD[@]}" >> "$LOGFILE" 2>&1 &
PID=$!
echo $PID > "$LOGFILE.pid"
echo "PID: $PID (log: $LOGFILE)"

