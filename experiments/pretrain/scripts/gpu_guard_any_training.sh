#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
mkdir -p experiments/pretrain/logs
log="experiments/pretrain/logs/gpu_guard_any_training.log"
echo "GPU guard start $(date -Iseconds), threshold=87C" >> "$log"
while pgrep -af '[p]ython train_' >/dev/null; do
  line=$(nvidia-smi --query-gpu=temperature.gpu,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true)
  echo "$(date -Iseconds),$line" >> "$log"
  temp=$(printf '%s' "$line" | awk -F',' '{gsub(/ /,"",$1); print $1}')
  if [ -n "${temp:-}" ] && [ "$temp" -ge 87 ] 2>/dev/null; then
    echo "$(date -Iseconds),TEMP_LIMIT_REACHED,$temp,stopping python train_*.py" >> "$log"
    pkill -TERM -f '[p]ython train_'
    exit 10
  fi
  sleep 30
done
echo "GPU guard stop $(date -Iseconds): train process not found" >> "$log"
