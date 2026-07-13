#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

SESSION="${SESSION:-minimind_openai_service}"
PORT="${PORT:-8998}"
HOST="${HOST:-127.0.0.1}"
LOG_DIR="${LOG_DIR:-experiments/pretrain/diagnostics/service_tmux}"
WEIGHT_PATH="${WEIGHT_PATH:-experiments/pretrain/platform_runs/infra_baseline_gqa64m_lr5e4_40m_20260712/infra_baseline_gqa64m_lr5e4_40m_768.pth}"
HIDDEN_SIZE="${HIDDEN_SIZE:-768}"
NUM_HIDDEN_LAYERS="${NUM_HIDDEN_LAYERS:-8}"
NUM_ATTENTION_HEADS="${NUM_ATTENTION_HEADS:-8}"
NUM_KEY_VALUE_HEADS="${NUM_KEY_VALUE_HEADS:-4}"
INTERMEDIATE_SIZE="${INTERMEDIATE_SIZE:-2432}"
mkdir -p "$LOG_DIR"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "already running: $SESSION"
  exit 0
fi

.venv/bin/python experiments/pretrain/scripts/infra_preflight.py \
  --mode serving \
  --root "$ROOT" \
  --min-free-disk-gb "${MIN_FREE_DISK_GB:-10}" \
  --max-gpu-memory-used-mb "${MAX_GPU_MEMORY_USED_MB:-512}" \
  --required-file "$WEIGHT_PATH" \
  --output "$LOG_DIR/preflight.json"

CMD=".venv/bin/python scripts/serve_openai_api.py --weight_path $WEIGHT_PATH --hidden_size $HIDDEN_SIZE --num_hidden_layers $NUM_HIDDEN_LAYERS --num_attention_heads $NUM_ATTENTION_HEADS --num_key_value_heads $NUM_KEY_VALUE_HEADS --intermediate_size $INTERMEDIATE_SIZE --max_seq_len 2048 --host $HOST --port $PORT --dynamic_batching --batch_max_size 4 --batch_wait_ms 8.0"
tmux new-session -d -s "$SESSION" "cd '$ROOT' && exec $CMD >'$LOG_DIR/service.log' 2>&1"
for _ in $(seq 1 120); do
  if curl -fsS "http://$HOST:$PORT/healthz" >/dev/null 2>&1; then
    echo "started: $SESSION http://$HOST:$PORT"
    exit 0
  fi
  if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    tail -n 120 "$LOG_DIR/service.log" >&2 || true
    exit 1
  fi
  sleep 1
done
tmux kill-session -t "$SESSION" 2>/dev/null || true
echo "native service health check timed out" >&2
exit 1
