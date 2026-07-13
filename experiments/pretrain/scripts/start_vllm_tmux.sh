#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

SESSION="${SESSION:-minimind_vllm_service}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-19099}"
LOG_DIR="${LOG_DIR:-experiments/pretrain/diagnostics/service_vllm}"
LOG_PATH="$LOG_DIR/service.log"
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
  --required-file out/releases/current.json \
  --output "$LOG_DIR/preflight.json"

tmux new-session -d -s "$SESSION" \
  "cd '$ROOT' && exec bash experiments/pretrain/scripts/serve_vllm.sh >'$LOG_PATH' 2>&1"

for _ in $(seq 1 120); do
  if curl -fsS "http://$HOST:$PORT/health" >/dev/null 2>&1; then
    echo "started: $SESSION http://$HOST:$PORT"
    exit 0
  fi
  if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    tail -n 120 "$LOG_PATH" >&2 || true
    echo "vLLM session exited before becoming healthy" >&2
    exit 1
  fi
  sleep 1
done

tail -n 120 "$LOG_PATH" >&2 || true
tmux kill-session -t "$SESSION" 2>/dev/null || true
echo "vLLM health check timed out" >&2
exit 1
