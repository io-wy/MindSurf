#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

SESSION="${SESSION:-minimind_openai_service}"
PORT="${PORT:-8998}"
HOST="${HOST:-127.0.0.1}"
LOG_DIR="${LOG_DIR:-experiments/pretrain/diagnostics/service_tmux}"
mkdir -p "$LOG_DIR"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "already running: $SESSION"
  exit 0
fi

CMD=".venv/bin/python scripts/serve_openai_api.py --weight_path experiments/pretrain/platform_runs/pretrain_stage12_ext80_replay_from_stage11_s512_lr3e7/01_continue/pretrain_stage12_ext80_replay_from_stage11_s512_lr3e7_continue_768.pth --hidden_size 768 --num_hidden_layers 8 --num_attention_heads 8 --num_key_value_heads 8 --intermediate_size 3072 --max_seq_len 2048 --host $HOST --port $PORT --dynamic_batching --batch_max_size 4 --batch_wait_ms 8.0"
tmux new-session -d -s "$SESSION" "cd '$ROOT' && exec $CMD >'$LOG_DIR/service.log' 2>&1"
echo "started: $SESSION http://$HOST:$PORT"
