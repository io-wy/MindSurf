#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

PORT="${PORT:-18998}"
RUN_NAME="${RUN_NAME:-stage12_ext80_replay_api_latency}"
OUTPUT_DIR="experiments/pretrain/diagnostics/api_latency/${RUN_NAME}"
LOG_PATH="${OUTPUT_DIR}/server.log"
WEIGHT_PATH="${WEIGHT_PATH:-experiments/pretrain/platform_runs/pretrain_stage12_ext80_replay_from_stage11_s512_lr3e7/01_continue/pretrain_stage12_ext80_replay_from_stage11_s512_lr3e7_continue_768.pth}"

mkdir -p "$OUTPUT_DIR"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

.venv/bin/python scripts/serve_openai_api.py \
  --host 127.0.0.1 \
  --port "$PORT" \
  --load_from model \
  --weight_path "$WEIGHT_PATH" \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --num_attention_heads 8 \
  --num_key_value_heads 8 \
  --intermediate_size 3072 \
  --max_seq_len 2048 \
  --device cuda \
  >"$LOG_PATH" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 90); do
  if grep -q "Uvicorn running" "$LOG_PATH"; then
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    cat "$LOG_PATH"
    exit 1
  fi
  sleep 1
done

if ! grep -q "Uvicorn running" "$LOG_PATH"; then
  cat "$LOG_PATH"
  echo "server did not become ready" >&2
  exit 1
fi

.venv/bin/python experiments/pretrain/scripts/benchmark_openai_api_latency.py \
  --endpoint "http://127.0.0.1:${PORT}/v1/chat/completions" \
  --output_dir "$OUTPUT_DIR" \
  --prompt_chars 128 512 1024 \
  --concurrency 1 2 4 \
  --requests_per_case 3 \
  --max_tokens 64 \
  --stream

cp "$0" "${OUTPUT_DIR}/launch.sh"
