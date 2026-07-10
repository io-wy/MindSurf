#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

WEIGHT_PATH="${WEIGHT_PATH:-experiments/pretrain/platform_runs/pretrain_stage12_ext80_replay_from_stage11_s512_lr3e7/01_continue/pretrain_stage12_ext80_replay_from_stage11_s512_lr3e7_continue_768.pth}"
BASE_PORT="${BASE_PORT:-18998}"

run_case() {
  local run_name="$1"
  local port="$2"
  shift 2
  local output_dir="experiments/pretrain/diagnostics/api_latency/${run_name}"
  local log_path="${output_dir}/server.log"
  mkdir -p "$output_dir"

  .venv/bin/python scripts/serve_openai_api.py \
    --host 127.0.0.1 \
    --port "$port" \
    --load_from model \
    --weight_path "$WEIGHT_PATH" \
    --hidden_size 768 \
    --num_hidden_layers 8 \
    --num_attention_heads 8 \
    --num_key_value_heads 8 \
    --intermediate_size 3072 \
    --max_seq_len 2048 \
    --device cuda \
    "$@" \
    >"$log_path" 2>&1 &
  local server_pid=$!

  cleanup_case() {
    if kill -0 "$server_pid" 2>/dev/null; then
      kill "$server_pid" 2>/dev/null || true
      wait "$server_pid" 2>/dev/null || true
    fi
  }

  for _ in $(seq 1 90); do
    if grep -q "Uvicorn running" "$log_path"; then
      break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      cat "$log_path"
      return 1
    fi
    sleep 1
  done

  if ! grep -q "Uvicorn running" "$log_path"; then
    cat "$log_path"
    cleanup_case
    echo "server did not become ready" >&2
    return 1
  fi

  .venv/bin/python experiments/pretrain/scripts/benchmark_openai_api_latency.py \
    --endpoint "http://127.0.0.1:${port}/v1/chat/completions" \
    --output_dir "$output_dir" \
    --prompt_chars 128 512 1024 \
    --concurrency 1 2 4 8 \
    --requests_per_case 3 \
    --max_tokens 64 \
    --no-stream

  cleanup_case
  cp "$0" "${output_dir}/launch.sh"
}

run_case "stage12_ext80_replay_api_latency_nostream_no_batch" "$BASE_PORT"
run_case "stage12_ext80_replay_api_latency_dynamic_batch_b4_w8ms" "$((BASE_PORT + 1))" \
  --dynamic_batching \
  --batch_max_size 4 \
  --batch_wait_ms 8
