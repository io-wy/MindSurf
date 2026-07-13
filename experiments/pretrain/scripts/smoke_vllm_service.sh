#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-19099}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/pretrain/diagnostics/service_vllm/smoke}"

curl -fsS "http://$HOST:$PORT/health" >/dev/null
.venv/bin/python experiments/pretrain/scripts/benchmark_openai_api_latency.py \
  --endpoint "http://$HOST:$PORT/v1/chat/completions" \
  --model minimind \
  --output_dir "$OUTPUT_DIR" \
  --prompt_chars 128 \
  --concurrency 1 4 \
  --requests_per_case 1 \
  --max_tokens 16 \
  --temperature 0 \
  --top_p 1 \
  --no-stream
curl -fsS "http://$HOST:$PORT/metrics" > "$OUTPUT_DIR/metrics.prom"
