#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT"
.venv/bin/python experiments/pretrain/scripts/benchmark_openai_api_latency.py --endpoint http://127.0.0.1:8998/v1/chat/completions --model minimind --output_dir experiments/pretrain/serving_snapshots/stage12_ext80_replay_dynamic_batch/api_latency --prompt_chars 128 512 1024 --concurrency 1 2 4 8 --requests_per_case 4 --max_tokens 64 --no-stream
