#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
OUT="${OUT:-experiments/pretrain/diagnostics/nsight/nanovllm_decode_greedy_nsys}"
mkdir -p "$(dirname "$OUT")"
NANOVLLM_DIST_PORT="${NANOVLLM_DIST_PORT:-2933}" \
PYTHONPATH=experiments/pretrain/vendor/nano-vllm \
experiments/pretrain/scripts/nsys_tool.sh profile \
  --force-overwrite=true \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --output="$OUT" \
  .venv/bin/python experiments/pretrain/scripts/benchmark_nanovllm.py \
    --model_dir experiments/pretrain/exports/stage12_ext80_replay_qwen \
    --output_json experiments/pretrain/diagnostics/nanovllm/nsys_smoke_driver.json \
    --prompt_chars 128 --requests_per_case 4 --max_tokens 32 --temperature 0 --ignore_eos \
    --warmup --max_model_len 2048 --max_num_seqs 16 --max_num_batched_tokens 4096 \
    --gpu_memory_utilization 0.5
