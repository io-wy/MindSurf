#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT"
exec .venv/bin/python scripts/serve_openai_api.py --weight_path experiments/pretrain/platform_runs/pretrain_stage12_ext80_replay_from_stage11_s512_lr3e7/01_continue/pretrain_stage12_ext80_replay_from_stage11_s512_lr3e7_continue_768.pth --hidden_size 768 --num_hidden_layers 8 --num_attention_heads 8 --num_key_value_heads 8 --intermediate_size 3072 --max_seq_len 2048 --host 0.0.0.0 --port 8998 --dynamic_batching --batch_max_size 4 --batch_wait_ms 8.0
