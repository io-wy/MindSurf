#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
mkdir -p "$ROOT/experiments/pretrain/logs"
cd "$ROOT/trainer"
PYTHONUNBUFFERED=1 "$PY" train_pretrain.py --from_resume 1 --num_workers 0 --log_interval 50 --save_interval 1000 > "$ROOT/experiments/pretrain/logs/pretrain_resume_current.log" 2>&1
