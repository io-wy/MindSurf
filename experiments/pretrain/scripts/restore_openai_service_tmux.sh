#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

SNAPSHOT_DIR="${SNAPSHOT_DIR:-experiments/pretrain/serving_snapshots/stage12_ext80_replay_dynamic_batch}"
.venv/bin/python experiments/pretrain/scripts/verify_serving_snapshot.py "$SNAPSHOT_DIR" >/dev/null
exec experiments/pretrain/scripts/start_openai_service_tmux.sh
