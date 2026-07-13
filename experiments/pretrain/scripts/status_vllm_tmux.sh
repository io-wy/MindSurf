#!/usr/bin/env bash
set -euo pipefail

SESSION="${SESSION:-minimind_vllm_service}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-19099}"

if ! tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "not running: $SESSION" >&2
  exit 1
fi

curl -fsS "http://$HOST:$PORT/health" >/dev/null
curl -fsS "http://$HOST:$PORT/v1/models"
echo
curl -fsS "http://$HOST:$PORT/metrics" | grep -E 'vllm:(num_requests_running|num_requests_waiting|kv_cache_usage_perc)' || true
