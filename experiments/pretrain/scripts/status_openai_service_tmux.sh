#!/usr/bin/env bash
set -euo pipefail

SESSION="${SESSION:-minimind_openai_service}"
PORT="${PORT:-8998}"
if ! tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "not running: $SESSION"
  exit 1
fi
curl -fsS "http://127.0.0.1:$PORT/healthz"
echo
