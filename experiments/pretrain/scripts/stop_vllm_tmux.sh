#!/usr/bin/env bash
set -euo pipefail

SESSION="${SESSION:-minimind_vllm_service}"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux kill-session -t "$SESSION"
  echo "stopped: $SESSION"
else
  echo "not running: $SESSION"
fi
