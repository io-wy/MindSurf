#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

PORT="${PORT:-8999}"
LOG_DIR="${LOG_DIR:-experiments/pretrain/diagnostics/service_smoke}"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/openai_service.log"
SUMMARY="$LOG_DIR/summary.json"
WEIGHT_PATH="${WEIGHT_PATH:-experiments/pretrain/platform_runs/infra_baseline_gqa64m_lr5e4_40m_20260712/infra_baseline_gqa64m_lr5e4_40m_768.pth}"

.venv/bin/python scripts/serve_openai_api.py \
  --weight_path "$WEIGHT_PATH" \
  --hidden_size 768 --num_hidden_layers 8 --num_attention_heads 8 --num_key_value_heads 4 \
  --intermediate_size 2432 --max_seq_len 2048 --host 127.0.0.1 --port "$PORT" \
  --dynamic_batching --batch_max_size 4 --batch_wait_ms 8.0 >"$LOG" 2>&1 &
PID=$!
trap 'kill "$PID" >/dev/null 2>&1 || true' EXIT

.venv/bin/python - "$PORT" "$SUMMARY" <<'PY'
import json, sys, time, urllib.error, urllib.request
port, summary = sys.argv[1], sys.argv[2]
base = f"http://127.0.0.1:{port}"
deadline = time.time() + 120
while True:
    try:
        with urllib.request.urlopen(base + "/healthz", timeout=3) as resp:
            health = json.load(resp)
        break
    except Exception as exc:
        if time.time() > deadline:
            raise RuntimeError(f"healthz timeout: {exc}") from exc
        time.sleep(1)
payload = {
    "model": "minimind",
    "messages": [{"role": "user", "content": "用一句话说明你是谁。"}],
    "stream": False,
    "temperature": 0,
    "max_tokens": 16,
}
req = urllib.request.Request(
    base + "/v1/chat/completions",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
start = time.time()
with urllib.request.urlopen(req, timeout=60) as resp:
    body = json.load(resp)
elapsed = time.time() - start
result = {"health": health, "elapsed_sec": elapsed, "response": body}
open(summary, "w", encoding="utf-8").write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
PY
