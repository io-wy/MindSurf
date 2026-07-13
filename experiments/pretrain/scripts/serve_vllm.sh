#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VLLM_ENV="${VLLM_ENV:-$ROOT/.venv-vllm}"
CURRENT_RELEASE="${CURRENT_RELEASE:-$ROOT/out/releases/current.json}"
if [[ -z "${MODEL_PATH:-}" ]]; then
  if [[ -f "$CURRENT_RELEASE" ]]; then
    MODEL_PATH="$("$VLLM_ENV/bin/python" - "$ROOT" "$CURRENT_RELEASE" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
payload = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
artifact = (root / payload["artifact_dir"]).resolve()
if not artifact.is_relative_to(root):
    raise SystemExit("release artifact escapes project root")
print(artifact)
PY
)"
  else
    MODEL_PATH="$ROOT/out/infra_baseline_gqa64m_lr5e4_40m_hf_fp16_vllm"
  fi
fi
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-19099}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-512}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.10}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

if [[ ! -x "$VLLM_ENV/bin/vllm" ]]; then
  echo "vLLM executable not found: $VLLM_ENV/bin/vllm" >&2
  exit 1
fi
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "Hugging Face artifact not found: $MODEL_PATH" >&2
  exit 1
fi

FFMPEG_LIB_DIR="$("$VLLM_ENV/bin/python" - <<'PY'
from pathlib import Path
import sysconfig

print(Path(sysconfig.get_paths()["purelib"]) / "PyNvVideoCodec")
PY
)"
export LD_LIBRARY_PATH="$FFMPEG_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# For this 64M model, eager mode avoids disproportionate compile/JIT startup,
# while a 10% memory budget still leaves capacity for 182k KV-cache tokens.
export VLLM_USE_FLASHINFER_SAMPLER
exec "$VLLM_ENV/bin/vllm" serve "$MODEL_PATH" \
  --model-impl transformers \
  --trust-remote-code \
  --served-model-name minimind \
  --host "$HOST" \
  --port "$PORT" \
  --dtype half \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --enforce-eager \
  "$@"
