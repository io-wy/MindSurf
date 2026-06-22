#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

WHEEL="${1:-experiments/pretrain/wheelhouse/flash_attn/flash_attn-2.8.3+cu12torch2.11cxx11abiTRUE-cp312-cp312-linux_x86_64.whl}"
EXPECTED_SHA256="3d0c8e60f820321eedd7166e79c33cb816263d8be6e35c3f5ba8fe2df6fea697"
OUT_DIR="experiments/pretrain/diagnostics/flash_attn"

mkdir -p "$OUT_DIR"

if [[ ! -f "$WHEEL" ]]; then
  echo "missing wheel: $WHEEL" >&2
  exit 1
fi

echo "${EXPECTED_SHA256}  ${WHEEL}" | sha256sum -c -

.venv/bin/python experiments/pretrain/scripts/probe_flash_attn_env.py \
  --output_json "$OUT_DIR/server_env_before_flash_attn_install.json"

if [[ "${CHECK_ONLY:-0}" == "1" ]]; then
  echo "check-only ok"
  exit 0
fi

.venv/bin/python -m pip install --no-deps --no-index "$WHEEL"

.venv/bin/python experiments/pretrain/scripts/flash_attn_smoke.py \
  --output_json "$OUT_DIR/flash_attn_smoke.json"

.venv/bin/python experiments/pretrain/scripts/probe_flash_attn_env.py \
  --output_json "$OUT_DIR/server_env_after_flash_attn_install.json"
