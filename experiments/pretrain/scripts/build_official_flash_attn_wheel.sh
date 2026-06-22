#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

OUT_DIR="${OUT_DIR:-experiments/pretrain/wheelhouse/flash_attn_official}"
DIAG_DIR="${DIAG_DIR:-experiments/pretrain/diagnostics/flash_attn}"
FLASH_ATTN_REF="${FLASH_ATTN_REF:-v2.8.3}"
mkdir -p "$OUT_DIR" "$DIAG_DIR"

PROJECT_CUDA_HOME="${PROJECT_CUDA_HOME:-$ROOT/.tools/cuda-nvcc-12.9.86}"
if [[ -x "$PROJECT_CUDA_HOME/bin/nvcc" ]]; then
  export CUDA_HOME="$PROJECT_CUDA_HOME"
  export CUDA_PATH="$PROJECT_CUDA_HOME"
  export PATH="$CUDA_HOME/bin:$PATH"
  export LD_LIBRARY_PATH="$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
fi

.venv/bin/python - <<'PY'
import json, urllib.request
from pathlib import Path

out = Path("experiments/pretrain/diagnostics/flash_attn/official_release_probe.json")
pattern = "torch2.11"
url = "https://api.github.com/repos/Dao-AILab/flash-attention/releases?per_page=100"
assets = []
error = None
try:
    page = 1
    while page <= 3:
        req = urllib.request.Request(f"{url}&page={page}", headers={"User-Agent": "minimind"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            releases = json.load(resp)
        if not releases:
            break
        for release in releases:
            for asset in release.get("assets", []):
                name = asset["name"]
                if name.startswith("flash_attn"):
                    assets.append({"tag": release["tag_name"], "name": name, "url": asset["browser_download_url"]})
        page += 1
except Exception as exc:
    error = repr(exc)
matches = [a for a in assets if pattern in a["name"] and "cp312" in a["name"] and "linux_x86_64" in a["name"]]
out.write_text(json.dumps({"query": "torch2.11 cp312 linux_x86_64", "matches": matches, "asset_count": len(assets), "error": error}, indent=2) + "\n")
print(out)
PY

if [[ "${QUERY_ONLY:-0}" == "1" ]]; then
  exit 0
fi

if ! command -v nvcc >/dev/null 2>&1; then
  cat > "$DIAG_DIR/official_build_blocked.json" <<JSON
{"status":"blocked","reason":"nvcc not found on PATH","ref":"$FLASH_ATTN_REF"}
JSON
  echo "blocked: nvcc not found" >&2
  exit 2
fi

nvcc --version | tee "$DIAG_DIR/nvcc_version.txt"

export FLASH_ATTN_CUDA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-89}"
export MAX_JOBS="${MAX_JOBS:-4}"
export NVCC_THREADS="${NVCC_THREADS:-1}"

if [[ -n "${FLASH_ATTN_SOURCE_DIR:-}" ]]; then
  .venv/bin/python -m pip wheel --no-build-isolation --no-deps \
    "$FLASH_ATTN_SOURCE_DIR" \
    -w "$OUT_DIR"
else
  .venv/bin/python -m pip wheel --no-build-isolation --no-deps \
    "git+https://github.com/Dao-AILab/flash-attention.git@${FLASH_ATTN_REF}" \
    -w "$OUT_DIR"
fi
