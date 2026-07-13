#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VLLM_ENV="${VLLM_ENV:-$ROOT/.venv-vllm}"
PYTHON="${PYTHON:-python3}"
VLLM_VERSION="${VLLM_VERSION:-0.25.0}"

if [[ ! -x "$VLLM_ENV/bin/python" ]]; then
  "$PYTHON" -m venv "$VLLM_ENV"
fi

"$VLLM_ENV/bin/python" -m pip install --upgrade pip
"$VLLM_ENV/bin/pip" install "vllm==$VLLM_VERSION"
"$VLLM_ENV/bin/pip" check

# vLLM's PyNvVideoCodec wheel already carries compatible shared FFmpeg 8
# libraries. Expose that project-local directory so TorchCodec can load without
# requiring a host-wide FFmpeg installation.
FFMPEG_LIB_DIR="$("$VLLM_ENV/bin/python" - <<'PY'
from pathlib import Path
import sysconfig

print(Path(sysconfig.get_paths()["purelib"]) / "PyNvVideoCodec")
PY
)"
export LD_LIBRARY_PATH="$FFMPEG_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$VLLM_ENV/bin/python" -c "import torchcodec; print(torchcodec.__version__)"
"$VLLM_ENV/bin/vllm" --version
"$VLLM_ENV/bin/pip" freeze > "$VLLM_ENV/installed.freeze.txt"
