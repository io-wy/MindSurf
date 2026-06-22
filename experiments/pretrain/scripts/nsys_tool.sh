#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec "$ROOT/.tools/nsight/nsys-2026.3.1/opt/nvidia/nsight-systems-cli/2026.3.1/target-linux-x64/nsys" "$@"
