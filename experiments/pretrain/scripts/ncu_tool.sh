#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export NV_COMPUTE_PROFILER_DISABLE_STOCK_FILE_DEPLOYMENT="${NV_COMPUTE_PROFILER_DISABLE_STOCK_FILE_DEPLOYMENT:-1}"
exec "$ROOT/.tools/nsight/ncu-2026.2.0/pkg/target/linux-desktop-glibc_2_11_3-x64/ncu" "$@"
