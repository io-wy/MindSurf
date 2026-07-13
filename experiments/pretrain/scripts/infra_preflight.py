from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


def evaluate_snapshot(
    snapshot: dict[str, Any],
    *,
    mode: str,
    min_free_disk_gb: float,
    max_gpu_memory_used_mb: int,
    max_gpu_temperature_c: int,
) -> dict[str, Any]:
    failures = []
    if snapshot["disk_free_gb"] < min_free_disk_gb:
        failures.append(
            f"disk free expected >= {min_free_disk_gb:.1f} GiB, got {snapshot['disk_free_gb']:.1f} GiB"
        )
    if snapshot["gpu_memory_used_mb"] > max_gpu_memory_used_mb:
        failures.append(
            f"GPU memory used expected <= {max_gpu_memory_used_mb} MiB, got {snapshot['gpu_memory_used_mb']} MiB"
        )
    if snapshot["gpu_temperature_c"] > max_gpu_temperature_c:
        failures.append(
            f"GPU temperature expected <= {max_gpu_temperature_c} C, got {snapshot['gpu_temperature_c']} C"
        )
    if mode == "training" and snapshot["gpu_compute_processes"] > 0:
        failures.append(
            f"training requires an idle GPU, found {snapshot['gpu_compute_processes']} compute processes"
        )
    return {"passed": not failures, "mode": mode, "failures": failures, "snapshot": snapshot}


def _run_nvidia_smi(args: list[str]) -> str:
    result = subprocess.run(
        ["nvidia-smi", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def collect_snapshot(root: Path) -> dict[str, Any]:
    disk = shutil.disk_usage(root)
    gpu_line = _run_nvidia_smi(
        ["--query-gpu=memory.used,temperature.gpu", "--format=csv,noheader,nounits"]
    ).splitlines()[0]
    memory_used, temperature = [int(part.strip()) for part in gpu_line.split(",")]
    process_lines = _run_nvidia_smi(
        ["--query-compute-apps=pid", "--format=csv,noheader,nounits"]
    ).splitlines()
    return {
        "disk_free_gb": disk.free / 1024**3,
        "gpu_memory_used_mb": memory_used,
        "gpu_temperature_c": temperature,
        "gpu_compute_processes": len([line for line in process_lines if line.strip()]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only MiniMind GPU and disk preflight")
    parser.add_argument("--mode", choices=["training", "serving"], required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--min-free-disk-gb", type=float, default=20.0)
    parser.add_argument("--max-gpu-memory-used-mb", type=int, default=512)
    parser.add_argument("--max-gpu-temperature-c", type=int, default=85)
    parser.add_argument("--required-file", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    snapshot = collect_snapshot(args.root)
    report = evaluate_snapshot(
        snapshot,
        mode=args.mode,
        min_free_disk_gb=args.min_free_disk_gb,
        max_gpu_memory_used_mb=args.max_gpu_memory_used_mb,
        max_gpu_temperature_c=args.max_gpu_temperature_c,
    )
    for raw_path in args.required_file:
        path = raw_path if raw_path.is_absolute() else args.root / raw_path
        if not path.is_file():
            report["failures"].append(f"required file missing: {raw_path}")
    report["passed"] = not report["failures"]
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
