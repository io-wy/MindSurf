"""Run a command only after an atomic shared-GPU memory reservation succeeds."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from python_starter.experiments.run_registry import RunRegistry  # noqa: E402
from python_starter.infrastructure.gpu_capacity import GpuLeaseStore  # noqa: E402


def _monitor_gpu(
    *,
    stop: threading.Event,
    output: Path,
    gpu_index: int,
    child_pid: int,
    checkpoint_root: Path,
    interval_seconds: float,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    while not stop.is_set():
        try:
            row = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={gpu_index}",
                    "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu,"
                    "temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            values = [value.strip() for value in row.split(",")]
            checkpoint_free_bytes = shutil.disk_usage(checkpoint_root).free
            event: dict[str, object] = {
                "timestamp": datetime.now(UTC).isoformat(),
                "child_pid": child_pid,
                "memory_total_mib": int(values[0]),
                "memory_used_mib": int(values[1]),
                "memory_free_mib": int(values[2]),
                "gpu_utilization_percent": int(values[3]),
                "temperature_celsius": int(values[4]),
                "power_watts": float(values[5]),
                "checkpoint_free_bytes": checkpoint_free_bytes,
            }
            warnings = []
            if int(values[2]) < 512:
                warnings.append("gpu_memory_below_512_mib")
            if checkpoint_free_bytes < 10 * 1024**3:
                warnings.append("checkpoint_disk_below_10_gib")
            event["warnings"] = warnings
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            event = {
                "timestamp": datetime.now(UTC).isoformat(),
                "child_pid": child_pid,
                "monitor_error": str(exc),
            }
        with output.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        stop.wait(interval_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--required-memory-mib", type=int, required=True)
    parser.add_argument("--safety-margin-mib", type=int, default=1536)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--wait-seconds", type=int, default=3600)
    parser.add_argument(
        "--state",
        type=Path,
        default=Path.home() / ".cache/mindsurf/gpu_leases.json",
    )
    parser.add_argument(
        "--run-registry",
        type=Path,
        default=Path.home() / ".cache/mindsurf/run_registry.json",
    )
    parser.add_argument("--retry", action="store_true")
    parser.add_argument(
        "--telemetry-dir",
        type=Path,
        default=Path.home() / ".cache/mindsurf/telemetry",
    )
    parser.add_argument("--monitor-interval-seconds", type=float, default=5.0)
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT / "models/checkpoints")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise SystemExit("a command is required after --")

    store = GpuLeaseStore(args.state)
    registry = RunRegistry(args.run_registry)
    registry.transition(
        args.run_id,
        "queued",
        detail={"command": command, "required_memory_mib": args.required_memory_mib},
        allow_retry=args.retry,
    )
    deadline = time.monotonic() + args.wait_seconds
    lease_id = None
    decision: dict[str, object] = {}
    while lease_id is None:
        lease_id, decision = store.try_acquire(
            required_mib=args.required_memory_mib,
            safety_margin_mib=args.safety_margin_mib,
            run_id=args.run_id,
            gpu_index=args.gpu_index,
        )
        if lease_id is not None:
            break
        if time.monotonic() >= deadline:
            registry.transition(args.run_id, "rejected", detail=decision)
            print(json.dumps(decision, ensure_ascii=False, sort_keys=True), file=sys.stderr)
            raise SystemExit("GPU capacity wait timed out")
        time.sleep(5)

    print(json.dumps({"lease_id": lease_id, **decision}, ensure_ascii=False, sort_keys=True))
    try:
        process = subprocess.Popen(command, cwd=ROOT)
        store.set_child_pid(lease_id, process.pid)
        registry.transition(
            args.run_id,
            "resumed" if args.retry else "running",
            detail={"child_pid": process.pid, "lease_id": lease_id},
        )
        monitor_stop = threading.Event()
        monitor = threading.Thread(
            target=_monitor_gpu,
            kwargs={
                "stop": monitor_stop,
                "output": args.telemetry_dir / f"{args.run_id}.jsonl",
                "gpu_index": args.gpu_index,
                "child_pid": process.pid,
                "checkpoint_root": args.checkpoint_root,
                "interval_seconds": args.monitor_interval_seconds,
            },
            daemon=True,
        )
        monitor.start()
        return_code = process.wait()
        monitor_stop.set()
        monitor.join(timeout=max(1.0, args.monitor_interval_seconds + 1))
        registry.transition(
            args.run_id,
            "completed" if return_code == 0 else "failed",
            detail={"return_code": return_code},
        )
        raise SystemExit(return_code)
    finally:
        store.release(lease_id)


if __name__ == "__main__":
    main()
