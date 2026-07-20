"""Watch a live training run and report alerts as machine-readable evidence.

Runs beside the training process, reads only the tracker log and host state,
and exits non-zero when any alert fires so a supervisor can act on it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from python_starter.core.data_contract import write_json_atomic  # noqa: E402
from python_starter.infrastructure.training_monitor import (  # noqa: E402
    AlertThresholds,
    MonitorState,
    evaluate_alerts,
    parse_metrics,
)


def _run_finished(path: Path) -> bool:
    """True once the tracker has written its finish row."""
    if not path.is_file():
        return False
    for line in reversed(path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        return bool(json.loads(line).get("type") == "finish")
    return False


def _gpu_total_bytes(index: int) -> int | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--id={index}",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return int(float(completed.stdout.strip())) * 1024 * 1024


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument(
        "--max-polls",
        type=int,
        default=0,
        help="Stop after this many polls; 0 watches until the run finishes",
    )
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--stall-seconds", type=float, default=600.0)
    parser.add_argument("--loss-spike-ratio", type=float, default=1.5)
    parser.add_argument("--loss-spike-window", type=int, default=40)
    parser.add_argument("--free-bytes-min", type=int, default=20_000_000_000)
    parser.add_argument("--gpu-reserved-fraction-max", type=float, default=0.95)
    args = parser.parse_args()

    thresholds = AlertThresholds(
        loss_spike_ratio=args.loss_spike_ratio,
        loss_spike_window=args.loss_spike_window,
        stall_seconds=args.stall_seconds,
        gpu_reserved_fraction_max=args.gpu_reserved_fraction_max,
        free_bytes_min=args.free_bytes_min,
    )
    gpu_total_bytes = _gpu_total_bytes(args.gpu_index)
    started_at = datetime.now(UTC).isoformat()
    fired: list[dict[str, object]] = []
    polls = 0
    last_step: int | None = None
    last_step_at = time.monotonic()
    highest_step = 0

    while True:
        polls += 1
        samples = parse_metrics(args.metrics) if args.metrics.is_file() else []
        finished = _run_finished(args.metrics)
        now = time.monotonic()
        if samples and samples[-1].step != last_step:
            last_step = samples[-1].step
            last_step_at = now
        highest_step = max(highest_step, last_step or 0)

        state = MonitorState(
            samples=samples,
            seconds_since_last_step=0.0 if finished else now - last_step_at,
            free_bytes=shutil.disk_usage(args.output.parent if args.output.parent.exists() else ROOT).free,
            gpu_total_bytes=gpu_total_bytes,
        )
        alerts = evaluate_alerts(state, thresholds)
        for alert in alerts:
            record = {**alert, "observed_at": datetime.now(UTC).isoformat()}
            fired.append(record)
            print(json.dumps(record), flush=True)

        write_json_atomic(
            args.output,
            {
                "schema_version": 1,
                "started_at": started_at,
                "updated_at": datetime.now(UTC).isoformat(),
                "metrics_path": str(args.metrics),
                "thresholds": thresholds.as_dict(),
                "polls": polls,
                "observed_steps": highest_step,
                "observed_samples": len(samples),
                "gpu_total_bytes": gpu_total_bytes,
                "free_bytes": state.free_bytes,
                "alerts": fired,
                "clean": not fired,
                "run_finished": finished,
            },
        )

        if finished or (args.max_polls and polls >= args.max_polls):
            break
        time.sleep(args.poll_seconds)

    raise SystemExit(1 if fired else 0)


if __name__ == "__main__":
    main()
