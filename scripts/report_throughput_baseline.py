"""Record the effective-throughput and MFU baseline for a finished training run."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from python_starter.core.data_contract import write_json_atomic  # noqa: E402
from python_starter.core.utils import (  # noqa: E402
    DEVICE_PEAK_BF16_FLOPS,
    model_flops_per_token,
    model_flops_utilization,
)


def _device_name(index: int) -> str | None:
    try:
        completed = subprocess.run(
            ["nvidia-smi", f"--id={index}", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-layer", type=int, default=8)
    parser.add_argument("--n-embed", type=int, default=768)
    parser.add_argument("--sequence-length", type=int, default=384)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--device-name")
    parser.add_argument(
        "--device-peak-flops",
        type=float,
        help="Override the dense bf16 peak when the device is not in the table",
    )
    args = parser.parse_args()

    summary = json.loads(args.training_summary.read_text(encoding="utf-8"))
    device_name = args.device_name or _device_name(args.gpu_index)
    peak = args.device_peak_flops or DEVICE_PEAK_BF16_FLOPS.get(str(device_name), 0.0)
    if not peak:
        raise SystemExit(
            f"no dense bf16 peak known for device {device_name!r}; pass --device-peak-flops"
        )

    parameter_count = int(summary["parameter_count"])
    tokens_per_second = float(summary["tokens_per_second"])
    flops_per_token = model_flops_per_token(
        parameter_count=parameter_count,
        n_layer=args.n_layer,
        n_embed=args.n_embed,
        sequence_length=args.sequence_length,
    )
    mfu = model_flops_utilization(
        flops_per_token=flops_per_token,
        tokens_per_second=tokens_per_second,
        device_peak_flops=peak,
    )

    write_json_atomic(
        args.output,
        {
            "schema_version": 1,
            "recorded_at": datetime.now(UTC).isoformat(),
            "training_summary": str(args.training_summary),
            "device_name": device_name,
            "device_peak_bf16_flops": peak,
            "parameter_count": parameter_count,
            "geometry": {
                "n_layer": args.n_layer,
                "n_embed": args.n_embed,
                "sequence_length": args.sequence_length,
            },
            "consumed_tokens": int(summary["consumed_tokens"]),
            "global_step": int(summary["global_step"]),
            "training_elapsed_seconds": float(summary["training_elapsed_seconds"]),
            "tokens_per_second": tokens_per_second,
            "model_flops_per_token": flops_per_token,
            "achieved_flops": flops_per_token * tokens_per_second,
            "mfu": mfu,
            "tokens_per_parameter": int(summary["consumed_tokens"]) / parameter_count,
        },
    )
    print(
        f"tokens/s {tokens_per_second:,.0f}  achieved {flops_per_token * tokens_per_second / 1e12:.1f} TFLOPS  MFU {mfu:.1%}"
    )


if __name__ == "__main__":
    main()
