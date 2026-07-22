"""Derive the training summary of an interrupted run from its checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from python_starter.core.data_contract import write_json_atomic  # noqa: E402
from python_starter.core.trainer import build_training_summary  # noqa: E402


def parameter_count(model_state: dict[str, Any]) -> int:
    """Count parameters once each, even where the checkpoint ties two names."""
    seen: dict[int, int] = {}
    for tensor in model_state.values():
        seen[tensor.untyped_storage().data_ptr()] = tensor.numel()
    return sum(seen.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Defaults to training_summary.json beside the checkpoint",
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    progress = checkpoint.get("progress")
    if not isinstance(progress, dict):
        raise SystemExit(f"checkpoint carries no progress block: {args.checkpoint}")

    summary = build_training_summary(
        progress=progress,
        run_config=checkpoint.get("run_config", {}),
        parameter_count=parameter_count(checkpoint["model_state_dict"]),
    )
    # The summary a live run writes is emitted at the last step; this one is
    # read back off a checkpoint, and a reader has to be able to tell which.
    summary["derived_from_checkpoint"] = str(args.checkpoint)

    output = args.output or args.checkpoint.parent / "training_summary.json"
    write_json_atomic(output, summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
