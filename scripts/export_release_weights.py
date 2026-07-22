"""Export a registered candidate's weights for public release.

A training checkpoint carries the optimizer, scheduler, and RNG state that only
a resume needs; a release needs the weights, the config to build the model, and
the lineage that says where they came from. Lineage is read from the candidate
registry rather than retyped, so released weights cannot claim a provenance the
registry does not hold, and it travels inside the file so it cannot be
separated from the terms it was released under.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from python_starter.core.data_contract import sha256_file, write_json_atomic  # noqa: E402


def registry_record(registry_path: Path, name: str) -> dict[str, Any]:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    for record in registry.get("models", []):
        if record.get("name") == name:
            return dict(record)
    raise SystemExit(f"no candidate named {name!r} in {registry_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="Registered candidate name")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--registry", type=Path, default=Path("artifacts/model_registry.json"))
    parser.add_argument(
        "--license",
        type=Path,
        default=Path("configs/release/pretrain_80m_license.json"),
        dest="license_path",
    )
    # A registry written on the training host records that host's paths, so an
    # export elsewhere has to be told where the assets are now. The sha256 is
    # what identifies them, and it is still checked against the registry.
    parser.add_argument("--checkpoint", type=Path, help="Where the registered checkpoint is now")
    parser.add_argument(
        "--training-summary", type=Path, help="Where the registered training summary is now"
    )
    args = parser.parse_args()

    record = registry_record(args.registry, args.name)

    def resolve(kind: str, override: Path | None) -> Path | None:
        registered = record.get(f"{kind}_path")
        if registered is None:
            return None
        path = override or Path(registered)
        actual = sha256_file(path)
        if actual != record[f"{kind}_sha256"]:
            raise SystemExit(
                f"{kind} does not match the registered candidate: "
                f"{actual} != {record[f'{kind}_sha256']}"
            )
        return path

    checkpoint_path = resolve("checkpoint", args.checkpoint)
    assert checkpoint_path is not None  # a record without a checkpoint cannot exist
    summary_path = resolve("training_summary", args.training_summary)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    summary = (
        json.loads(summary_path.read_text(encoding="utf-8")) if summary_path is not None else {}
    )

    release = {
        "name": args.name,
        "exported_at": datetime.now(UTC).isoformat(),
        "source_checkpoint_sha256": record["checkpoint_sha256"],
        "source_git_head": record.get("source_git_head"),
        "global_step": summary.get("global_step"),
        "consumed_tokens": summary.get("consumed_tokens"),
        "parameter_count": summary.get("parameter_count"),
        "identity": summary.get("identity"),
        "gate": {
            "config": record.get("gate_config"),
            "reference": record.get("gate_reference"),
            "verdict_sha256": record.get("verdict_sha256"),
            "evaluation_sha256": record.get("evaluation_sha256"),
        },
        "limitations": record.get("limitations", []),
        "terms": json.loads(args.license_path.read_text(encoding="utf-8")),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_config": checkpoint["model_config"],
            "model_state_dict": checkpoint["model_state_dict"],
            "release": release,
        },
        args.output,
    )
    manifest = {**release, "weights_file": args.output.name, "weights_sha256": sha256_file(args.output)}
    write_json_atomic(args.output.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
