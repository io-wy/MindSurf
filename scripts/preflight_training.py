"""Fail-fast checks before occupying the single shared GPU."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from python_starter.core.data_contract import (
    load_dataset_spec,
    sha256_file,
    verify_training_view_manifest,
)


def _nvidia_smi(*query: str) -> list[str]:
    completed = subprocess.run(
        ["nvidia-smi", *query],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("artifacts/data/minimind_official_v1/audit.json"),
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("configs/datasets/minimind_official_v1.json"),
    )
    parser.add_argument(
        "--training-view-manifest",
        type=Path,
        default=Path("artifacts/data/minimind_official_v1/training_view.json"),
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path("data/processed/minimind_official_v1/pretrain_train_nfkc_dedup.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("models/checkpoints"))
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    required_gates = (
        "schema_valid",
        "metadata_contract_valid",
        "file_identity_valid",
        "validation_test_disjoint",
        "train_holdout_disjoint",
    )
    if audit.get("status") != "passed" or not all(
        audit.get("gates", {}).get(name) is True for name in required_gates
    ):
        raise SystemExit("dataset audit is not valid for internal training")
    spec = load_dataset_spec(args.spec)
    if audit.get("dataset_revision") != spec.get("revision"):
        raise SystemExit("dataset audit revision does not match the source spec")
    training_view = verify_training_view_manifest(
        args.training_view_manifest,
        train_path=args.train_path,
        dataset_id=str(audit["dataset_id"]),
        revision=str(audit["dataset_revision"]),
        source_sha256=str(audit["splits"]["train"]["sha256"]),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(args.output_dir).free
    if free_bytes < 15 * 1024**3:
        raise SystemExit("less than 15 GiB free at the checkpoint destination")

    gpu: dict[str, object] = {"required": args.require_cuda}
    if args.require_cuda:
        devices = _nvidia_smi(
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        )
        if len(devices) != 1:
            raise SystemExit(f"expected exactly one visible GPU, found {len(devices)}")
        processes = _nvidia_smi(
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        )
        if processes:
            raise SystemExit(f"GPU already has compute processes: {processes}")
        gpu = {"required": True, "device": devices[0], "compute_processes": []}

    result = {
        "status": "passed",
        "dataset_id": audit["dataset_id"],
        "dataset_revision": audit["dataset_revision"],
        "dataset_spec_sha256": sha256_file(args.spec),
        "training_view_sha256": training_view["output"]["sha256"],
        "training_view_rows": training_view["output"]["rows"],
        "free_bytes": free_bytes,
        "gpu": gpu,
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
