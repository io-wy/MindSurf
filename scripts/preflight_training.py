"""Fail-fast identity, disk, and capacity checks before shared-GPU training."""

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
    write_json_atomic,
)
from python_starter.infrastructure.gpu_capacity import capacity_decision, query_gpu_snapshot


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
    parser.add_argument("--required-gpu-memory-mib", type=int, default=11_000)
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=0,
        help="Device to admit against; a second arm on a two-card host must not "
        "be judged by the card the first arm already fills",
    )
    parser.add_argument("--gpu-safety-margin-mib", type=int, default=1536)
    parser.add_argument("--require-branch")
    parser.add_argument(
        "--report",
        type=Path,
        help="Where to persist the preflight record; defaults to preflight.json in --output-dir",
    )
    parser.add_argument(
        "--min-free-bytes",
        type=int,
        default=15 * 1024**3,
        help="Space budget at the checkpoint destination; callers that know the "
        "retained checkpoint count should pass their own figure",
    )
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
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
    if free_bytes < args.min_free_bytes:
        raise SystemExit(
            f"{free_bytes / 1024**3:.1f} GiB free at the checkpoint destination, "
            f"budget {args.min_free_bytes / 1024**3:.1f} GiB"
        )

    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=args.source_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    git_branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=args.source_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if args.require_branch is not None and git_branch != args.require_branch:
        raise SystemExit(
            f"source branch mismatch: expected {args.require_branch!r}, got {git_branch!r}"
        )

    gpu: dict[str, object] = {"required": args.require_cuda}
    if args.require_cuda:
        snapshot = query_gpu_snapshot(args.gpu_index)
        decision = capacity_decision(
            snapshot,
            required_mib=args.required_gpu_memory_mib,
            safety_margin_mib=args.gpu_safety_margin_mib,
            reservations=[],
        )
        if not decision["admitted"]:
            raise SystemExit(f"insufficient shared GPU capacity: {decision}")
        gpu = {"required": True, "capacity": decision}

    result = {
        "status": "passed",
        "dataset_id": audit["dataset_id"],
        "dataset_revision": audit["dataset_revision"],
        "dataset_spec_sha256": sha256_file(args.spec),
        "training_view_sha256": training_view["output"]["sha256"],
        "training_view_rows": training_view["output"]["rows"],
        "free_bytes": free_bytes,
        "min_free_bytes": args.min_free_bytes,
        "source": {"git_head": git_head, "git_branch": git_branch},
        "gpu": gpu,
    }
    # Persist beside the checkpoint: the source commit is the only record of
    # which code produced the run, and a log line does not survive as evidence.
    report_path = args.report or args.output_dir / "preflight.json"
    write_json_atomic(report_path, result)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
