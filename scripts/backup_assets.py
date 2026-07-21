"""Mirror durable run assets to a content-addressed backup and verify them back.

DVC is configured for the data DAG, but the training views and checkpoints on
this host are symlinks into the canonical ``repo/`` checkout. Pointing DVC at
them would rewrite those paths into cache links and put the only authoritative
copy at risk, so durability is handled by copying instead: read the source,
hash it, store it under its digest, and re-hash the stored copy to prove the
restore path works.

The backup root is a plain directory, so it can be a second filesystem today
and an object-store mount later without changing the inventory format.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from python_starter.core.data_contract import sha256_file, write_json_atomic  # noqa: E402

# Logical name -> repository-relative source. Upstream raw corpora are absent on
# purpose: they are re-fetchable by dataset id and revision, which the dataset
# index already pins.
DEFAULT_ASSETS: dict[str, str] = {
    "official_training_view": "data/processed/minimind_official_v1/pretrain_train_nfkc_dedup.jsonl",
    "official_strict_val": "data/raw/minimind_official_v1/strict_splits/pretrain_strict_val_2k.jsonl",
    "official_strict_test": "data/raw/minimind_official_v1/strict_splits/pretrain_strict_test_2k.jsonl",
    "official_training_view_manifest": "artifacts/data/minimind_official_v1/training_view.json",
    "official_audit": "artifacts/data/minimind_official_v1/audit.json",
    "pretrain_mcq_benchmark_v2": "configs/evaluation/pretrain_mcq_benchmark_v2.jsonl",
    "pretrain_gate": "configs/evaluation/pretrain_gate_80m.json",
    "parent_checkpoint": "models/checkpoints/minimind_official_v1_80m/final_model.pt",
    "parent_training_summary": "models/checkpoints/minimind_official_v1_80m/training_summary.json",
}


def _resolve(relative: str) -> Path:
    return (ROOT / relative).resolve()


def _store_path(backup_root: Path, digest: str) -> Path:
    return backup_root / "sha256" / digest[:2] / digest


def _restore_command(stored: Path, relative: str) -> str:
    return f"cp {stored} {relative}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--asset",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Extra logical asset to mirror; repeatable",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Re-hash the stored copies without writing anything new",
    )
    args = parser.parse_args()

    assets = dict(DEFAULT_ASSETS)
    for item in args.asset:
        name, _, relative = item.partition("=")
        if not name or not relative:
            raise SystemExit(f"--asset expects NAME=PATH, got {item!r}")
        assets[name] = relative

    backup_root = args.backup_root.resolve()
    entries: list[dict[str, Any]] = []
    missing: list[str] = []
    failed: list[str] = []

    for name, relative in sorted(assets.items()):
        source = _resolve(relative)
        if not source.is_file():
            missing.append(name)
            continue

        digest = sha256_file(source)
        stored = _store_path(backup_root, digest)
        if not args.verify_only and not stored.is_file():
            stored.parent.mkdir(parents=True, exist_ok=True)
            staging = stored.with_suffix(".partial")
            shutil.copyfile(source, staging)
            staging.replace(stored)

        verified = stored.is_file() and sha256_file(stored) == digest
        if not verified:
            failed.append(name)

        entries.append(
            {
                "logical_name": name,
                "source": relative,
                "resolved_source": str(source),
                "size_bytes": source.stat().st_size,
                "sha256": digest,
                "backup_path": str(stored),
                "restore_command": _restore_command(stored, relative),
                "verified": verified,
            }
        )

    total_bytes = sum(int(entry["size_bytes"]) for entry in entries)
    write_json_atomic(
        args.output,
        {
            "schema_version": 1,
            "recorded_at": datetime.now(UTC).isoformat(),
            "backup_root": str(backup_root),
            "durability_scope": (
                "second filesystem on the training host; survives workspace loss and "
                "accidental deletion, not host loss"
            ),
            "verify_only": args.verify_only,
            "asset_count": len(entries),
            "total_bytes": total_bytes,
            "missing_assets": missing,
            "failed_assets": failed,
            "assets": entries,
        },
    )

    print(
        f"{len(entries)} assets, {total_bytes / 1e9:.2f} GB, "
        f"missing {len(missing)}, failed {len(failed)}"
    )
    if failed or missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
