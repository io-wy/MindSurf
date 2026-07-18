"""Build an NFKC-deduplicated training view from the pinned shuffled split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from python_starter.core.data_contract import (  # noqa: E402
    FileIdentity,
    iter_jsonl,
    load_dataset_spec,
    text_digest,
    verify_file,
    write_json_atomic,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=Path("configs/datasets/mindsurf_team_v1.json"))
    parser.add_argument("--root", type=Path, default=Path("data/raw/mindsurf_team_v1"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/mindsurf_team_v1/pretrain_train_nfkc_dedup.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/data/mindsurf_team_v1/training_view.json"),
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("artifacts/data/mindsurf_team_v1/audit.json"),
    )
    args = parser.parse_args()

    spec = load_dataset_spec(args.spec)
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if audit.get("status") != "passed":
        raise ValueError("source dataset audit must pass before building the training view")
    if (
        audit.get("dataset_id") != spec["dataset_id"]
        or audit.get("dataset_revision") != spec["revision"]
    ):
        raise ValueError("source audit identity does not match the dataset specification")
    identities = {name: FileIdentity.from_mapping(value) for name, value in spec["files"].items()}
    source = args.root / identities["train"].path
    verify_file(source, identities["train"])

    holdout_hashes = {
        text_digest(str(row["text"]))
        for split in ("validation", "test")
        for _, row in iter_jsonl(args.root / identities[split].path)
    }
    seen: set[bytes] = set()
    source_rows = 0
    output_rows = 0
    duplicate_rows_removed = 0
    holdout_rows_removed = 0
    output_sha256 = hashlib.sha256()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{uuid.uuid4().hex}.tmp")
    try:
        with source.open("rb") as input_handle, temporary.open("wb") as output_handle:
            for line_number, raw_line in enumerate(input_handle, start=1):
                if not raw_line.strip():
                    continue
                source_rows += 1
                try:
                    row = json.loads(raw_line)
                    digest = text_digest(str(row["text"]))
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    raise ValueError(f"invalid training row at {source}:{line_number}") from exc
                if digest in holdout_hashes:
                    holdout_rows_removed += 1
                    continue
                if digest in seen:
                    duplicate_rows_removed += 1
                    continue
                seen.add(digest)
                canonical_line = raw_line.rstrip(b"\r\n") + b"\n"
                output_handle.write(canonical_line)
                output_sha256.update(canonical_line)
                output_rows += 1
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)

    if source_rows != identities["train"].rows:
        raise ValueError(
            f"source row mismatch: expected {identities['train'].rows}, got {source_rows}"
        )
    expected_duplicates = int(audit["splits"]["train"]["duplicate_texts"])
    if duplicate_rows_removed != expected_duplicates:
        raise ValueError(
            "derived-view duplicate count does not match the full source audit: "
            f"expected {expected_duplicates}, got {duplicate_rows_removed}"
        )
    if holdout_rows_removed != 0:
        raise ValueError(
            f"source audit/view disagreement: removed {holdout_rows_removed} holdout rows"
        )
    manifest = {
        "schema_version": 1,
        "dataset_id": spec["dataset_id"],
        "dataset_revision": spec["revision"],
        "source": {
            "path": str(source.as_posix()),
            "sha256": identities["train"].sha256,
            "rows": source_rows,
        },
        "output": {
            "path": str(args.output.as_posix()),
            "sha256": output_sha256.hexdigest(),
            "size": args.output.stat().st_size,
            "rows": output_rows,
        },
        "normalization": "NFKC plus whitespace collapse",
        "duplicate_rows_removed": duplicate_rows_removed,
        "holdout_rows_removed": holdout_rows_removed,
        "gates": {
            "source_identity_valid": True,
            "nfkc_duplicate_free": True,
            "holdout_disjoint": True,
        },
    }
    write_json_atomic(args.manifest, manifest)
    print(
        json.dumps(
            {
                "output_rows": output_rows,
                "duplicate_rows_removed": duplicate_rows_removed,
                "holdout_rows_removed": holdout_rows_removed,
                "sha256": output_sha256.hexdigest(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
