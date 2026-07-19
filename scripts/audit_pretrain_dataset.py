"""Audit the pinned team pretraining corpus and emit a machine-readable contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from python_starter.core.data_contract import (
    FileIdentity,
    audit_split,
    iter_jsonl,
    load_dataset_spec,
    published_text_digest,
    sha256_file,
    verify_file,
    write_json_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="configs/datasets/mindsurf_team_v1.json")
    parser.add_argument("--root", default="data/raw/mindsurf_team_v1")
    parser.add_argument(
        "--output",
        default="artifacts/data/mindsurf_team_v1/audit.json",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Generated split metadata; defaults to the pinned metadata file when present",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Smoke-only row cap; disables full-file identity assertions",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec_path = Path(args.spec)
    root = Path(args.root)
    spec = load_dataset_spec(spec_path)
    identities = {name: FileIdentity.from_mapping(value) for name, value in spec["files"].items()}
    if args.metadata:
        metadata_path = Path(args.metadata)
    elif "metadata" in identities:
        metadata_path = root / identities["metadata"].path
    else:
        raise ValueError("--metadata is required when the dataset spec has no metadata file")
    if "metadata" in identities:
        verify_file(metadata_path, identities["metadata"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("dataset_id") not in (None, spec["dataset_id"]):
        raise ValueError("Split metadata dataset identity is inconsistent")
    require_source_id = "source_key" in spec.get("schema", {}) and "id" in spec.get("schema", {})

    validation, validation_hashes = audit_split(
        root / identities["validation"].path,
        expected=identities["validation"],
        require_source_id=require_source_id,
    )
    test, test_hashes = audit_split(
        root / identities["test"].path,
        expected=identities["test"],
        reserved_hashes=validation_hashes,
        require_source_id=require_source_id,
    )
    if test["reserved_overlap_rows"]:
        raise ValueError("Validation/test normalized-text overlap detected")
    expected_validation_hashes = {
        bytes.fromhex(value) for value in metadata.get("val_text_hashes", [])
    }
    expected_test_hashes = {bytes.fromhex(value) for value in metadata.get("test_text_hashes", [])}
    observed_validation_hashes = {
        published_text_digest(str(row["text"]))
        for _, row in iter_jsonl(root / identities["validation"].path)
    }
    observed_test_hashes = {
        published_text_digest(str(row["text"]))
        for _, row in iter_jsonl(root / identities["test"].path)
    }
    if observed_validation_hashes != expected_validation_hashes:
        raise ValueError("Validation text hashes do not match the published split metadata")
    if observed_test_hashes != expected_test_hashes:
        raise ValueError("Test text hashes do not match the published split metadata")
    if (
        metadata.get("val_sha256") != identities["validation"].sha256
        or metadata.get("test_sha256") != identities["test"].sha256
        or metadata.get("train_lines") != identities["train"].rows
        or metadata.get("seed") != int(spec.get("split_seed", 20260511))
    ):
        raise ValueError("Published strict-split metadata identity is inconsistent")
    holdout_hashes = validation_hashes | test_hashes
    train, _ = audit_split(
        root / identities["train"].path,
        expected=identities["train"],
        reserved_hashes=holdout_hashes,
        max_rows=args.max_train_rows,
        require_source_id=require_source_id,
    )
    if train["reserved_overlap_rows"]:
        raise ValueError("Training/holdout normalized-text overlap detected")

    tokenizer_files = {}
    for name, raw_identity in spec["tokenizer"]["files"].items():
        identity = FileIdentity.from_mapping(raw_identity)
        path = root / "tokenizer" / name
        tokenizer_files[name] = {
            "path": str(path.as_posix()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "expected_sha256": identity.sha256,
        }
        if tokenizer_files[name]["sha256"] != identity.sha256:
            raise ValueError(f"Tokenizer identity mismatch for {path}")

    result = {
        "schema_version": 1,
        "status": "passed",
        "dataset_id": spec["dataset_id"],
        "dataset_revision": spec["revision"],
        "dataset_spec_sha256": sha256_file(spec_path),
        "license": spec["license"],
        "license_status": spec["license_status"],
        "declared_token_count": spec.get("declared_token_count"),
        "split_metadata": {
            "path": str(metadata_path.as_posix()),
            "sha256": sha256_file(metadata_path),
            "selection": metadata.get("selection"),
            "seed": metadata.get("seed"),
            "source_lines": metadata.get("source_lines"),
            "unique_texts": metadata.get("unique_texts"),
            "duplicate_texts": metadata.get("duplicate_texts"),
        },
        "splits": {
            "train": train,
            "validation": validation,
            "test": test,
        },
        "tokenizer": {
            "revision": spec["tokenizer"]["revision"],
            "vocab_size": spec["tokenizer"]["vocab_size"],
            "files": tokenizer_files,
        },
        "gates": {
            "schema_valid": True,
            "metadata_contract_valid": True,
            "file_identity_valid": args.max_train_rows is None,
            "validation_test_disjoint": True,
            "train_holdout_disjoint": True,
            "public_release_license_ready": spec["license_status"] == "ready",
        },
    }
    write_json_atomic(args.output, result)
    print(json.dumps(result["gates"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
