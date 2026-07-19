"""Reproduce the historical deterministic MiniMind official strict splits."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from python_starter.core.data_contract import (  # noqa: E402
    FileIdentity,
    load_dataset_spec,
    published_text_digest,
    verify_file,
    write_json_atomic,
)


def _score(seed: int, digest: bytes) -> int:
    payload = f"{seed}:{digest.hex()}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def _select_holdouts(
    source: Path, seed: int, limit: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    heap: list[tuple[int, int, dict[str, Any]]] = []
    seen: set[bytes] = set()
    stats = {
        "source_lines": 0,
        "nonempty_json_lines": 0,
        "unique_texts": 0,
        "duplicate_texts": 0,
        "skipped_bad_json": 0,
        "skipped_empty_text": 0,
    }
    with source.open("r", encoding="utf-8", newline="") as handle:
        for line_index, line in enumerate(handle):
            stats["source_lines"] += 1
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                stats["skipped_bad_json"] += 1
                continue
            text = str(row.get("text", ""))
            if not text.strip():
                stats["skipped_empty_text"] += 1
                continue
            stats["nonempty_json_lines"] += 1
            digest = published_text_digest(text)
            if digest in seen:
                stats["duplicate_texts"] += 1
                continue
            seen.add(digest)
            stats["unique_texts"] += 1
            score = _score(seed, digest)
            item: dict[str, Any] = {
                "line_index": line_index,
                "hash": digest.hex(),
                "score": score,
                "line": line,
            }
            entry = (-score, line_index, item)
            if len(heap) < limit:
                heapq.heappush(heap, entry)
            elif score < -heap[0][0]:
                heapq.heapreplace(heap, entry)
    selected = [entry[2] for entry in heap]
    selected.sort(key=lambda item: item["score"])
    return selected, stats


def _write_split(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in sorted(rows, key=lambda item: item["line_index"]):
            handle.write(str(row["line"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec", type=Path, default=Path("configs/datasets/minimind_official_v1.json")
    )
    parser.add_argument("--root", type=Path, default=Path("data/raw/minimind_official_v1"))
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("artifacts/data/minimind_official_v1/split_metadata.json"),
    )
    parser.add_argument("--split-size", type=int, default=2000)
    args = parser.parse_args()

    spec = load_dataset_spec(args.spec)
    identities = {name: FileIdentity.from_mapping(value) for name, value in spec["files"].items()}
    source = args.root / identities["source"].path
    verify_file(source, identities["source"])
    seed = int(spec["split_seed"])
    selected, stats = _select_holdouts(source, seed, args.split_size * 2)
    if len(selected) != args.split_size * 2:
        raise ValueError("official source does not contain enough unique rows")

    validation_rows = selected[: args.split_size]
    test_rows = selected[args.split_size :]
    reserved = {bytes.fromhex(str(item["hash"])) for item in selected}
    output_dir = args.root / "strict_splits"
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_dir = output_dir / f".build-{uuid.uuid4().hex}"
    temporary_dir.mkdir()
    try:
        validation = temporary_dir / "pretrain_strict_val_2k.jsonl"
        test = temporary_dir / "pretrain_strict_test_2k.jsonl"
        train = temporary_dir / "pretrain_strict_train.jsonl"
        _write_split(validation, validation_rows)
        _write_split(test, test_rows)

        seen: set[bytes] = set()
        train_rows = 0
        train_duplicates_skipped = 0
        train_reserved_skipped = 0
        with (
            source.open("r", encoding="utf-8", newline="") as source_handle,
            train.open("w", encoding="utf-8", newline="\n") as train_handle,
        ):
            for line in source_handle:
                if not line.strip():
                    continue
                try:
                    text = str(json.loads(line).get("text", ""))
                except json.JSONDecodeError:
                    continue
                if not text.strip():
                    continue
                digest = published_text_digest(text)
                if digest in reserved:
                    train_reserved_skipped += 1
                    continue
                if digest in seen:
                    train_duplicates_skipped += 1
                    continue
                seen.add(digest)
                train_handle.write(line)
                train_rows += 1

        for name, temporary in (
            ("train", train),
            ("validation", validation),
            ("test", test),
        ):
            verify_file(temporary, identities[name])
            os.replace(temporary, args.root / identities[name].path)
    finally:
        for child in temporary_dir.glob("*"):
            child.unlink(missing_ok=True)
        temporary_dir.rmdir()

    write_json_atomic(
        args.metadata,
        {
            "schema_version": 1,
            "dataset_id": spec["dataset_id"],
            "dataset_revision": spec["revision"],
            **stats,
            "seed": seed,
            "split_size": args.split_size,
            "selection": "smallest sha256(seed:text_sha1) scores after whitespace-normalized deduplication",
            "train_lines": train_rows,
            "train_duplicates_skipped": train_duplicates_skipped,
            "train_reserved_skipped": train_reserved_skipped,
            "val_text_hashes": [item["hash"] for item in validation_rows],
            "test_text_hashes": [item["hash"] for item in test_rows],
            "train_sha256": identities["train"].sha256,
            "val_sha256": identities["validation"].sha256,
            "test_sha256": identities["test"].sha256,
        },
    )
    print(
        json.dumps(
            {
                "train_rows": train_rows,
                "validation_rows": len(validation_rows),
                "test_rows": len(test_rows),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
