"""Carve a holdout-disjoint train file out of the full official corpus.

The pipeline contract is that `train` is already disjoint from the strict
holdout by the time the audit runs: `audit_pretrain_dataset.py` raises on any
overlap, and the training-view builder then only deduplicates. The frozen
80M work satisfied that by splitting the mini corpus into train/validation/test
in one step.

The full corpus cannot be split that way without abandoning the frozen holdout,
which every existing threshold and comparison depends on. So it is filtered
instead: keep the 4,000 frozen holdout rows exactly as they are, and remove any
row of the full corpus whose published text digest matches one of them. The
count of removed rows is itself the answer to whether the mini file was a subset
of the full one.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from python_starter.core.data_contract import (  # noqa: E402
    iter_jsonl,
    published_text_digest,
    write_json_atomic,
)


def _holdout_digests(paths: list[Path]) -> tuple[set[bytes], dict[str, list[str]]]:
    digests: set[bytes] = set()
    published: dict[str, list[str]] = {}
    for path in paths:
        rows = [published_text_digest(str(row["text"])) for _, row in iter_jsonl(path)]
        digests.update(rows)
        published[path.stem] = [value.hex() for value in rows]
    return digests, published


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--dataset-id", default="gongjy/minimind_dataset")
    parser.add_argument("--revision", default="74aad49fa4443e7ed640d44bc4e9c7d1fe71ada5")
    parser.add_argument("--split-seed", type=int, default=20260511)
    args = parser.parse_args()

    holdout, published = _holdout_digests([args.validation, args.test])

    source_rows = 0
    train_rows = 0
    holdout_rows_removed = 0
    empty_rows = 0
    train_sha = hashlib.sha256()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{uuid.uuid4().hex}.tmp")
    try:
        with args.corpus.open("rb") as source, temporary.open("wb") as sink:
            for line_number, raw in enumerate(source, start=1):
                if not raw.strip():
                    empty_rows += 1
                    continue
                source_rows += 1
                try:
                    text = str(json.loads(raw)["text"])
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    raise ValueError(f"invalid row at {args.corpus}:{line_number}") from exc
                if published_text_digest(text) in holdout:
                    holdout_rows_removed += 1
                    continue
                canonical = raw.rstrip(b"\r\n") + b"\n"
                sink.write(canonical)
                train_sha.update(canonical)
                train_rows += 1
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)

    write_json_atomic(
        args.metadata,
        {
            "schema_version": 1,
            "prepared_at": datetime.now(UTC).isoformat(),
            "dataset_id": args.dataset_id,
            "dataset_revision": args.revision,
            "seed": args.split_seed,
            "split_size": 2000,
            "selection": (
                "frozen strict holdout inherited unchanged; the full corpus is filtered "
                "by published text digest so the train file is holdout-disjoint"
            ),
            "source": args.corpus.name,
            "source_lines": source_rows,
            "empty_rows_skipped": empty_rows,
            "holdout_rows_removed": holdout_rows_removed,
            "mini_is_subset_of_full": holdout_rows_removed > 0,
            "train_lines": train_rows,
            "train_sha256": train_sha.hexdigest(),
            "val_sha256": hashlib.sha256(args.validation.read_bytes()).hexdigest(),
            "test_sha256": hashlib.sha256(args.test.read_bytes()).hexdigest(),
            "val_text_hashes": published[args.validation.stem],
            "test_text_hashes": published[args.test.stem],
        },
    )
    print(
        json.dumps(
            {
                "source_lines": source_rows,
                "holdout_rows_removed": holdout_rows_removed,
                "train_lines": train_rows,
            }
        )
    )


if __name__ == "__main__":
    main()
