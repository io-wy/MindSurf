"""Apply near-duplicate, quality and PII filtering to a training view.

One pass, three stages, in the order the reference standard gives: quality and
safety filtering first because they are cheap and per-row, then near-duplicate
detection over what survives. Every dropped row is accounted for by reason, so
the output manifest states what was removed rather than only how much.

The result is a new view with its own identity. The input view is left intact
so the two remain comparable, which is the whole point of running this as an
arm rather than a rewrite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from python_starter.core.data_contract import write_json_atomic  # noqa: E402
from python_starter.core.near_duplicates import (  # noqa: E402
    MinHashConfig,
    duplicate_indices,
    permutation_parameters,
    shingle_hashes,
    signature,
)
from python_starter.core.quality_filters import (  # noqa: E402
    QualityThresholds,
    evaluate_row,
    language_profile,
)

_WORKER: dict[str, Any] = {}


def _init_worker(config: MinHashConfig, thresholds: QualityThresholds, drop_on_pii: bool) -> None:
    a, b = permutation_parameters(config)
    _WORKER.update(
        {"config": config, "thresholds": thresholds, "a": a, "b": b, "drop_on_pii": drop_on_pii}
    )


def _assess_chunk(payload: tuple[int, list[str]]) -> tuple[int, list[dict[str, Any]]]:
    index, lines = payload
    config: MinHashConfig = _WORKER["config"]
    thresholds: QualityThresholds = _WORKER["thresholds"]
    results: list[dict[str, Any]] = []
    for line in lines:
        text = str(json.loads(line)["text"])
        keep, detail = evaluate_row(text, thresholds, drop_on_pii=_WORKER["drop_on_pii"])
        results.append(
            {
                "keep": keep,
                "reasons": detail["reasons"],
                "pii": detail["pii"],
                "language": language_profile(text),
                # Only survivors need a signature; skipping the rest saves the
                # dominant cost on a corpus with many short rows.
                "signature": (
                    signature(shingle_hashes(text, config.shingle_size), _WORKER["a"], _WORKER["b"])
                    if keep
                    else None
                ),
            }
        )
    return index, results


def _chunks(path: Path, size: int) -> Any:
    batch: list[str] = []
    index = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            batch.append(line)
            if len(batch) >= size:
                yield index, batch
                index += 1
                batch = []
    if batch:
        yield index, batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=64)
    parser.add_argument("--bands", type=int, default=8)
    parser.add_argument("--shingle-size", type=int, default=5)
    parser.add_argument("--minhash-seed", type=int, default=20260511)
    parser.add_argument("--min-characters", type=int, default=20)
    parser.add_argument(
        "--keep-pii-rows",
        action="store_true",
        help="Report personal identifiers without dropping the rows that carry them",
    )
    parser.add_argument("--chunk-rows", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()

    config = MinHashConfig(
        permutations=args.permutations,
        bands=args.bands,
        shingle_size=args.shingle_size,
        seed=args.minhash_seed,
    )
    thresholds = QualityThresholds(min_characters=args.min_characters)

    keep_flags: list[bool] = []
    reason_counts: Counter[str] = Counter()
    pii_counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    signatures: list[np.ndarray] = []
    survivor_positions: list[int] = []

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(config, thresholds, not args.keep_pii_rows),
    ) as pool:
        position = 0
        for _, results in pool.map(
            _assess_chunk, _chunks(args.input, args.chunk_rows), chunksize=1
        ):
            for result in results:
                keep_flags.append(bool(result["keep"]))
                language_counts[result["language"]] += 1
                for reason in result["reasons"]:
                    reason_counts[reason] += 1
                for kind, count in result["pii"].items():
                    pii_counts[kind] += count
                if result["keep"]:
                    signatures.append(result["signature"])
                    survivor_positions.append(position)
                position += 1

    total_rows = len(keep_flags)
    near_duplicates: set[int] = set()
    if signatures:
        stacked = np.vstack(signatures)
        # Indices here are into the survivor list, mapped back to source rows.
        near_duplicates = {survivor_positions[index] for index in duplicate_indices(stacked, config)}
    reason_counts["near_duplicate"] = len(near_duplicates)

    digest = hashlib.sha256()
    written = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{uuid.uuid4().hex}.tmp")
    try:
        with args.input.open("rb") as source, temporary.open("wb") as sink:
            position = 0
            for raw in source:
                if not raw.strip():
                    continue
                if keep_flags[position] and position not in near_duplicates:
                    line = raw.rstrip(b"\r\n") + b"\n"
                    sink.write(line)
                    digest.update(line)
                    written += 1
                position += 1
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)

    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    manifest = dict(source_manifest)
    manifest["output"] = {
        "path": args.output.as_posix(),
        "sha256": digest.hexdigest(),
        "size": args.output.stat().st_size,
        "rows": written,
    }
    manifest["filtering"] = {
        "source_rows": total_rows,
        "kept_rows": written,
        "removed_rows": total_rows - written,
        "removed_fraction": (total_rows - written) / total_rows if total_rows else 0.0,
        "reasons": dict(reason_counts),
        "pii_occurrences": dict(pii_counts),
        "language_profile": dict(language_counts),
        "near_duplicate_config": {
            "permutations": config.permutations,
            "bands": config.bands,
            "rows_per_band": config.rows_per_band,
            "shingle_size": config.shingle_size,
            "seed": config.seed,
            "detection_probability_at_0_9": config.detection_probability(0.9),
            "detection_probability_at_0_5": config.detection_probability(0.5),
        },
        "quality_thresholds": {
            "min_characters": thresholds.min_characters,
            "max_repetition_ratio": thresholds.max_repetition_ratio,
            "min_distinct_character_ratio": thresholds.min_distinct_character_ratio,
        },
        "pii_rows_dropped": not args.keep_pii_rows,
        "source_training_view": args.source_manifest.as_posix(),
        "source_output_sha256": source_manifest.get("output", {}).get("sha256"),
        "recorded_at": datetime.now(UTC).isoformat(),
    }
    write_json_atomic(args.manifest, manifest)
    write_json_atomic(args.report, {"schema_version": 1, **manifest["filtering"]})
    print(json.dumps({"source_rows": total_rows, "kept_rows": written}))


if __name__ == "__main__":
    main()
