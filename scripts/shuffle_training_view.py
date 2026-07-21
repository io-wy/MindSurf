"""Globally shuffle a training view into a new, separately identified view.

A window shuffle cannot fix source ordering: with a 20,000-row buffer over a
1.27M-row file, the English and code section still only appears near the end,
so any run consuming less than a full epoch still never sees it. Only a global
permutation makes a prefix a representative sample.

The output is a new file with its own digest rather than an in-place rewrite,
so the ordered view stays reproducible and the two can be compared.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from python_starter.core.data_contract import write_json_atomic  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        help="Training-view manifest of the input, carried through for lineage",
    )
    args = parser.parse_args()

    # Offsets rather than payloads: 8 bytes a row instead of the row itself, so
    # an 8 GB corpus costs about 70 MB here and never has to fit in memory.
    offsets: list[tuple[int, int]] = []
    with args.input.open("rb") as handle:
        position = 0
        for raw in handle:
            length = len(raw)
            if raw.strip():
                offsets.append((position, length))
            position += length

    random.Random(args.seed).shuffle(offsets)

    digest = hashlib.sha256()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{uuid.uuid4().hex}.tmp")
    try:
        with args.input.open("rb") as source, temporary.open("wb") as sink:
            for start, length in offsets:
                source.seek(start)
                line = source.read(length).rstrip(b"\r\n") + b"\n"
                sink.write(line)
                digest.update(line)
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)

    source_manifest = (
        json.loads(args.source_manifest.read_text(encoding="utf-8"))
        if args.source_manifest and args.source_manifest.is_file()
        else None
    )
    write_json_atomic(
        args.manifest,
        {
            "schema_version": 1,
            "recorded_at": datetime.now(UTC).isoformat(),
            "derivation": "global row permutation",
            "shuffle_seed": args.seed,
            "input": {
                "path": args.input.as_posix(),
                "rows": len(offsets),
                "sha256": (source_manifest or {}).get("output", {}).get("sha256"),
            },
            "output": {
                "path": args.output.as_posix(),
                "rows": len(offsets),
                "sha256": digest.hexdigest(),
                "size": args.output.stat().st_size,
            },
            "source_training_view": (
                args.source_manifest.as_posix() if args.source_manifest else None
            ),
            "note": (
                "Row content is untouched; only order changes. Deduplication and "
                "holdout removal are inherited from the source view."
            ),
        },
    )
    print(json.dumps({"rows": len(offsets), "sha256": digest.hexdigest()}))


if __name__ == "__main__":
    main()
