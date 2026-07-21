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
        required=True,
        help="Training-view manifest of the input; identity and safety gates are inherited",
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

    # A shuffled view is still a training view and must satisfy the same
    # contract, so the identity and safety gates are inherited rather than
    # re-asserted: reordering rows cannot change which rows are present, so the
    # source's dedup and holdout guarantees carry over unchanged.
    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    if source_manifest.get("output", {}).get("rows") != len(offsets):
        raise SystemExit(
            f"row count disagrees with the source manifest: "
            f"{source_manifest.get('output', {}).get('rows')} vs {len(offsets)}"
        )

    manifest = dict(source_manifest)
    manifest["output"] = {
        "path": args.output.as_posix(),
        "sha256": digest.hexdigest(),
        "size": args.output.stat().st_size,
        "rows": len(offsets),
    }
    manifest["derivation"] = {
        "kind": "global row permutation",
        "shuffle_seed": args.seed,
        "recorded_at": datetime.now(UTC).isoformat(),
        "source_training_view": args.source_manifest.as_posix(),
        "source_output_sha256": source_manifest.get("output", {}).get("sha256"),
        "note": "Row content is untouched; only order changes.",
    }
    write_json_atomic(args.manifest, manifest)
    print(json.dumps({"rows": len(offsets), "sha256": digest.hexdigest()}))


if __name__ == "__main__":
    main()
