"""Tokenise a training view once into a memory-mapped block array.

Tokenising inside the training loop costs one saturated core per run while the
rest of the host idles, about 6.5% of wall clock in data wait, and a full
re-tokenisation of the corpus on every epoch. Measured on a 28-core host: two
training processes at 109% and 91% CPU with 390s and 342s of accumulated data
wait.

Doing it once produces a flat uint16 array of packed blocks. The training
loader then slices rather than parses, which keeps ``num_workers=0`` and the
absolute block cursor while removing the bottleneck entirely. uint16 is safe
here and checked at build time: the tokenizer's vocabulary is 6,400.

Ordering is preserved exactly. Shuffling remains the shuffled-view's job, so
this stage cannot silently change which data a run sees.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import uuid
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from python_starter.core.data_contract import write_json_atomic  # noqa: E402
from python_starter.core.tokenizer import load_tokenizer  # noqa: E402

DTYPE = np.uint16
_WORKER: dict[str, Any] = {}


def _init_worker(tokenizer_path: str, text_key: str) -> None:
    _WORKER["tokenizer"] = load_tokenizer(tokenizer_path)
    _WORKER["text_key"] = text_key


def _encode_chunk(payload: tuple[int, list[str]]) -> tuple[int, np.ndarray]:
    """Encode one chunk of raw JSONL lines into a token array."""
    index, lines = payload
    tokenizer = _WORKER["tokenizer"]
    text_key = _WORKER["text_key"]
    eos = tokenizer.eos_token_id
    tokens: list[int] = []
    for line in lines:
        row = json.loads(line)
        text = row[text_key]
        if not isinstance(text, str):
            raise ValueError(f"{text_key!r} must be a string")
        tokens.extend(tokenizer.encode(text, add_special_tokens=False))
        tokens.append(eos)
    return index, np.asarray(tokens, dtype=DTYPE)


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
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--text-key", default="text")
    parser.add_argument("--chunk-rows", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer)
    vocab_size = len(tokenizer)
    if vocab_size > np.iinfo(DTYPE).max + 1:
        raise SystemExit(
            f"vocabulary of {vocab_size} does not fit in {DTYPE.__name__}; widen the dtype"
        )
    if tokenizer.eos_token_id is None:
        raise SystemExit("tokenizer must define eos_token_id")

    block_size = args.max_length + 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{uuid.uuid4().hex}.tmp")

    carry = np.empty(0, dtype=DTYPE)
    blocks = 0
    rows = 0
    digest = hashlib.sha256()

    try:
        with (
            temporary.open("wb") as sink,
            ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=_init_worker,
                initargs=(args.tokenizer, args.text_key),
            ) as pool,
        ):
            # Ordered map: chunks are encoded in parallel but written back in
            # source order, so the block cursor means the same thing it did
            # when the loader tokenised inline.
            for _, tokens in pool.map(
                _encode_chunk, _chunks(args.input, args.chunk_rows), chunksize=1
            ):
                rows += 1
                carry = np.concatenate((carry, tokens)) if carry.size else tokens
                usable = (carry.size // block_size) * block_size
                if usable:
                    payload = carry[:usable]
                    sink.write(payload.tobytes())
                    digest.update(payload.tobytes())
                    blocks += usable // block_size
                    carry = carry[usable:]
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
        "rows": source_manifest.get("output", {}).get("rows"),
    }
    manifest["pretokenized"] = {
        "dtype": DTYPE.__name__,
        "block_size": block_size,
        "max_length": args.max_length,
        "blocks": blocks,
        "tokens": blocks * block_size,
        "vocab_size": vocab_size,
        "eos_token_id": int(tokenizer.eos_token_id),
        "tokenizer": str(args.tokenizer),
        "source_training_view": args.source_manifest.as_posix(),
        "source_output_sha256": source_manifest.get("output", {}).get("sha256"),
        "recorded_at": datetime.now(UTC).isoformat(),
        "note": (
            "Row order preserved; the trailing partial block is discarded, "
            "matching the inline packer."
        ),
    }
    write_json_atomic(args.manifest, manifest)
    print(json.dumps({"blocks": blocks, "tokens": blocks * block_size, "chunks": rows}))


if __name__ == "__main__":
    main()
