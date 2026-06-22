import argparse
import hashlib
import heapq
import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = ROOT / "dataset" / "pretrain_t2t_mini.jsonl"
DEFAULT_OUT = ROOT / "experiments" / "pretrain" / "strict_splits"


def text_hash(text: str) -> str:
    normalized = " ".join(str(text).split())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def score_hash(seed: int, digest: str) -> int:
    payload = f"{seed}:{digest}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row["line"])


def push_candidate(heap: list, limit: int, item: dict) -> None:
    entry = (-item["score"], item["line_index"], item)
    if len(heap) < limit:
        heapq.heappush(heap, entry)
        return
    largest_score = -heap[0][0]
    if item["score"] < largest_score:
        heapq.heapreplace(heap, entry)


def first_pass(source: Path, seed: int, split_size: int) -> tuple[list[dict], dict]:
    heap: list = []
    seen = set()
    stats = {
        "source": str(source),
        "source_lines": 0,
        "nonempty_json_lines": 0,
        "unique_texts": 0,
        "duplicate_texts": 0,
        "skipped_bad_json": 0,
        "skipped_empty_text": 0,
    }
    limit = split_size * 2
    with source.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            stats["source_lines"] += 1
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                stats["skipped_bad_json"] += 1
                continue
            text = str(payload.get("text", ""))
            if not text.strip():
                stats["skipped_empty_text"] += 1
                continue
            stats["nonempty_json_lines"] += 1
            digest = text_hash(text)
            if digest in seen:
                stats["duplicate_texts"] += 1
                continue
            seen.add(digest)
            stats["unique_texts"] += 1
            push_candidate(
                heap,
                limit,
                {
                    "line_index": line_index,
                    "hash": digest,
                    "score": score_hash(seed, digest),
                    "line": line,
                    "chars": len(text),
                },
            )
    selected = [entry[2] for entry in heap]
    selected.sort(key=lambda item: item["score"])
    return selected, stats


def optional_write_train(source: Path, train_path: Path, reserved_hashes: set[str]) -> dict:
    seen = set()
    stats = {"train_lines": 0, "train_duplicates_skipped": 0, "train_reserved_skipped": 0}
    with source.open("r", encoding="utf-8") as src, train_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = str(payload.get("text", ""))
            if not text.strip():
                continue
            digest = text_hash(text)
            if digest in reserved_hashes:
                stats["train_reserved_skipped"] += 1
                continue
            if digest in seen:
                stats["train_duplicates_skipped"] += 1
                continue
            seen.add(digest)
            dst.write(line)
            stats["train_lines"] += 1
    stats["train_sha256"] = file_sha256(train_path)
    return stats


def write_manifest(out_dir: Path, metadata: dict) -> None:
    manifest = out_dir / "STRICT_SPLITS_MANIFEST.md"
    lines = [
        "# Strict Pretrain Splits Manifest",
        "",
        f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "This folder contains deterministic text-hash based validation and test splits for MiniMind pretraining.",
        "The selection is global over the source corpus after exact normalized-text deduplication, so it is stricter than using only the first 2k or tail 2k lines.",
        "",
        "## Files",
        "",
        f"- validation: `{metadata['val_path']}`",
        f"- test: `{metadata['test_path']}`",
        f"- metadata: `{metadata['meta_path']}`",
    ]
    if metadata.get("train_path"):
        lines.append(f"- train: `{metadata['train_path']}`")
    lines.extend(
        [
            "",
            "## Counts",
            "",
            f"- source lines: `{metadata['source_lines']}`",
            f"- unique texts: `{metadata['unique_texts']}`",
            f"- duplicate texts skipped: `{metadata['duplicate_texts']}`",
            f"- val lines: `{metadata['val_lines']}`",
            f"- test lines: `{metadata['test_lines']}`",
            "",
            "## Platform Command",
            "",
            "```bash",
            "python experiments/pretrain/scripts/prepare_strict_splits.py --write-train",
            "```",
        ]
    )
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create deterministic strict train/val/test splits for pretraining.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--split-size", type=int, default=2000)
    parser.add_argument("--write-train", action="store_true", help="Also write a de-duplicated train jsonl excluding val/test hashes.")
    args = parser.parse_args()

    source = Path(args.source)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected, stats = first_pass(source, args.seed, args.split_size)
    expected = args.split_size * 2
    if len(selected) < expected:
        raise RuntimeError(f"expected {expected} selected rows, got {len(selected)}")

    val_rows = sorted(selected[: args.split_size], key=lambda item: item["line_index"])
    test_rows = sorted(selected[args.split_size : expected], key=lambda item: item["line_index"])
    val_path = out_dir / "pretrain_strict_val_2k.jsonl"
    test_path = out_dir / "pretrain_strict_test_2k.jsonl"
    write_jsonl(val_path, val_rows)
    write_jsonl(test_path, test_rows)

    reserved_hashes = {item["hash"] for item in val_rows + test_rows}
    train_path = out_dir / "pretrain_strict_train.jsonl" if args.write_train else None
    train_stats = optional_write_train(source, train_path, reserved_hashes) if train_path else {}

    metadata = {
        **stats,
        **train_stats,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": args.seed,
        "split_size": args.split_size,
        "selection": "smallest sha256(seed:text_sha1) scores after normalized text deduplication",
        "val_path": str(val_path),
        "test_path": str(test_path),
        "train_path": str(train_path) if train_path else "",
        "meta_path": str(out_dir / "strict_splits_meta.json"),
        "val_lines": len(val_rows),
        "test_lines": len(test_rows),
        "val_sha256": file_sha256(val_path),
        "test_sha256": file_sha256(test_path),
        "val_min_line_index": min(item["line_index"] for item in val_rows),
        "val_max_line_index": max(item["line_index"] for item in val_rows),
        "test_min_line_index": min(item["line_index"] for item in test_rows),
        "test_max_line_index": max(item["line_index"] for item in test_rows),
        "val_text_hashes": [item["hash"] for item in val_rows],
        "test_text_hashes": [item["hash"] for item in test_rows],
    }
    meta_path = out_dir / "strict_splits_meta.json"
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    write_manifest(out_dir, metadata)
    print(meta_path)


if __name__ == "__main__":
    main()
