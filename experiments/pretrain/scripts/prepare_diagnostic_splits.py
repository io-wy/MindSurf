import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments" / "pretrain" / "scripts"))

from prepare_quality_pretrain_sources import (  # noqa: E402
    is_quality_text,
    max_run_length,
    normalize,
    repeated_shingle_ratio,
)
from prepare_strict_splits import text_hash  # noqa: E402


DEFAULT_OUTPUT_DIR = ROOT / "experiments" / "pretrain" / "diagnostic_splits"
DEFAULT_HELDOUT = [
    ROOT / "experiments" / "pretrain" / "strict_splits" / "pretrain_strict_val_2k.jsonl",
    ROOT / "experiments" / "pretrain" / "strict_splits" / "pretrain_strict_test_2k.jsonl",
]
DEFAULT_TRAIN = ROOT / "experiments" / "pretrain" / "strict_splits" / "pretrain_strict_train.jsonl"

MOJIBAKE_MARKERS = (
    "锛",
    "銆",
    "鐨",
    "璇",
    "绛",
    "鍙",
    "浣",
    "涓",
    "€",
    "俓",
    "鏄",
    "鍦",
    "姣",
    "甯",
    "殑",
)
CODE_MARKERS = (
    "```",
    " def ",
    "class ",
    "import ",
    "return ",
    "python",
    "torch",
    "json",
    "function",
    "{",
    "}",
    "->",
    "=>",
)


def stable_score(seed: int, text: str) -> int:
    digest = hashlib.sha256(f"{seed}:{text_hash(text)}".encode("utf-8")).digest()
    return int.from_bytes(digest[:16], "big")


def ascii_alpha_ratio(text: str) -> float:
    chars = [char for char in text if not char.isspace()]
    if not chars:
        return 0.0
    alpha = sum(1 for char in chars if ("a" <= char.lower() <= "z"))
    return alpha / len(chars)


def digit_operator_ratio(text: str) -> float:
    chars = [char for char in text if not char.isspace()]
    if not chars:
        return 0.0
    math_chars = sum(1 for char in chars if char.isdigit() or char in "+-*/=^%<>")
    return math_chars / len(chars)


def mojibake_score(text: str) -> float:
    if not text:
        return 0.0
    hits = sum(text.count(marker) for marker in MOJIBAKE_MARKERS)
    return hits / max(len(text), 1)


def classify(text: str) -> tuple[set[str], dict]:
    norm = normalize(text)
    lower = norm.lower()
    length = len(norm)
    repeat_ratio = repeated_shingle_ratio(norm)
    max_run = max_run_length(norm)
    quality = is_quality_text(norm)
    ascii_ratio = ascii_alpha_ratio(norm)
    math_ratio = digit_operator_ratio(norm)
    mojibake = mojibake_score(norm)

    tags = {"all_heldout"}
    tags.add("quality_pass" if quality else "quality_fail")
    if length < 200:
        tags.add("length_short")
    elif length <= 800:
        tags.add("length_medium")
    else:
        tags.add("length_long")
    if repeat_ratio > 0.12 or max_run >= 7:
        tags.add("repeat_high")
    else:
        tags.add("repeat_low")
    if ascii_ratio >= 0.35:
        tags.add("english_or_code_heavy")
    if any(marker in lower for marker in CODE_MARKERS):
        tags.add("code_like")
    if math_ratio >= 0.08 or re.search(r"\b\d+\s*[\+\-\*/=]\s*\d+", norm):
        tags.add("math_like")
    if mojibake >= 0.012:
        tags.add("mojibake_like")
    else:
        tags.add("mojibake_light")

    features = {
        "chars": length,
        "quality": quality,
        "repeat_ratio": round(repeat_ratio, 5),
        "max_run": max_run,
        "ascii_alpha_ratio": round(ascii_ratio, 5),
        "digit_operator_ratio": round(math_ratio, 5),
        "mojibake_score": round(mojibake, 5),
    }
    return tags, features


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            text = normalize(item.get("text", ""))
            if not text:
                continue
            yield {
                "text": text,
                "line": json.dumps({"text": text}, ensure_ascii=False) + "\n",
                "source_path": str(path),
                "line_number": line_number,
                "hash": text_hash(text),
            }


def select_bucket(rows: list[dict], max_rows: int, seed: int) -> list[dict]:
    rows = sorted(rows, key=lambda row: (stable_score(seed, row["text"]), row["hash"]))
    if max_rows > 0:
        rows = rows[:max_rows]
    return sorted(rows, key=lambda row: (row["source_path"], row["line_number"]))


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row["line"])


def summarize_train_distribution(train_path: Path, max_scan_rows: int) -> dict:
    counters = Counter()
    total = 0
    for row in iter_jsonl(train_path):
        total += 1
        tags, _ = classify(row["text"])
        counters.update(tags - {"all_heldout"})
        if max_scan_rows > 0 and total >= max_scan_rows:
            break
    return {
        "path": str(train_path),
        "scanned_rows": total,
        "tag_counts": dict(sorted(counters.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic diagnostic holdout buckets for pretraining eval.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--heldout", type=Path, nargs="+", default=DEFAULT_HELDOUT)
    parser.add_argument("--train_path", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--max_rows_per_bucket", type=int, default=512)
    parser.add_argument("--min_rows", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--train_scan_rows", type=int, default=50000)
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    buckets: dict[str, list[dict]] = defaultdict(list)
    feature_summaries: dict[str, list[dict]] = defaultdict(list)
    seen_hashes = set()
    source_counts = Counter()

    for path in args.heldout:
        path = path if path.is_absolute() else ROOT / path
        for row in iter_jsonl(path):
            if row["hash"] in seen_hashes:
                continue
            seen_hashes.add(row["hash"])
            tags, features = classify(row["text"])
            row["features"] = features
            source_counts[str(path)] += 1
            for tag in tags:
                buckets[tag].append(row)
                feature_summaries[tag].append(features)

    manifest = {
        "heldout_sources": [str(path if path.is_absolute() else ROOT / path) for path in args.heldout],
        "output_dir": str(output_dir),
        "max_rows_per_bucket": args.max_rows_per_bucket,
        "min_rows": args.min_rows,
        "seed": args.seed,
        "source_counts": dict(source_counts),
        "buckets": {},
        "train_distribution_sample": {},
    }
    counts_lines = ["bucket\trows_written\trows_available\tpath\tmean_chars\tmean_repeat\tmean_mojibake"]

    for bucket_name, rows in sorted(buckets.items()):
        if len(rows) < args.min_rows:
            continue
        selected = select_bucket(rows, args.max_rows_per_bucket, args.seed)
        path = output_dir / f"{bucket_name}.jsonl"
        write_jsonl(path, selected)
        features = [row["features"] for row in rows]
        mean_chars = sum(item["chars"] for item in features) / max(len(features), 1)
        mean_repeat = sum(item["repeat_ratio"] for item in features) / max(len(features), 1)
        mean_mojibake = sum(item["mojibake_score"] for item in features) / max(len(features), 1)
        manifest["buckets"][bucket_name] = {
            "path": str(path),
            "rows_available": len(rows),
            "rows_written": len(selected),
            "mean_chars": round(mean_chars, 2),
            "mean_repeat_ratio": round(mean_repeat, 5),
            "mean_mojibake_score": round(mean_mojibake, 5),
        }
        counts_lines.append(
            "\t".join(
                [
                    bucket_name,
                    str(len(selected)),
                    str(len(rows)),
                    str(path),
                    f"{mean_chars:.2f}",
                    f"{mean_repeat:.5f}",
                    f"{mean_mojibake:.5f}",
                ]
            )
        )

    train_path = args.train_path if args.train_path.is_absolute() else ROOT / args.train_path
    if train_path.exists():
        manifest["train_distribution_sample"] = summarize_train_distribution(train_path, args.train_scan_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "diagnostic_split_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "diagnostic_split_counts.tsv").write_text("\n".join(counts_lines) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
