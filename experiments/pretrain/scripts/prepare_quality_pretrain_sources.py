import argparse
import json
import random
import re
from pathlib import Path


BOILERPLATE_PATTERNS = [
    "版权所有",
    "免责声明",
    "点击查看",
    "扫码",
    "关注公众号",
    "下载客户端",
    "广告",
]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def max_run_length(text: str) -> int:
    best = 0
    last = None
    run = 0
    for char in text:
        if char == last:
            run += 1
        else:
            last = char
            run = 1
        best = max(best, run)
    return best


def repeated_shingle_ratio(text: str, width: int = 6) -> float:
    chars = [char for char in text if not char.isspace()]
    if len(chars) < width * 8:
        return 0.0
    shingles = ["".join(chars[i : i + width]) for i in range(len(chars) - width + 1)]
    counts = {}
    for shingle in shingles:
        counts[shingle] = counts.get(shingle, 0) + 1
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / max(len(shingles), 1)


def is_quality_text(text: str) -> bool:
    text = normalize(text)
    length = len(text)
    if length < 80 or length > 1600:
        return False
    if any(pattern in text for pattern in BOILERPLATE_PATTERNS):
        return False
    if text.count("http://") + text.count("https://") > 2:
        return False
    chars = [char for char in text if not char.isspace()]
    if not chars:
        return False
    unique_ratio = len(set(chars)) / len(chars)
    if unique_ratio < 0.18:
        return False
    if max_run_length(text) >= 10:
        return False
    if repeated_shingle_ratio(text) > 0.22:
        return False
    return True


def reservoir_add(reservoir: list[dict], row: dict, max_rows: int, seen: int, rng: random.Random) -> None:
    if len(reservoir) < max_rows:
        reservoir.append(row)
        return
    index = rng.randrange(seen)
    if index < max_rows:
        reservoir[index] = row


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_mix(path: Path, strict_train: Path, supplement: Path, strict_weight: float, supplement_weight: float) -> None:
    payload = {
        "description": "Quality-filtered pretraining mix. Sources are sampled by weight.",
        "sources": [
            {"name": "strict_train", "path": str(strict_train), "weight": strict_weight},
            {"name": supplement.stem, "path": str(supplement), "weight": supplement_weight},
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a quality-filtered pretraining source from strict train.")
    parser.add_argument("--source", type=Path, default=Path("experiments/pretrain/strict_splits/pretrain_strict_train.jsonl"))
    parser.add_argument("--output_dir", type=Path, default=Path("experiments/pretrain/flywheel_sources"))
    parser.add_argument("--max_rows", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    reservoir = []
    seen_quality = 0
    seen_total = 0
    with args.source.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            seen_total += 1
            obj = json.loads(line)
            text = normalize(obj.get("text", ""))
            if not is_quality_text(text):
                continue
            seen_quality += 1
            reservoir_add(reservoir, {"text": text}, args.max_rows, seen_quality, rng)

    rng.shuffle(reservoir)
    quality_path = args.output_dir / f"pretrain_quality_core_{len(reservoir)}.jsonl"
    write_jsonl(quality_path, reservoir)
    write_mix(args.output_dir / "mix_strict90_quality10.json", args.source, quality_path, 0.90, 0.10)
    write_mix(args.output_dir / "mix_strict80_quality20.json", args.source, quality_path, 0.80, 0.20)

    manifest = {
        "source": str(args.source),
        "quality_path": str(quality_path),
        "seen_total": seen_total,
        "seen_quality": seen_quality,
        "written_rows": len(reservoir),
        "filters": {
            "length_chars": [80, 1600],
            "min_unique_char_ratio": 0.18,
            "max_same_char_run": 9,
            "max_repeated_6gram_ratio": 0.22,
            "max_urls": 2,
            "boilerplate_patterns": BOILERPLATE_PATTERNS,
        },
        "mixes": [
            str(args.output_dir / "mix_strict90_quality10.json"),
            str(args.output_dir / "mix_strict80_quality20.json"),
        ],
    }
    manifest_path = args.output_dir / "quality_pretrain_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
