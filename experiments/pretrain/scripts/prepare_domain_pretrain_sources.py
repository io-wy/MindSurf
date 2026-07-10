import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments" / "pretrain" / "scripts"))

from prepare_quality_pretrain_sources import is_quality_text, normalize, repeated_shingle_ratio  # noqa: E402


DEFAULT_SOURCE = ROOT / "experiments" / "pretrain" / "strict_splits" / "pretrain_strict_train.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "experiments" / "pretrain" / "flywheel_sources"
DEFAULT_QUALITY_CORE = ROOT / "experiments" / "pretrain" / "flywheel_sources" / "pretrain_quality_core_200000.jsonl"

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
    "javascript",
    "java",
    "sql",
    "{",
    "}",
    "->",
    "=>",
)


def text_hash(text: str) -> str:
    return hashlib.sha1(" ".join(text.split()).encode("utf-8")).hexdigest()


def stable_key(seed: int, text: str) -> int:
    digest = hashlib.sha256(f"{seed}:{text_hash(text)}".encode("utf-8")).digest()
    return int.from_bytes(digest[:16], "big")


def ascii_alpha_ratio(text: str) -> float:
    chars = [char for char in text if not char.isspace()]
    if not chars:
        return 0.0
    return sum(1 for char in chars if "a" <= char.lower() <= "z") / len(chars)


def digit_operator_ratio(text: str) -> float:
    chars = [char for char in text if not char.isspace()]
    if not chars:
        return 0.0
    return sum(1 for char in chars if char.isdigit() or char in "+-*/=^%<>") / len(chars)


def looks_code_or_english(text: str) -> bool:
    lower = text.lower()
    return ascii_alpha_ratio(text) >= 0.35 or any(marker in lower for marker in CODE_MARKERS)


def looks_math(text: str) -> bool:
    return digit_operator_ratio(text) >= 0.08 or re.search(r"\b\d+\s*[\+\-\*/=]\s*\d+", text) is not None


def reservoir_add(reservoir: list[dict], row: dict, limit: int, seen: int, rng: random.Random) -> None:
    if len(reservoir) < limit:
        reservoir.append(row)
        return
    index = rng.randrange(seen)
    if index < limit:
        reservoir[index] = row


def write_jsonl(path: Path, rows: list[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: stable_key(0, row["text"]))
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({"text": row["text"]}, ensure_ascii=False) + "\n")
    return len(rows)


def write_mix(path: Path, sources: list[tuple[str, Path, float]]) -> None:
    payload = {
        "description": "Diagnostic-aware pretraining mix. Weights are source-sampling weights.",
        "sources": [
            {"name": name, "path": str(source.relative_to(ROOT) if source.is_relative_to(ROOT) else source), "weight": weight}
            for name, source, weight in sources
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare pretraining sources for weak diagnostic buckets.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--quality_core", type=Path, default=DEFAULT_QUALITY_CORE)
    parser.add_argument("--english_code_rows", type=int, default=80000)
    parser.add_argument("--long_rows", type=int, default=50000)
    parser.add_argument("--math_rows", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=20260613)
    args = parser.parse_args()

    source = args.source if args.source.is_absolute() else ROOT / args.source
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    quality_core = args.quality_core if args.quality_core.is_absolute() else ROOT / args.quality_core
    rng = random.Random(args.seed)
    reservoirs = {
        "english_code": [],
        "long": [],
        "math": [],
    }
    seen = {name: 0 for name in reservoirs}
    seen_total = 0
    seen_quality = 0
    seen_duplicate = 0
    hashes = set()

    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            text = normalize(payload.get("text", ""))
            if not text:
                continue
            digest = text_hash(text)
            if digest in hashes:
                seen_duplicate += 1
                continue
            hashes.add(digest)
            seen_total += 1
            if not is_quality_text(text):
                continue
            seen_quality += 1
            row = {"text": text}
            if looks_code_or_english(text):
                seen["english_code"] += 1
                reservoir_add(reservoirs["english_code"], row, args.english_code_rows, seen["english_code"], rng)
            if len(text) >= 700:
                seen["long"] += 1
                reservoir_add(reservoirs["long"], row, args.long_rows, seen["long"], rng)
            if looks_math(text):
                seen["math"] += 1
                reservoir_add(reservoirs["math"], row, args.math_rows, seen["math"], rng)

    paths = {
        "english_code": output_dir / f"pretrain_domain_english_code_{len(reservoirs['english_code'])}.jsonl",
        "long": output_dir / f"pretrain_domain_long_{len(reservoirs['long'])}.jsonl",
        "math": output_dir / f"pretrain_domain_math_{len(reservoirs['math'])}.jsonl",
    }
    written = {name: write_jsonl(path, reservoirs[name]) for name, path in paths.items()}

    mixes = {
        "mix_quality75_ec15_long10.json": [
            ("quality_core", quality_core, 0.75),
            ("english_code", paths["english_code"], 0.15),
            ("long", paths["long"], 0.10),
        ],
        "mix_quality70_ec15_long10_math5.json": [
            ("quality_core", quality_core, 0.70),
            ("english_code", paths["english_code"], 0.15),
            ("long", paths["long"], 0.10),
            ("math", paths["math"], 0.05),
        ],
        "mix_quality85_long15.json": [
            ("quality_core", quality_core, 0.85),
            ("long", paths["long"], 0.15),
        ],
    }
    for filename, sources in mixes.items():
        write_mix(output_dir / filename, sources)

    manifest = {
        "source": str(source),
        "quality_core": str(quality_core),
        "seen_total": seen_total,
        "seen_quality": seen_quality,
        "seen_duplicate": seen_duplicate,
        "candidate_counts": seen,
        "written": {name: {"rows": count, "path": str(paths[name])} for name, count in written.items()},
        "filters": {
            "quality_filter": "prepare_quality_pretrain_sources.is_quality_text",
            "english_code": "ascii_alpha_ratio >= 0.35 or code marker",
            "long": "quality text with chars >= 700",
            "math": "digit/operator ratio >= 0.08 or simple arithmetic pattern",
        },
        "mixes": {filename: str(output_dir / filename) for filename in mixes},
        "note": "These are still pretraining text sources, not SFT training.",
    }
    manifest_path = output_dir / "domain_pretrain_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
