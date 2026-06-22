import argparse
import hashlib
import json
import random
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = ROOT / "experiments" / "pretrain" / "flywheel_sources"
DEFAULT_QUALITY_CORE = DEFAULT_OUTPUT_DIR / "pretrain_quality_core_200000.jsonl"

BAD_PHRASES = (
    "jingyaogong",
    "我是由",
    "背后的模型",
    "模型版本",
    "训练数据来源",
    "作为AI",
    "作为 AI",
    "我无法观看",
    "我无法阅读",
    "我没有味觉",
    "我没有个人",
    "无法浏览",
    "I cannot",
    "Please check",
    "random number",
    "current weather",
)

CODE_MARKERS = (
    "```",
    " def ",
    "class ",
    "import ",
    "return ",
    "function ",
    "javascript",
    "for ",
    "while ",
    "if ",
    "SELECT ",
)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def text_hash(text: str) -> str:
    return hashlib.sha1(normalize(text).encode("utf-8")).hexdigest()


def stable_key(seed: int, text: str) -> int:
    digest = hashlib.sha256(f"{seed}:{text_hash(text)}".encode("utf-8")).digest()
    return int.from_bytes(digest[:16], "big")


def ascii_alpha_ratio(text: str) -> float:
    chars = [char for char in text if not char.isspace()]
    if not chars:
        return 0.0
    return sum(1 for char in chars if "a" <= char.lower() <= "z") / len(chars)


def unique_char_ratio(text: str) -> float:
    chars = [char for char in text if not char.isspace()]
    if not chars:
        return 0.0
    return len(set(chars)) / len(chars)


def repeated_shingle_ratio(text: str, width: int = 6) -> float:
    chars = [char for char in text if not char.isspace()]
    if len(chars) < width * 8:
        return 0.0
    shingles = ["".join(chars[index : index + width]) for index in range(len(chars) - width + 1)]
    counts = {}
    for shingle in shingles:
        counts[shingle] = counts.get(shingle, 0) + 1
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / max(len(shingles), 1)


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


def is_quality_text(text: str, min_chars: int = 80, max_chars: int = 2000) -> bool:
    text = normalize(text)
    if len(text) < min_chars or len(text) > max_chars:
        return False
    if any(phrase in text for phrase in BAD_PHRASES):
        return False
    if text.count("http://") + text.count("https://") > 1:
        return False
    if unique_char_ratio(text) < 0.16:
        return False
    if max_run_length(text) >= 10:
        return False
    if repeated_shingle_ratio(text) > 0.25:
        return False
    return True


def looks_code(text: str) -> bool:
    if "```" in text:
        return True
    lower = text.lower()
    if re.search(r"\bdef\s+\w+\s*\(", lower):
        return True
    if re.search(r"\bimport\s+[a-zA-Z_]\w*", lower):
        return True
    if re.search(r"\breturn\s+.+", lower):
        return True
    if re.search(r"\bfor\s+.+\s+in\s+.+:", lower):
        return True
    if re.search(r"\bwhile\s+.+:", lower):
        return True
    if re.search(r"\bif\s+.+:", lower):
        return True
    if re.search(r"\bselect\b.+\bfrom\b", lower):
        return True
    return False


def looks_english(text: str) -> bool:
    return ascii_alpha_ratio(text) >= 0.55 and len(text.split()) >= 20


def looks_calibration(text: str) -> bool:
    return any(phrase in text for phrase in ("不确定", "无法确认", "不知道", "查证", "核实", "可靠来源"))


def reservoir_add(reservoir: list[dict], row: dict, limit: int, seen: int, rng: random.Random) -> None:
    if limit <= 0:
        return
    if len(reservoir) < limit:
        reservoir.append(row)
        return
    index = rng.randrange(seen)
    if index < limit:
        reservoir[index] = row


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def iter_assistant_texts(paths: list[Path]):
    for path in paths:
        for item in read_jsonl(path):
            for message in item.get("conversations", []):
                if message.get("role") != "assistant":
                    continue
                text = normalize(message.get("content", ""))
                if text:
                    yield text


def iter_agent_math(path: Path):
    for item in read_jsonl(path):
        user_texts = [
            normalize(message.get("content", ""))
            for message in item.get("conversations", [])
            if message.get("role") == "user" and normalize(message.get("content", ""))
        ]
        answers = [normalize(answer) for answer in item.get("gt", []) if normalize(answer)]
        if not user_texts or not answers:
            continue
        yield f"数学计算题：{user_texts[-1]} 结果：{'；'.join(answers)}。"


def write_jsonl(path: Path, rows: list[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda row: stable_key(0, row["text"]))
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({"text": row["text"]}, ensure_ascii=False) + "\n")
    return len(rows)


def write_mix(path: Path, sources: list[tuple[str, Path, float]]) -> None:
    payload = {
        "description": "Gap-focused pretraining mix built after MCQ benchmark v1 exposed math, code, english, long-context and calibration weaknesses.",
        "sources": [
            {
                "name": name,
                "path": str(source.relative_to(ROOT) if source.is_relative_to(ROOT) else source),
                "weight": weight,
            }
            for name, source, weight in sources
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare filtered pretraining sources for benchmark gaps.")
    parser.add_argument("--sft_path", type=Path, default=ROOT / "dataset" / "sft_t2t_mini.jsonl")
    parser.add_argument("--agent_path", type=Path, default=ROOT / "dataset" / "agent_rl.jsonl")
    parser.add_argument("--agent_math_path", type=Path, default=ROOT / "dataset" / "agent_rl_math.jsonl")
    parser.add_argument("--quality_core", type=Path, default=DEFAULT_QUALITY_CORE)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--general_rows", type=int, default=60000)
    parser.add_argument("--english_rows", type=int, default=30000)
    parser.add_argument("--code_rows", type=int, default=30000)
    parser.add_argument("--long_rows", type=int, default=30000)
    parser.add_argument("--calibration_rows", type=int, default=20000)
    parser.add_argument("--math_rows", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=20260615)
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    quality_core = args.quality_core if args.quality_core.is_absolute() else ROOT / args.quality_core
    rng = random.Random(args.seed)
    limits = {
        "general": args.general_rows,
        "english": args.english_rows,
        "code": args.code_rows,
        "long": args.long_rows,
        "calibration": args.calibration_rows,
        "math": args.math_rows,
    }
    reservoirs = {name: [] for name in limits}
    seen = {name: 0 for name in limits}
    rejected = {"duplicate": 0, "quality": 0}
    hashes = set()

    assistant_paths = [args.sft_path, args.agent_path]
    assistant_paths = [path if path.is_absolute() else ROOT / path for path in assistant_paths]
    agent_math_path = args.agent_math_path if args.agent_math_path.is_absolute() else ROOT / args.agent_math_path

    for text in iter_assistant_texts(assistant_paths):
        text = normalize(text)
        digest = text_hash(text)
        if digest in hashes:
            rejected["duplicate"] += 1
            continue
        hashes.add(digest)
        if not is_quality_text(text):
            rejected["quality"] += 1
            continue
        row = {"text": text}
        seen["general"] += 1
        reservoir_add(reservoirs["general"], row, limits["general"], seen["general"], rng)
        if looks_english(text):
            seen["english"] += 1
            reservoir_add(reservoirs["english"], row, limits["english"], seen["english"], rng)
        if looks_code(text):
            seen["code"] += 1
            reservoir_add(reservoirs["code"], row, limits["code"], seen["code"], rng)
        if len(text) >= 700:
            seen["long"] += 1
            reservoir_add(reservoirs["long"], row, limits["long"], seen["long"], rng)
        if looks_calibration(text):
            seen["calibration"] += 1
            reservoir_add(reservoirs["calibration"], row, limits["calibration"], seen["calibration"], rng)

    for text in iter_agent_math(agent_math_path):
        text = normalize(text)
        digest = text_hash(text)
        if digest in hashes:
            rejected["duplicate"] += 1
            continue
        hashes.add(digest)
        if not is_quality_text(text, min_chars=20, max_chars=1000):
            rejected["quality"] += 1
            continue
        seen["math"] += 1
        reservoir_add(reservoirs["math"], {"text": text}, limits["math"], seen["math"], rng)

    paths = {
        name: output_dir / f"gap_{name}_{len(rows)}.jsonl"
        for name, rows in reservoirs.items()
        if rows
    }
    written = {name: write_jsonl(path, reservoirs[name]) for name, path in paths.items()}

    if quality_core.exists():
        write_mix(
            output_dir / "mix_quality55_gap_math15_code10_english10_long5_calib5.json",
            [
                ("quality_core", quality_core, 0.55),
                ("gap_math", paths["math"], 0.15),
                ("gap_code", paths["code"], 0.10),
                ("gap_english", paths["english"], 0.10),
                ("gap_long", paths["long"], 0.05),
                ("gap_calibration", paths["calibration"], 0.05),
            ],
        )
        write_mix(
            output_dir / "mix_quality65_gap_math15_code10_english5_long5.json",
            [
                ("quality_core", quality_core, 0.65),
                ("gap_math", paths["math"], 0.15),
                ("gap_code", paths["code"], 0.10),
                ("gap_english", paths["english"], 0.05),
                ("gap_long", paths["long"], 0.05),
            ],
        )

    manifest = {
        "sources": {
            "sft_path": str(args.sft_path),
            "agent_path": str(args.agent_path),
            "agent_math_path": str(args.agent_math_path),
        },
        "seen": seen,
        "written": {name: {"rows": count, "path": str(paths[name])} for name, count in written.items()},
        "rejected": rejected,
        "filters": {
            "bad_phrases": list(BAD_PHRASES),
            "general": "assistant content only; 80-2000 chars; low repetition; no identity boilerplate",
            "math": "agent math user query plus ground-truth answer; not from eval benchmark",
            "note": "Do not train on local_mcq_benchmark_v1 items.",
        },
        "mixes": [
            str(output_dir / "mix_quality55_gap_math15_code10_english10_long5_calib5.json"),
            str(output_dir / "mix_quality65_gap_math15_code10_english5_long5.json"),
        ],
    }
    manifest_path = output_dir / "gap_pretrain_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
