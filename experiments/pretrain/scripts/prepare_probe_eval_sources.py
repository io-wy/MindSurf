import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = ROOT / "experiments" / "pretrain" / "probe_eval_sets"


def normalize(text: str) -> str:
    return " ".join(str(text or "").split())


def stable_score(seed: int, text: str) -> int:
    digest = hashlib.sha256(f"{seed}:{normalize(text)}".encode("utf-8")).digest()
    return int.from_bytes(digest[:16], "big")


def compact_messages(conversations: list[dict]) -> str:
    parts = []
    for message in conversations:
        role = message.get("role", "")
        content = normalize(message.get("content", ""))
        if not content:
            continue
        if role == "system":
            continue
        if role == "user":
            parts.append(f"用户：{content}")
        elif role == "assistant":
            parts.append(f"助手：{content}")
    return "\n".join(parts)


def compact_math(item: dict) -> str:
    user_messages = [
        normalize(message.get("content", ""))
        for message in item.get("conversations", [])
        if message.get("role") == "user" and normalize(message.get("content", ""))
    ]
    answers = [normalize(value) for value in item.get("gt", []) if normalize(value)]
    if not user_messages or not answers:
        return ""
    return "问题：" + user_messages[-1] + "\n答案：" + "，".join(answers)


def select_rows(path: Path, mode: str, limit: int, seed: int) -> list[dict]:
    candidates = []
    seen = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if mode == "sft":
                text = compact_messages(item.get("conversations", []))
            elif mode == "math":
                text = compact_math(item)
            else:
                raise ValueError(mode)
            text = normalize(text)
            if not text:
                continue
            digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            candidates.append({"text": text, "score": stable_score(seed, text)})
    candidates.sort(key=lambda row: row["score"])
    selected = candidates[:limit] if limit > 0 else candidates
    return [{"text": row["text"]} for row in selected]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local probe PPL eval sets from SFT/math data.")
    parser.add_argument("--sft_path", type=Path, default=ROOT / "dataset" / "sft_t2t_mini.jsonl")
    parser.add_argument("--math_path", type=Path, default=ROOT / "dataset" / "agent_rl_math.jsonl")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sft_rows", type=int, default=2000)
    parser.add_argument("--math_rows", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260613)
    args = parser.parse_args()

    sft_path = args.sft_path if args.sft_path.is_absolute() else ROOT / args.sft_path
    math_path = args.math_path if args.math_path.is_absolute() else ROOT / args.math_path
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir

    sft_rows = select_rows(sft_path, "sft", args.sft_rows, args.seed)
    math_rows = select_rows(math_path, "math", args.math_rows, args.seed)
    sft_out = output_dir / f"sft_dialogue_probe_{len(sft_rows)}.jsonl"
    math_out = output_dir / f"math_tool_probe_{len(math_rows)}.jsonl"
    write_jsonl(sft_out, sft_rows)
    write_jsonl(math_out, math_rows)
    manifest = {
        "created_from": {
            "sft_path": str(sft_path),
            "math_path": str(math_path),
        },
        "selection": "smallest sha256(seed:normalized_text) after exact normalized dedup",
        "seed": args.seed,
        "sets": {
            "sft_dialogue_probe": {"path": str(sft_out), "rows": len(sft_rows)},
            "math_tool_probe": {"path": str(math_out), "rows": len(math_rows)},
        },
        "note": "Probe sets are for PPL comparison only. They are not strict pretraining holdouts unless the training recipe explicitly excludes these source files.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "probe_eval_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
