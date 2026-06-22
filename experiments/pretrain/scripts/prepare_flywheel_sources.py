import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = ROOT / "experiments" / "pretrain" / "flywheel_sources"


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            text = row.get("text", "").strip()
            if not text:
                continue
            handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            count += 1
    return count


def compact_messages(conversations):
    parts = []
    for message in conversations:
        role = message.get("role", "")
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        if role == "user":
            parts.append(f"用户：{content}")
        elif role == "assistant":
            parts.append(f"助手：{content}")
    return "\n".join(parts)


def iter_sft(path: Path, limit: int):
    emitted = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if limit > 0 and emitted >= limit:
                break
            if not line.strip():
                continue
            item = json.loads(line)
            text = compact_messages(item.get("conversations", []))
            if text:
                emitted += 1
                yield {"text": text}


def iter_agent_math(path: Path, limit: int):
    emitted = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if limit > 0 and emitted >= limit:
                break
            if not line.strip():
                continue
            item = json.loads(line)
            conversations = item.get("conversations", [])
            user_texts = [
                str(message.get("content", "")).strip()
                for message in conversations
                if message.get("role") == "user" and str(message.get("content", "")).strip()
            ]
            answers = [str(value).strip() for value in item.get("gt", []) if str(value).strip()]
            if not user_texts or not answers:
                continue
            text = "\n".join([f"用户：{user_texts[-1]}", f"答案：{'；'.join(answers)}"])
            emitted += 1
            yield {"text": text}


def write_mix(path: Path, strict_train: Path, supplement: Path, strict_weight: float, supplement_weight: float):
    payload = {
        "description": "Data flywheel pretraining mix. Weights are source-sampling weights for StreamingMixedPackedPretrainDataset.",
        "sources": [
            {"name": "strict_train", "path": str(strict_train), "weight": strict_weight},
            {"name": supplement.stem, "path": str(supplement), "weight": supplement_weight},
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Prepare small data-flywheel sources for MiniMind pretraining continuation.")
    parser.add_argument("--sft_path", type=Path, default=ROOT / "dataset" / "sft_t2t_mini.jsonl")
    parser.add_argument("--agent_math_path", type=Path, default=ROOT / "dataset" / "agent_rl_math.jsonl")
    parser.add_argument("--strict_train_path", type=Path, default=ROOT / "experiments" / "pretrain" / "strict_splits" / "pretrain_strict_train.jsonl")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sft_limit", type=int, default=100000)
    parser.add_argument("--agent_math_limit", type=int, default=20000)
    parser.add_argument("--strict_weight", type=float, default=0.95)
    parser.add_argument("--supplement_weight", type=float, default=0.05)
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    sft_output = output_dir / f"sft_dialogue_as_pretrain_{args.sft_limit}.jsonl"
    math_output = output_dir / f"agent_math_as_pretrain_{args.agent_math_limit}.jsonl"
    sft_count = write_jsonl(sft_output, iter_sft(args.sft_path, args.sft_limit))
    math_count = write_jsonl(math_output, iter_agent_math(args.agent_math_path, args.agent_math_limit))
    write_mix(output_dir / "mix_strict95_sft5.json", args.strict_train_path, sft_output, args.strict_weight, args.supplement_weight)
    write_mix(output_dir / "mix_strict95_math5.json", args.strict_train_path, math_output, args.strict_weight, args.supplement_weight)
    manifest = {
        "sft_output": str(sft_output),
        "sft_rows": sft_count,
        "math_output": str(math_output),
        "math_rows": math_count,
        "mixes": [
            str(output_dir / "mix_strict95_sft5.json"),
            str(output_dir / "mix_strict95_math5.json"),
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
