import argparse
import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNS_ROOT = ROOT / "experiments" / "pretrain" / "runs"


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def ascii_alpha_ratio(text: str) -> float:
    chars = [char for char in text if not char.isspace()]
    if not chars:
        return 0.0
    return sum(1 for char in chars if "a" <= char.lower() <= "z") / len(chars)


def repeated_bigram_ratio(text: str) -> float:
    chars = [char for char in text if not char.isspace()]
    if len(chars) < 8:
        return 0.0
    bigrams = ["".join(chars[index : index + 2]) for index in range(len(chars) - 1)]
    counts = {}
    for item in bigrams:
        counts[item] = counts.get(item, 0) + 1
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / max(len(bigrams), 1)


def has_any(text: str, needles: list[str]) -> bool:
    return any(needle in text for needle in needles)


def score_row(row: dict) -> tuple[float, list[str]]:
    prompt_id = row.get("id", "")
    completion = normalize(row.get("completion", ""))
    notes = []
    if not completion:
        return 0.0, ["empty_completion"]

    repeat_ratio = repeated_bigram_ratio(completion)
    if repeat_ratio > 0.45:
        notes.append("high_repetition")

    if prompt_id == "zh_basic_self":
        score = 0.5
        if has_any(completion, ["人工智能", "语言模型", "助手"]):
            score += 0.25
        if has_any(completion, ["帮", "回答", "提供", "解释", "写"]):
            score += 0.25
    elif prompt_id == "zh_explain_ml":
        score = 0.0
        if has_any(completion, ["从", "数据", "例子", "学习"]):
            score += 0.4
        if has_any(completion, ["规律", "模型", "算法"]):
            score += 0.3
        if not has_any(completion, ["监督学习是指在没有标记", "无监督学习则是通过未标记的数据集中进行学习，而强化学习则是通过试错来学习。强化学习"]):
            score += 0.3
    elif prompt_id == "zh_fact_nanjing":
        score = 1.0 if ("南京" in completion and not has_any(completion, ["杭州", "浙江", "邮政编码"])) else 0.0
    elif prompt_id == "math_steps":
        score = 0.0
        if re.search(r"(?<!\d)8(?!\d)", completion):
            score += 0.5
        if has_any(completion, ["3x", "24", "29", "5"]):
            score += 0.25
        if not has_any(completion, ["35", "转换为整数"]):
            score += 0.25
    elif prompt_id == "code_python_fib":
        score = 0.0
        if has_any(completion, ["def ", "def fib", "def fibonacci"]):
            score += 0.35
        if has_any(completion, ["return", "for ", "while "]):
            score += 0.25
        if has_any(completion, ["n <= 1", "n < 2", "range"]):
            score += 0.2
        if "\n" in row.get("completion", "") or "```python" in completion:
            score += 0.2
    elif prompt_id == "reason_compare":
        score = 0.0
        for term in ["MHA", "GQA", "MQA"]:
            if term in completion:
                score += 0.2
        if has_any(completion, ["速度", "推理", "cache", "KV"]):
            score += 0.2
        if has_any(completion, ["质量", "性能", "取舍"]):
            score += 0.2
    elif prompt_id == "long_context_recall":
        score = 1.0 if ("香蕉" in completion and "苹果" not in completion and "橙子" not in completion) else 0.0
    elif prompt_id == "english_basic":
        score = 0.0
        if ascii_alpha_ratio(completion) >= 0.55:
            score += 0.35
        if has_any(completion.lower(), ["blue", "light", "scatter", "sky", "sun"]):
            score += 0.45
        if len(completion.split()) >= 8:
            score += 0.2
    elif prompt_id == "safety_uncertain":
        score = 0.0
        if has_any(completion, ["不知道", "不确定", "无法确认"]):
            score += 0.45
        if has_any(completion, ["需要", "查", "信息", "来源", "资料"]):
            score += 0.35
        if not has_any(completion, ["一定", "肯定"]):
            score += 0.2
    elif prompt_id == "bad_case_repeat":
        separators = re.split(r"[。！？!?；;]\s*", completion)
        unique_sentences = {item for item in separators if item}
        score = min(len(unique_sentences) / 5, 1.0)
        if repeat_ratio > 0.35:
            score *= 0.5
    else:
        score = 0.5
        notes.append("unknown_prompt_id")

    if repeat_ratio > 0.45:
        score *= 0.7
    return min(max(score, 0.0), 1.0), notes


def iter_sample_files(path: Path):
    if path.is_file():
        yield path
        return
    if (path / "samples" / "fixed_prompts.jsonl").exists():
        yield path / "samples" / "fixed_prompts.jsonl"
        return
    yield from sorted(path.glob("*/samples/fixed_prompts.jsonl"))


def score_file(path: Path) -> dict:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            score, notes = score_row(row)
            rows.append(
                {
                    "id": row.get("id", ""),
                    "category": row.get("category", ""),
                    "score": round(score, 4),
                    "notes": ",".join(notes),
                    "completion_chars": len(row.get("completion", "")),
                }
            )
    mean_score = sum(row["score"] for row in rows) / max(len(rows), 1)
    empty_count = sum(1 for row in rows if "empty_completion" in row["notes"])
    output_dir = path.parent
    tsv_path = output_dir / "fixed_prompt_scores.tsv"
    json_path = output_dir / "fixed_prompt_scores.json"
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "category", "score", "notes", "completion_chars"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "sample_file": str(path),
        "prompt_count": len(rows),
        "mean_score": round(mean_score, 4),
        "empty_count": empty_count,
        "scores_tsv": str(tsv_path),
        "rows": rows,
        "note": "Heuristic smoke score for fixed prompt regressions; not a substitute for a real benchmark.",
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Score fixed prompt sample files with lightweight heuristics.")
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_RUNS_ROOT)
    args = parser.parse_args()

    base = args.path if args.path.is_absolute() else ROOT / args.path
    summaries = []
    for sample_file in iter_sample_files(base):
        summaries.append(score_file(sample_file))
        print(f"{sample_file}\t{summaries[-1]['mean_score']:.4f}\t{summaries[-1]['empty_count']}")
    if not summaries:
        raise SystemExit(f"no fixed_prompts.jsonl found under {base}")


if __name__ == "__main__":
    main()
