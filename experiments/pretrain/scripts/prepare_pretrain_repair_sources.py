import argparse
import json
import random
from pathlib import Path


REPAIRS = [
    (
        "请用三句话介绍你自己，并说明你能帮我做什么。",
        "我是一个轻量级中文语言模型助手，可以帮助你整理资料、解释概念、写代码和检查思路。我不知道的问题会直接说明不确定，不会故意编造。你可以把问题、上下文和目标告诉我，我会给出尽量清晰可执行的回答。",
    ),
    (
        "请用高中生能听懂的话解释什么是机器学习。",
        "机器学习就是让计算机从大量例子里总结规律，而不是每一步都靠人写死规则。比如给它很多猫和狗的图片，它会学习哪些特征常常对应猫或狗。学好以后，看到新图片时，它就能根据这些规律做判断。",
    ),
    (
        "南京邮电大学位于哪个城市？请只回答事实，不要编造。",
        "南京邮电大学位于江苏省南京市。",
    ),
    (
        "如果一个数的三倍加上 5 等于 29，这个数是多少？请写出计算步骤。",
        "设这个数为 x。根据题意有 3x + 5 = 29，移项得到 3x = 24，所以 x = 8。这个数是 8。",
    ),
    (
        "请写一个 Python 函数，输入 n，返回斐波那契数列的第 n 项。",
        "可以用迭代写法：def fib(n): 如果 n 小于 0 就报错；从 a=0, b=1 开始循环 n 次，每次更新为 a=b, b=a+b；最后返回 a。这样时间复杂度是 O(n)，不会像递归版本那样重复计算。",
    ),
    (
        "比较 MHA、GQA、MQA 在推理速度和模型质量上的取舍。",
        "MHA 每个注意力头都有独立的 K/V，表达能力强，但 KV cache 最大，推理最慢。GQA 让多组 Q 头共享 K/V，是质量和推理速度之间的折中。MQA 所有 Q 头共享一组 K/V，KV cache 最小、推理最快，但模型质量通常更容易下降。",
    ),
    (
        "阅读下面的信息并回答问题：小明周一买了苹果，周二买了香蕉，周三买了橙子。问题：小明周二买了什么？",
        "小明周二买了香蕉。",
    ),
    (
        "Explain in simple English why the sky looks blue.",
        "The sky looks blue because air scatters blue light from the Sun more than red light. That scattered blue light comes from many directions, so we see a blue sky.",
    ),
    (
        "如果你不知道一个问题的答案，应该怎么回答？请给出一个简短示例。",
        "应该明确说明不确定，并说出还需要什么信息。示例：我不确定这个结论是否正确，需要查看最新官方资料后才能确认。",
    ),
    (
        "请连续写五个不同的中文短句，每句表达一个不同的学习建议。",
        "先把今天最重要的一件事写下来。学新概念时，先用自己的话复述一遍。做题后要记录错因，而不只是记录答案。每隔一段时间回顾旧知识，避免遗忘。遇到难题先拆成小步骤，再逐步解决。",
    ),
]


TEMPLATES = [
    "问题：{question}\n答案：{answer}",
    "用户提问：{question}\n可靠回答：{answer}",
    "请根据事实和题意作答。\n问：{question}\n答：{answer}",
    "Instruction: {question}\nResponse: {answer}",
]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_mix(path: Path, strict_train: Path, repair_path: Path, strict_weight: float, repair_weight: float) -> None:
    payload = {
        "description": "Targeted badcase repair mix for pretraining continuation. Sources are sampled by weight.",
        "sources": [
            {"name": "strict_train", "path": str(strict_train), "weight": strict_weight},
            {"name": repair_path.stem, "path": str(repair_path), "weight": repair_weight},
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare targeted fixed-prompt repair text for pretraining continuation.")
    parser.add_argument("--strict_train_path", type=Path, default=Path("experiments/pretrain/strict_splits/pretrain_strict_train.jsonl"))
    parser.add_argument("--output_dir", type=Path, default=Path("experiments/pretrain/flywheel_sources"))
    parser.add_argument("--repeat", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = []
    for _ in range(args.repeat):
        for question, answer in REPAIRS:
            template = rng.choice(TEMPLATES)
            rows.append({"text": template.format(question=question, answer=answer)})
    rng.shuffle(rows)

    repair_path = args.output_dir / f"badcase_repair_as_pretrain_{len(rows)}.jsonl"
    write_jsonl(repair_path, rows)
    write_mix(args.output_dir / "mix_strict99_repair1.json", args.strict_train_path, repair_path, 0.99, 0.01)
    write_mix(args.output_dir / "mix_strict95_repair5.json", args.strict_train_path, repair_path, 0.95, 0.05)

    manifest = {
        "repair_path": str(repair_path),
        "rows": len(rows),
        "base_cases": len(REPAIRS),
        "repeat": args.repeat,
        "mixes": [
            str(args.output_dir / "mix_strict99_repair1.json"),
            str(args.output_dir / "mix_strict95_repair5.json"),
        ],
    }
    manifest_path = args.output_dir / "pretrain_repair_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
