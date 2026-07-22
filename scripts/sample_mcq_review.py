"""Draw a reproducible review sample from the frozen MCQ benchmark.

The audit records a maintainer review of answer correctness, ambiguity, and
category assignment. That claim is only worth anything if the reviewed items
are pinned, so the sample is drawn from a stated seed, stratified across
categories, and written out with the benchmark digest it came from.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from python_starter.core.data_contract import (  # noqa: E402
    iter_jsonl,
    sha256_file,
    write_json_atomic,
)

LETTERS = "ABCD"


def _stratified_sample(items: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    """Spread the sample evenly over categories, then fill by global draw."""
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_category[str(item["category"])].append(item)

    rng = random.Random(seed)
    categories = sorted(by_category)
    per_category, remainder = divmod(count, len(categories))

    picked: list[dict[str, Any]] = []
    for category in categories:
        pool = sorted(by_category[category], key=lambda row: str(row["id"]))
        picked.extend(rng.sample(pool, min(per_category, len(pool))))

    if remainder:
        chosen = {str(item["id"]) for item in picked}
        rest = sorted(
            (item for item in items if str(item["id"]) not in chosen),
            key=lambda row: str(row["id"]),
        )
        picked.extend(rng.sample(rest, min(remainder, len(rest))))

    return sorted(picked, key=lambda row: (str(row["category"]), str(row["id"])))


def _repo_relative(path: Path) -> str:
    """Absolute paths cannot be checked against a public repository."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.name


def _render(items: list[dict[str, Any]], benchmark: str, digest: str, seed: int) -> str:
    lines = [
        "# MCQ 人工抽检样本",
        "",
        f"- 题库：`{benchmark}`",
        f"- 题库 SHA-256：`{digest}`",
        f"- 抽样种子：`{seed}`",
        f"- 样本量：{len(items)}",
        "",
        "逐题检查三件事：**答案是否正确**、**题干或选项是否有歧义**、**领域归类是否恰当**。",
        "",
    ]
    for index, item in enumerate(items, start=1):
        answer = int(item["answer"])
        lines.append(f"## {index}. `{item['id']}`  —  {item['category']}")
        lines.append("")
        lines.append("```")
        lines.append(str(item["prompt"]).rstrip("\n"))
        lines.append("```")
        lines.append("")
        for position, choice in enumerate(item["choices"]):
            mark = " **← 标注答案**" if position == answer else ""
            lines.append(f"- {LETTERS[position]}. `{str(choice).strip()}`{mark}")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=ROOT / "configs/evaluation/pretrain_mcq_benchmark_v2.jsonl",
    )
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    args = parser.parse_args()

    items = [row for _, row in iter_jsonl(args.benchmark)]
    if args.count > len(items):
        raise SystemExit(f"asked for {args.count} items, benchmark holds {len(items)}")

    digest = sha256_file(args.benchmark)
    sample = _stratified_sample(items, args.count, args.seed)

    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    benchmark_name = _repo_relative(args.benchmark)
    args.markdown.write_text(_render(sample, benchmark_name, digest, args.seed), encoding="utf-8")
    write_json_atomic(
        args.record,
        {
            "schema_version": 1,
            "drawn_at": datetime.now(UTC).isoformat(),
            "benchmark": benchmark_name,
            "benchmark_sha256": digest,
            "seed": args.seed,
            "count": len(sample),
            "sampling": "stratified by category, then global fill",
            "item_ids": [str(item["id"]) for item in sample],
            "review_sheet": _repo_relative(args.markdown),
            "reviewed": False,
        },
    )
    print(f"{len(sample)} items -> {args.markdown}")


if __name__ == "__main__":
    main()
