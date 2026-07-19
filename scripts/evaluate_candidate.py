"""Evaluate strict loss, domains, MCQ, fixed prompts, and repetition gates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from python_starter.core.evaluation import run_candidate_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("data/raw/mindsurf_team_v1/tokenizer"),
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=Path("data/raw/mindsurf_team_v1/strict_splits/pretrain_strict_val_2k.jsonl"),
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=Path("data/raw/mindsurf_team_v1/strict_splits/pretrain_strict_test_2k.jsonl"),
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("artifacts/data/mindsurf_team_v1/audit.json"),
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=Path("configs/evaluation/candidate_thresholds_80m.json"),
    )
    parser.add_argument(
        "--mcq",
        type=Path,
        default=Path("configs/evaluation/local_mcq_benchmark_v1.jsonl"),
    )
    parser.add_argument(
        "--fixed-prompts",
        type=Path,
        default=Path("configs/evaluation/fixed_prompts.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/evaluation/mindsurf_team_v1_80m.json"),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--strict-batches", type=int, default=250)
    parser.add_argument("--domain-blocks", type=int, default=250)
    parser.add_argument("--fixed-new-tokens", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    result = run_candidate_evaluation(
        root=root,
        checkpoint_path=args.checkpoint.resolve(),
        tokenizer_path=args.tokenizer.resolve(),
        validation_path=args.validation.resolve(),
        test_path=args.test.resolve(),
        audit_path=args.audit.resolve(),
        thresholds_path=args.thresholds.resolve(),
        mcq_path=args.mcq.resolve(),
        fixed_prompts_path=args.fixed_prompts.resolve(),
        output_path=args.output.resolve(),
        device_name=args.device,
        max_length=args.max_length,
        batch_size=args.batch_size,
        strict_batches=args.strict_batches,
        domain_blocks=args.domain_blocks,
        fixed_new_tokens=args.fixed_new_tokens,
    )
    print(json.dumps(result["gate"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
