"""Compare two candidate MCQ bundles with paired statistical tests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from python_starter.core.data_contract import sha256_file, write_json_atomic  # noqa: E402
from python_starter.core.evaluation import compare_mcq_items  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    comparison = compare_mcq_items(
        baseline["metrics"]["mcq"]["items"],
        candidate["metrics"]["mcq"]["items"],
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    comparison["baseline_sha256"] = sha256_file(args.baseline)
    comparison["candidate_sha256"] = sha256_file(args.candidate)
    write_json_atomic(args.output, comparison)
    print(json.dumps(comparison, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
