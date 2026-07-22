"""Register an internally passing model candidate in the local registry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from python_starter.experiments.registry import LocalCandidateRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--training-summary", type=Path)
    parser.add_argument(
        "--preflight",
        type=Path,
        help="Preflight record; binds the source commit to the candidate",
    )
    parser.add_argument(
        "--verdict",
        type=Path,
        help=(
            "Gate verdict that re-judges --evaluation under a later gate; "
            "without it the gate embedded in the evaluation decides"
        ),
    )
    parser.add_argument(
        "--limitation",
        action="append",
        default=[],
        dest="limitations",
        help="Known limitation of this candidate; repeatable, recorded verbatim",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("artifacts/model_registry.json"),
    )
    args = parser.parse_args()
    record = LocalCandidateRegistry(args.registry).register(
        name=args.name,
        checkpoint_path=args.checkpoint,
        evaluation_path=args.evaluation,
        training_summary_path=args.training_summary,
        preflight_path=args.preflight,
        verdict_path=args.verdict,
        limitations=args.limitations,
    )
    print(json.dumps(record, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
