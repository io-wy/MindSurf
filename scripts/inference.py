"""Run real local inference from an exact-resume checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from python_starter.core.inference import InferenceEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = InferenceEngine(args.checkpoint, args.tokenizer, args.device)
    result = engine.generate(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(result.text)
    print(
        f"input_tokens={result.input_tokens} "
        f"output_tokens={result.output_tokens} "
        f"generation_time_ms={result.generation_time_ms:.2f}"
    )


if __name__ == "__main__":
    main()
