"""Data preprocessing script.

Usage:
    uv run scripts/preprocess.py --input data/raw/corpus.txt --output data/processed/train.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from python_starter.infrastructure.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess raw data")
    parser.add_argument("--input", required=True, help="Input raw data file or directory")
    parser.add_argument("--output", required=True, help="Output processed file")
    parser.add_argument("--min-length", type=int, default=10, help="Minimum line length")
    parser.add_argument("--max-length", type=int, default=10000, help="Maximum line length")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect input texts
    texts: list[str] = []
    if input_path.is_file():
        texts = input_path.read_text(encoding="utf-8").splitlines()
    else:
        for file in sorted(input_path.glob("*.txt")):
            texts.extend(file.read_text(encoding="utf-8").splitlines())

    # Filter and clean
    processed: list[str] = []
    for line in texts:
        line = line.strip()
        if args.min_length <= len(line) <= args.max_length:
            processed.append(line)

    # Write output
    output_path.write_text("\n".join(processed), encoding="utf-8")

    logger.info(
        "preprocessing_complete",
        input_lines=len(texts),
        output_lines=len(processed),
        output_path=str(output_path),
    )
    print(f"Processed {len(texts)} lines → {len(processed)} lines saved to {output_path}")


if __name__ == "__main__":
    main()
