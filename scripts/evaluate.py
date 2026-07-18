"""Evaluate packed JSONL next-token loss from an embedded-config checkpoint."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from python_starter.core.evaluation import packed_loss
from python_starter.core.inference import load_checkpoint_model
from python_starter.core.tokenizer import load_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--max-batches", type=int, default=250)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    model, _, device = load_checkpoint_model(args.checkpoint, args.device)
    tokenizer = load_tokenizer(str(args.tokenizer))
    result = packed_loss(
        model,
        tokenizer,
        args.data,
        device,
        max_length=args.max_length,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
    )
    print(
        f"loss={float(result['loss']):.6f} "
        f"perplexity={math.exp(min(float(result['loss']), 20)):.6f} "
        f"tokens={result['tokens']} batches={result['batches']}"
    )


if __name__ == "__main__":
    main()
