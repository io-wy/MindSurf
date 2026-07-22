"""Inference CLI for merged or PEFT post-training checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from python_starter.core.utils import get_device
from python_starter.post_training.inference import (
    generate_response,
    load_post_training_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a post-trained model")
    parser.add_argument("--model", required=True, help="Merged model or base model path")
    parser.add_argument("--adapter", help="Optional PEFT adapter path")
    parser.add_argument("--prompt", required=True, help="Prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda/mps")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = load_post_training_model(
        model_name_or_path=args.model,
        adapter_name_or_path=args.adapter,
        device=get_device(args.device),
        trust_remote_code=args.trust_remote_code,
    )
    response = generate_response(
        runtime,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(response)


if __name__ == "__main__":
    main()
