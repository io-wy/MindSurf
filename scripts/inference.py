"""Local inference CLI script.

Usage:
    uv run scripts/inference.py --checkpoint models/checkpoints/pretrain/final_model.pt \
        --prompt "Hello world" --max-length 128
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from python_starter.core.model import ModelConfig, TransformerLM
from python_starter.core.tokenizer import decode_tokens, encode_text, load_tokenizer
from python_starter.core.utils import get_device
from python_starter.infrastructure.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run model inference")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--prompt", required=True, help="Input prompt text")
    parser.add_argument("--max-length", type=int, default=128, help="Max output length")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling p")
    parser.add_argument("--tokenizer", default="gpt2", help="Tokenizer name")
    parser.add_argument("--device", default="auto", help="Device (auto/cpu/cuda)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)

    # Load tokenizer
    tokenizer = load_tokenizer(args.tokenizer)

    # Load model
    logger.info("loading_checkpoint", path=args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)

    # Infer model config from checkpoint if possible, else use default
    model_cfg = ModelConfig()  # defaults
    model = TransformerLM(model_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Encode prompt
    input_ids = encode_text(tokenizer, args.prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    # Generate
    logger.info("generating", prompt=args.prompt[:50])
    start = time.time()
    with torch.no_grad():
        output = model.generate(
            input_tensor,
            max_new_tokens=args.max_length,
            temperature=args.temperature,
            top_p=args.top_p,
            eos_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - start

    # Decode
    output_ids = output[0].tolist()
    generated_text = decode_tokens(tokenizer, output_ids, skip_special_tokens=True)
    response = generated_text[len(args.prompt):].strip()

    print(f"\nPrompt: {args.prompt}")
    print(f"Response: {response}")
    print(f"\nTime: {elapsed:.2f}s | Tokens: {len(output_ids)}")


if __name__ == "__main__":
    main()
