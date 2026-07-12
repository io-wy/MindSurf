"""Model evaluation script.

Usage:
    uv run scripts/evaluate.py --checkpoint models/checkpoints/pretrain/final_model.pt \
        --data data/raw/test.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from python_starter.core.dataset import TextDataset, collate_fn
from python_starter.core.model import ModelConfig, TransformerLM
from python_starter.core.tokenizer import load_tokenizer
from python_starter.core.trainer import Trainer, TrainerConfig
from python_starter.core.utils import get_device
from python_starter.infrastructure.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--data", required=True, help="Path to evaluation data")
    parser.add_argument("--tokenizer", default="gpt2", help="Tokenizer name")
    parser.add_argument("--batch-size", type=int, default=4, help="Evaluation batch size")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--device", default="auto", help="Device (auto/cpu/cuda)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)

    tokenizer = load_tokenizer(args.tokenizer)

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model_cfg = ModelConfig()
    model = TransformerLM(model_cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # Build eval dataset
    eval_dataset = TextDataset(
        args.data,
        tokenizer,
        max_length=args.max_length,
        stride=args.max_length,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Evaluate
    trainer_cfg = TrainerConfig(device=str(device))
    trainer = Trainer(model, trainer_cfg)
    eval_loss = trainer.evaluate(eval_loader)

    perplexity = torch.exp(torch.tensor(eval_loss)).item()

    logger.info("evaluation_complete", loss=eval_loss, perplexity=perplexity)
    print(f"\nEvaluation Results:")
    print(f"  Loss:       {eval_loss:.4f}")
    print(f"  Perplexity: {perplexity:.4f}")


if __name__ == "__main__":
    main()
