"""Unified training entry point driven by Hydra configuration.

Usage:
    uv run scripts/train.py training=pretrain model=minimind
    uv run scripts/train.py training=sft model=minimind_small data=default
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

# Allow importing src/python_starter as top-level package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from python_starter.core.dataset import SFTDataset, TextDataset, collate_fn
from python_starter.core.model import ModelConfig, TransformerLM
from python_starter.core.tokenizer import load_tokenizer
from python_starter.core.trainer import Trainer, TrainerConfig
from python_starter.core.utils import count_parameters, format_number, set_seed
from python_starter.experiments.tracker import ExperimentTracker
from python_starter.infrastructure.config import get_settings
from python_starter.infrastructure.logging import get_logger

logger = get_logger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    """Run training with Hydra configuration."""
    settings = get_settings()

    # Print resolved config
    logger.info("training_config", config=OmegaConf.to_container(cfg, resolve=True))

    # Set seed
    seed = cfg.get("seed", 42)
    set_seed(seed)

    # Load tokenizer
    tokenizer = load_tokenizer(cfg.data.tokenizer_name)

    # Build model
    model_cfg = ModelConfig(**cfg.model)
    model = TransformerLM(model_cfg)
    logger.info(
        "model_built",
        params=format_number(count_parameters(model)),
        config=cfg.model,
    )

    # Build datasets
    dataset_type = cfg.get("dataset_type", "pretrain")
    if dataset_type == "sft":
        train_dataset = SFTDataset(
            cfg.data.train_path,
            tokenizer,
            max_length=cfg.data.max_length,
        )
        val_dataset = None
        if cfg.data.get("val_path") and Path(cfg.data.val_path).exists():
            val_dataset = SFTDataset(
                cfg.data.val_path,
                tokenizer,
                max_length=cfg.data.max_length,
            )
    else:
        train_dataset = TextDataset(
            cfg.data.train_path,
            tokenizer,
            max_length=cfg.data.max_length,
            stride=cfg.data.get("stride", cfg.data.max_length),
        )
        val_dataset = None
        if cfg.data.get("val_path") and Path(cfg.data.val_path).exists():
            val_dataset = TextDataset(
                cfg.data.val_path,
                tokenizer,
                max_length=cfg.data.max_length,
                stride=cfg.data.get("stride", cfg.data.max_length),
            )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = None
    if val_dataset:
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )

    # Initialize experiment tracker
    tracker = ExperimentTracker(settings, experiment_name=cfg.get("experiment_name", "default"))
    tracker.start(
        run_name=cfg.get("run_name", None),
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # Train
    trainer_cfg = TrainerConfig(**cfg.training)
    trainer = Trainer(model, trainer_cfg, tracker=tracker)
    trainer.train(train_loader, val_loader)

    # Cleanup
    tracker.finish()
    logger.info("training_finished")


if __name__ == "__main__":
    main()
