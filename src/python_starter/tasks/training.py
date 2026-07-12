"""Celery tasks for asynchronous training jobs.

Long-running training tasks are offloaded to Celery workers
to avoid blocking the FastAPI request handler.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from celery import shared_task

from python_starter.core.dataset import TextDataset, collate_fn
from python_starter.core.model import ModelConfig, TransformerLM
from python_starter.core.tokenizer import load_tokenizer
from python_starter.core.trainer import Trainer, TrainerConfig
from python_starter.experiments.tracker import ExperimentTracker
from python_starter.infrastructure.config import get_settings
from python_starter.infrastructure.logging import get_logger
from python_starter.tasks.celery_app import celery_app

logger = get_logger(__name__)


@celery_app.task(bind=True, max_retries=3)
def run_training_task(
    self,
    experiment_name: str,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    data_config: dict[str, Any],
) -> dict[str, Any]:
    """Run a training job asynchronously.

    Args:
        experiment_name: Name for the experiment.
        model_config: Model hyperparameters.
        training_config: Training hyperparameters.
        data_config: Data paths and preprocessing config.

    Returns:
        Dict with training results and artifact paths.
    """
    settings = get_settings()
    self.update_state(state="STARTED", meta={"experiment": experiment_name})

    tracker = ExperimentTracker(settings, experiment_name=experiment_name)
    tracker.start(run_name=experiment_name, config={**model_config, **training_config})

    try:
        # Load tokenizer
        tokenizer = load_tokenizer(data_config.get("tokenizer_name", "gpt2"))

        # Build model
        model_cfg = ModelConfig(**model_config)
        model = TransformerLM(model_cfg)

        # Build datasets
        train_dataset = TextDataset(
            data_config["train_path"],
            tokenizer,
            max_length=data_config.get("max_length", 512),
            stride=data_config.get("stride", 512),
        )
        val_dataset = None
        if data_config.get("val_path") and Path(data_config["val_path"]).exists():
            val_dataset = TextDataset(
                data_config["val_path"],
                tokenizer,
                max_length=data_config.get("max_length", 512),
                stride=data_config.get("stride", 512),
            )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=training_config["batch_size"],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
        )
        val_loader = None
        if val_dataset:
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=training_config["batch_size"],
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=0,
            )

        # Train
        trainer_cfg = TrainerConfig(**training_config)
        trainer = Trainer(model, trainer_cfg, tracker=tracker)
        trainer.train(train_loader, val_loader)

        # Log final artifacts
        final_ckpt = trainer.config.output_dir / "final_model.pt"
        if final_ckpt.exists():
            tracker.log_artifact(str(final_ckpt), artifact_path="checkpoints")

        tracker.finish()

        return {
            "status": "completed",
            "experiment": experiment_name,
            "checkpoint_path": str(final_ckpt),
            "best_eval_loss": trainer.best_eval_loss,
            "total_steps": trainer.global_step,
        }

    except Exception as exc:
        logger.error("training_task_failed", error=str(exc))
        tracker.finish()
        self.retry(countdown=60, exc=exc)
        raise
