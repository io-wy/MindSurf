"""Generic training loop for language model pretraining and fine-tuning.

Supports:
- Pretraining (next-token prediction)
- Supervised Fine-Tuning (SFT)
- Checkpoint saving/loading
- LR scheduling (cosine with warmup)
- Gradient clipping
- Mixed precision training (AMP)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from python_starter.core.model import TransformerLM
from python_starter.core.utils import format_number, get_logger
from python_starter.experiments.tracker import ExperimentTracker

logger = get_logger(__name__)


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Create a cosine learning rate scheduler with linear warmup."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
            1.0 + torch.cos(torch.tensor(progress * 3.14159265)).item()
        )

    return LambdaLR(optimizer, lr_lambda)


class TrainerConfig:
    """Training hyperparameters."""

    def __init__(
        self,
        output_dir: str = "models/checkpoints",
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 5e-4,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        warmup_ratio: float = 0.1,
        eval_every: int = 500,
        save_every: int = 1000,
        logging_every: int = 10,
        device: str = "auto",
        dtype: str = "float32",
        compile_model: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.warmup_ratio = warmup_ratio
        self.eval_every = eval_every
        self.save_every = save_every
        self.logging_every = logging_every
        self.device = device
        self.dtype = dtype
        self.compile_model = compile_model


class Trainer:
    """Generic trainer for TransformerLM."""

    def __init__(
        self,
        model: TransformerLM,
        config: TrainerConfig,
        tracker: ExperimentTracker | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.tracker = tracker
        self.global_step = 0
        self.best_eval_loss = float("inf")

        self.device = self._resolve_device()
        self.dtype = self._resolve_dtype()
        self.model.to(self.device)

        if config.compile_model and hasattr(torch, "compile"):
            logger.info("compiling_model")
            self.model = torch.compile(self.model)  # type: ignore[assignment]

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "trainer_initialized",
            device=str(self.device),
            dtype=str(self.dtype),
            params=format_number(sum(p.numel() for p in model.parameters())),
        )

    def _resolve_device(self) -> torch.device:
        from python_starter.core.utils import get_device
        return get_device(self.config.device)

    def _resolve_dtype(self) -> torch.dtype:
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(self.config.dtype, torch.float32)

    def _create_optimizer(self) -> AdamW:
        no_decay = ["bias", "norm", "embed"]
        params = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return AdamW(params, lr=self.config.learning_rate)

    def save_checkpoint(self, filename: str | None = None, **extra: Any) -> None:
        """Save model checkpoint."""
        if filename is None:
            filename = f"checkpoint_step_{self.global_step}.pt"
        path = self.config.output_dir / filename

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "global_step": self.global_step,
            "best_eval_loss": self.best_eval_loss,
            **extra,
        }
        torch.save(checkpoint, path)
        logger.info("checkpoint_saved", path=str(path))

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        """Load model checkpoint."""
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.global_step = checkpoint.get("global_step", 0)
        self.best_eval_loss = checkpoint.get("best_eval_loss", float("inf"))
        logger.info("checkpoint_loaded", path=str(path), step=self.global_step)
        return checkpoint

    def train(
        self,
        train_loader: DataLoader,
        eval_loader: DataLoader | None = None,
    ) -> None:
        """Run the training loop."""
        optimizer = self._create_optimizer()
        total_steps = len(train_loader) * self.config.num_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, warmup_steps, total_steps
        )

        if self.tracker:
            self.tracker.log_params({
                "epochs": self.config.num_epochs,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "total_steps": total_steps,
            })

        scaler = None
        if self.dtype == torch.float16 and self.device.type == "cuda":
            scaler = torch.amp.GradScaler(device="cuda")
        self.model.train()

        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.config.num_epochs}")

            for batch in pbar:
                loss = self._train_step(batch, optimizer, scaler)
                scheduler.step()
                self.global_step += 1
                epoch_loss += loss

                pbar.set_postfix({"loss": f"{loss:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

                if self.global_step % self.config.logging_every == 0 and self.tracker:
                    self.tracker.log_metrics({
                        "train/loss": loss,
                        "train/lr": scheduler.get_last_lr()[0],
                    }, step=self.global_step)

                if self.global_step % self.config.save_every == 0:
                    self.save_checkpoint(optimizer_state_dict=optimizer.state_dict())

                if eval_loader and self.global_step % self.config.eval_every == 0:
                    eval_loss = self.evaluate(eval_loader)
                    if eval_loss < self.best_eval_loss:
                        self.best_eval_loss = eval_loss
                        self.save_checkpoint("best_model.pt", optimizer_state_dict=optimizer.state_dict())
                    self.model.train()

            avg_loss = epoch_loss / len(train_loader)
            logger.info("epoch_complete", epoch=epoch + 1, avg_loss=avg_loss)

        # Final save
        self.save_checkpoint("final_model.pt", optimizer_state_dict=optimizer.state_dict())

    def _train_step(
        self,
        batch: dict[str, torch.Tensor],
        optimizer: AdamW,
        scaler: torch.cuda.amp.GradScaler | None,
    ) -> float:
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        optimizer.zero_grad()

        if scaler:
            with torch.amp.autocast(device_type=str(self.device).split(":")[0], dtype=self.dtype):
                _, loss = self.model(input_ids, labels)
            scaler.scale(loss).backward()  # type: ignore[arg-type]
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            _, loss = self.model(input_ids, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            optimizer.step()

        return loss.item()

    @torch.no_grad()
    def evaluate(self, eval_loader: DataLoader) -> float:
        """Run evaluation and return average loss."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            _, loss = self.model(input_ids, labels)
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        logger.info("evaluation_complete", eval_loss=avg_loss)

        if self.tracker:
            self.tracker.log_metrics({"eval/loss": avg_loss}, step=self.global_step)

        return avg_loss
