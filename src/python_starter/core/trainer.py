"""Deterministic single-process language-model training.

The optimizer-step count, packed-block cursor, WSD schedule, and all mutable
training state are checkpointed together so an interrupted run can resume at
the same optimizer boundary.
"""

from __future__ import annotations

import json
import os
import random
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from python_starter.core.model import TransformerLM
from python_starter.core.utils import format_number
from python_starter.experiments.tracker import ExperimentTracker
from python_starter.infrastructure.logging import get_logger

logger = get_logger(__name__)
CHECKPOINT_SCHEMA_VERSION = 2


def get_wsd_schedule(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    stable_ratio: float = 0.8,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Create a linear-warmup, stable, cosine-decay schedule."""
    if num_training_steps <= 0:
        raise ValueError("num_training_steps must be positive")
    if not 0 <= num_warmup_steps < num_training_steps:
        raise ValueError("num_warmup_steps must be in [0, num_training_steps)")
    if not 0 <= stable_ratio <= 1:
        raise ValueError("stable_ratio must be in [0, 1]")
    if not 0 <= min_lr_ratio <= 1:
        raise ValueError("min_lr_ratio must be in [0, 1]")

    post_warmup_steps = num_training_steps - num_warmup_steps
    stable_steps = int(post_warmup_steps * stable_ratio)
    decay_start = num_warmup_steps + stable_steps

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step + 1) / float(max(1, num_warmup_steps))
        if current_step < decay_start:
            return 1.0
        decay_steps = max(1, num_training_steps - decay_start)
        progress = min(1.0, float(current_step - decay_start) / float(decay_steps))
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * float(cosine)

    return LambdaLR(optimizer, lr_lambda)


@dataclass(slots=True)
class TrainerConfig:
    """Fixed-step single-GPU training hyperparameters."""

    output_dir: Path = Path("models/checkpoints")
    max_steps: int = 10_000
    batch_size: int = 4
    accumulation_steps: int = 1
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    stable_ratio: float = 0.8
    min_lr_ratio: float = 0.1
    eval_every: int = 500
    eval_batches: int = 50
    save_every: int = 1_000
    checkpoint_keep_last: int = 2
    logging_every: int = 10
    device: str = "auto"
    dtype: str = "float32"
    compile_model: bool = False

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.accumulation_steps <= 0:
            raise ValueError("accumulation_steps must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.save_every <= 0 or self.logging_every <= 0:
            raise ValueError("save_every and logging_every must be positive")
        if self.checkpoint_keep_last <= 0:
            raise ValueError("checkpoint_keep_last must be positive")

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["output_dir"] = str(self.output_dir)
        return values


class Trainer:
    """Fixed-step trainer with atomic, exact-resume checkpoints."""

    def __init__(
        self,
        model: TransformerLM,
        config: TrainerConfig,
        tracker: ExperimentTracker | None = None,
        run_config: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.config = config
        self.tracker = tracker
        self.run_config = run_config or {}
        self.global_step = 0
        self.micro_step = 0
        self.consumed_blocks = 0
        self.consumed_tokens = 0
        self.best_eval_loss = float("inf")
        self.last_train_loss: float | None = None
        self.training_elapsed_seconds = 0.0
        self.peak_cuda_allocated_bytes = 0
        self.peak_cuda_reserved_bytes = 0
        self.data_wait_seconds = 0.0
        self.optimizer_step_seconds = 0.0
        self.checkpoint_write_seconds = 0.0
        self.last_gradient_norm: float | None = None
        self._session_started_at: float | None = None
        self._resume_checkpoint: dict[str, Any] | None = None
        self._optimizer: AdamW | None = None
        self._scheduler: LambdaLR | None = None
        self._scaler: torch.cuda.amp.GradScaler | None = None

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
        try:
            return dtype_map[self.config.dtype]
        except KeyError as exc:
            raise ValueError(f"unsupported dtype: {self.config.dtype}") from exc

    def _create_optimizer(self) -> AdamW:
        no_decay = ("bias", "norm", "embed")
        decay_params: list[torch.nn.Parameter] = []
        no_decay_params: list[torch.nn.Parameter] = []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            target = no_decay_params if any(item in name for item in no_decay) else decay_params
            target.append(parameter)
        return AdamW(
            [
                {"params": decay_params, "weight_decay": self.config.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.config.learning_rate,
        )

    def _model_for_state(self) -> TransformerLM:
        original = getattr(self.model, "_orig_mod", None)
        return original if original is not None else self.model

    def _rng_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    def _restore_rng_state(self, state: dict[str, Any]) -> None:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and "cuda" in state:
            torch.cuda.set_rng_state_all(state["cuda"])

    def save_checkpoint(self, filename: str | None = None, **extra: Any) -> Path:
        """Atomically save model, optimizer, scheduler, RNG, and data cursor."""
        if filename is None:
            filename = f"checkpoint_step_{self.global_step}.pt"
        path = self.config.output_dir / filename
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")

        checkpoint: dict[str, Any] = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model_state_dict": self._model_for_state().state_dict(),
            "model_config": self._model_for_state().config.to_dict(),
            "trainer_config": self.config.to_dict(),
            "run_config": self.run_config,
            "progress": {
                "global_step": self.global_step,
                "micro_step": self.micro_step,
                "consumed_blocks": self.consumed_blocks,
                "consumed_tokens": self.consumed_tokens,
                "best_eval_loss": self.best_eval_loss,
                "last_train_loss": self.last_train_loss,
                "training_elapsed_seconds": self._current_elapsed_seconds(),
                "peak_cuda_allocated_bytes": self._peak_cuda_allocated_bytes(),
                "peak_cuda_reserved_bytes": self._peak_cuda_reserved_bytes(),
                "data_wait_seconds": self.data_wait_seconds,
                "optimizer_step_seconds": self.optimizer_step_seconds,
                "checkpoint_write_seconds": self.checkpoint_write_seconds,
                "last_gradient_norm": self.last_gradient_norm,
            },
            "rng_state": self._rng_state(),
        }
        if self._optimizer is not None:
            checkpoint["optimizer_state_dict"] = self._optimizer.state_dict()
        if self._scheduler is not None:
            checkpoint["scheduler_state_dict"] = self._scheduler.state_dict()  # type: ignore[no-untyped-call]
        if self._scaler is not None:
            checkpoint["scaler_state_dict"] = self._scaler.state_dict()
        checkpoint.update(extra)

        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with temporary.open("wb") as handle:
                torch.save(checkpoint, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        logger.info("checkpoint_saved", path=str(path), step=self.global_step)
        if filename.startswith("checkpoint_step_"):
            self._prune_step_checkpoints()
        return path

    def _prune_step_checkpoints(self) -> None:
        checkpoints = sorted(
            self.config.output_dir.glob("checkpoint_step_*.pt"),
            key=lambda candidate: int(candidate.stem.rsplit("_", 1)[-1]),
        )
        for stale in checkpoints[: -self.config.checkpoint_keep_last]:
            stale.unlink()
            logger.info("checkpoint_pruned", path=str(stale))

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        """Stage a trusted checkpoint and restore its model/progress state."""
        path = Path(path)
        checkpoint = cast(
            dict[str, Any],
            torch.load(path, map_location="cpu", weights_only=False),
        )
        schema_version = checkpoint.get("schema_version", 1)
        if schema_version > CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(f"unsupported checkpoint schema: {schema_version}")

        checkpoint_model_config = checkpoint.get("model_config")
        if checkpoint_model_config is not None:
            actual_config = self._model_for_state().config.to_dict()
            if checkpoint_model_config != actual_config:
                raise ValueError("checkpoint model configuration does not match the model")

        self._model_for_state().load_state_dict(checkpoint["model_state_dict"])
        progress = checkpoint.get("progress", {})
        self.global_step = progress.get("global_step", checkpoint.get("global_step", 0))
        self.micro_step = progress.get(
            "micro_step", self.global_step * self.config.accumulation_steps
        )
        self.consumed_blocks = progress.get("consumed_blocks", 0)
        self.consumed_tokens = progress.get("consumed_tokens", 0)
        self.best_eval_loss = progress.get(
            "best_eval_loss", checkpoint.get("best_eval_loss", float("inf"))
        )
        self.last_train_loss = progress.get("last_train_loss")
        self.training_elapsed_seconds = progress.get("training_elapsed_seconds", 0.0)
        self.peak_cuda_allocated_bytes = progress.get("peak_cuda_allocated_bytes", 0)
        self.peak_cuda_reserved_bytes = progress.get("peak_cuda_reserved_bytes", 0)
        self.data_wait_seconds = progress.get("data_wait_seconds", 0.0)
        self.optimizer_step_seconds = progress.get("optimizer_step_seconds", 0.0)
        self.checkpoint_write_seconds = progress.get("checkpoint_write_seconds", 0.0)
        self.last_gradient_norm = progress.get("last_gradient_norm")
        self._resume_checkpoint = checkpoint
        logger.info(
            "checkpoint_loaded",
            path=str(path),
            step=self.global_step,
            consumed_blocks=self.consumed_blocks,
        )
        return checkpoint

    def load_model_weights(self, path: str | Path) -> dict[str, Any]:
        """Initialize a new run from trusted model weights without resuming mutable state."""
        path = Path(path)
        checkpoint = cast(
            dict[str, Any],
            torch.load(path, map_location="cpu", weights_only=False),
        )
        checkpoint_model_config = checkpoint.get("model_config")
        if checkpoint_model_config is not None:
            actual_config = self._model_for_state().config.to_dict()
            if checkpoint_model_config != actual_config:
                raise ValueError("checkpoint model configuration does not match the model")
        self._model_for_state().load_state_dict(checkpoint["model_state_dict"])
        logger.info("model_weights_initialized", path=str(path))
        return checkpoint

    def _restore_mutable_training_state(self) -> None:
        if self._resume_checkpoint is None:
            return
        checkpoint = self._resume_checkpoint
        if self._optimizer is not None and "optimizer_state_dict" in checkpoint:
            self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self._scheduler is not None and "scheduler_state_dict" in checkpoint:
            self._scheduler.load_state_dict(checkpoint["scheduler_state_dict"])  # type: ignore[no-untyped-call]
        if self._scaler is not None and "scaler_state_dict" in checkpoint:
            self._scaler.load_state_dict(checkpoint["scaler_state_dict"])
        if "rng_state" in checkpoint:
            self._restore_rng_state(checkpoint["rng_state"])
        self._resume_checkpoint = None

    def train(
        self,
        train_loader: DataLoader[dict[str, torch.Tensor]],
        eval_loader: DataLoader[dict[str, torch.Tensor]] | None = None,
    ) -> None:
        """Train to the configured optimizer-step target."""
        if train_loader.num_workers != 0:
            raise ValueError("exact resume requires DataLoader(num_workers=0)")

        dataset = train_loader.dataset
        set_skip_blocks = getattr(dataset, "set_skip_blocks", None)
        if callable(set_skip_blocks):
            set_skip_blocks(self.consumed_blocks)
        elif self.consumed_blocks:
            raise ValueError("resuming requires a dataset with set_skip_blocks()")

        self._optimizer = self._create_optimizer()
        self._scheduler = get_wsd_schedule(
            self._optimizer,
            self.config.warmup_steps,
            self.config.max_steps,
            stable_ratio=self.config.stable_ratio,
            min_lr_ratio=self.config.min_lr_ratio,
        )
        if self.dtype == torch.float16 and self.device.type == "cuda":
            self._scaler = torch.cuda.amp.GradScaler()
        self._restore_mutable_training_state()
        self._session_started_at = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        if self.tracker:
            self.tracker.log_params(
                {
                    "batch_size": self.config.batch_size,
                    "accumulation_steps": self.config.accumulation_steps,
                    "learning_rate": self.config.learning_rate,
                    "max_steps": self.config.max_steps,
                    "warmup_steps": self.config.warmup_steps,
                    "stable_ratio": self.config.stable_ratio,
                }
            )

        self.model.train()
        self._optimizer.zero_grad(set_to_none=True)
        progress = tqdm(
            total=self.config.max_steps,
            initial=self.global_step,
            desc="Training",
        )
        iterator = iter(train_loader)

        while self.global_step < self.config.max_steps:
            data_wait_started = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration as exc:
                raise RuntimeError("training data exhausted before reaching max_steps") from exc
            self.data_wait_seconds += time.perf_counter() - data_wait_started

            loss = self._backward_micro_batch(batch)
            self.last_train_loss = loss
            self.micro_step += 1
            self.consumed_blocks += int(batch["input_ids"].shape[0])
            self.consumed_tokens += int(batch["input_ids"].numel())
            if self.micro_step % self.config.accumulation_steps:
                continue

            optimizer_step_started = time.perf_counter()
            self.last_gradient_norm = self._optimizer_step()
            step_seconds = time.perf_counter() - optimizer_step_started
            self.optimizer_step_seconds += step_seconds
            self.global_step += 1
            progress.update(1)
            learning_rate = self._optimizer.param_groups[0]["lr"]
            progress.set_postfix(loss=f"{loss:.4f}", lr=f"{learning_rate:.2e}")

            if self.global_step % self.config.logging_every == 0 and self.tracker:
                self.tracker.log_metrics(
                    {
                        "train/loss": loss,
                        "train/lr": learning_rate,
                        "train/consumed_blocks": float(self.consumed_blocks),
                        "train/consumed_tokens": float(self.consumed_tokens),
                        "train/tokens_per_second": self._tokens_per_second(),
                        "train/data_wait_seconds": self.data_wait_seconds,
                        "train/optimizer_step_seconds": step_seconds,
                        "train/gradient_norm": self.last_gradient_norm,
                        "train/checkpoint_write_seconds": self.checkpoint_write_seconds,
                        "train/peak_cuda_reserved_bytes": float(self._peak_cuda_reserved_bytes()),
                    },
                    step=self.global_step,
                )

            if self.global_step % self.config.save_every == 0:
                checkpoint_started = time.perf_counter()
                self.save_checkpoint()
                self.checkpoint_write_seconds += time.perf_counter() - checkpoint_started

            if (
                eval_loader is not None
                and self.config.eval_every > 0
                and self.global_step % self.config.eval_every == 0
            ):
                eval_loss = self.evaluate(eval_loader, max_batches=self.config.eval_batches)
                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    self.save_checkpoint("best_model.pt")
                self.model.train()

        progress.close()
        self.save_checkpoint("final_model.pt")
        self.training_elapsed_seconds = self._current_elapsed_seconds()
        self.peak_cuda_allocated_bytes = self._peak_cuda_allocated_bytes()
        self.peak_cuda_reserved_bytes = self._peak_cuda_reserved_bytes()
        self._session_started_at = None
        summary = self._write_training_summary()
        if self.tracker:
            self.tracker.log_metrics(
                {
                    "train/final_loss": float(self.last_train_loss or 0.0),
                    "train/tokens_per_second": float(summary["tokens_per_second"]),
                    "train/peak_cuda_allocated_bytes": float(self.peak_cuda_allocated_bytes),
                    "train/peak_cuda_reserved_bytes": float(self.peak_cuda_reserved_bytes),
                },
                step=self.global_step,
            )

    def _current_elapsed_seconds(self) -> float:
        elapsed = self.training_elapsed_seconds
        if self._session_started_at is not None:
            elapsed += time.perf_counter() - self._session_started_at
        return elapsed

    def _peak_cuda_allocated_bytes(self) -> int:
        current = 0
        if self.device.type == "cuda":
            current = torch.cuda.max_memory_allocated(self.device)
        return max(self.peak_cuda_allocated_bytes, current)

    def _peak_cuda_reserved_bytes(self) -> int:
        current = 0
        if self.device.type == "cuda":
            current = torch.cuda.max_memory_reserved(self.device)
        return max(self.peak_cuda_reserved_bytes, current)

    def _tokens_per_second(self) -> float:
        elapsed = self._current_elapsed_seconds()
        return self.consumed_tokens / elapsed if elapsed > 0 else 0.0

    def _write_training_summary(self) -> dict[str, Any]:
        training_view = self.run_config.get("training_view", {})
        identity = {
            "dataset_id": self.run_config.get("data", {}).get("dataset_id"),
            "dataset_revision": self.run_config.get("data", {}).get("dataset_revision"),
            "training_view_sha256": training_view.get("output", {}).get("sha256"),
            "seed": self.run_config.get("seed"),
        }
        summary: dict[str, Any] = {
            "schema_version": 1,
            "global_step": self.global_step,
            "micro_step": self.micro_step,
            "consumed_blocks": self.consumed_blocks,
            "consumed_tokens": self.consumed_tokens,
            "training_elapsed_seconds": self.training_elapsed_seconds,
            "tokens_per_second": self._tokens_per_second(),
            "parameter_count": sum(
                parameter.numel() for parameter in self._model_for_state().parameters()
            ),
            "final_train_loss": self.last_train_loss,
            "best_eval_loss": self.best_eval_loss,
            "peak_cuda_allocated_bytes": self.peak_cuda_allocated_bytes,
            "peak_cuda_reserved_bytes": self.peak_cuda_reserved_bytes,
            "data_wait_seconds": self.data_wait_seconds,
            "optimizer_step_seconds": self.optimizer_step_seconds,
            "checkpoint_write_seconds": self.checkpoint_write_seconds,
            "last_gradient_norm": self.last_gradient_norm,
            "identity": identity,
        }
        path = self.config.output_dir / "training_summary.json"
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        logger.info("training_summary_written", path=str(path), **summary)
        return summary

    def _backward_micro_batch(self, batch: dict[str, torch.Tensor]) -> float:
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        labels = batch["labels"].to(self.device, non_blocking=True)
        autocast_enabled = self.device.type == "cuda" and self.dtype != torch.float32

        with torch.autocast(
            device_type=self.device.type,
            dtype=self.dtype,
            enabled=autocast_enabled,
        ):
            _, loss = self.model(input_ids, labels)
            if loss is None:
                raise RuntimeError("model did not return a training loss")
            scaled_loss = loss / self.config.accumulation_steps

        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss at step {self.global_step}")
        if self._scaler is not None:
            self._scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        return float(loss.detach().item())

    def _optimizer_step(self) -> float:
        if self._optimizer is None or self._scheduler is None:
            raise RuntimeError("optimizer and scheduler must be initialized")
        if self._scaler is not None:
            self._scaler.unscale_(self._optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm,
        )
        if self._scaler is not None:
            self._scaler.step(self._optimizer)
            self._scaler.update()
        else:
            self._optimizer.step()
        self._scheduler.step()
        self._optimizer.zero_grad(set_to_none=True)
        return float(gradient_norm.item())

    @torch.no_grad()
    def evaluate(
        self,
        eval_loader: DataLoader[dict[str, torch.Tensor]],
        max_batches: int | None = None,
    ) -> float:
        """Run bounded validation and return mean next-token loss."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        autocast_enabled = self.device.type == "cuda" and self.dtype != torch.float32

        for batch in tqdm(eval_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(self.device, non_blocking=True)
            labels = batch["labels"].to(self.device, non_blocking=True)
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.dtype,
                enabled=autocast_enabled,
            ):
                _, loss = self.model(input_ids, labels)
            if loss is None:
                raise RuntimeError("model did not return an evaluation loss")
            total_loss += float(loss.item())
            num_batches += 1
            if max_batches is not None and num_batches >= max_batches:
                break

        avg_loss = total_loss / max(num_batches, 1)
        logger.info("evaluation_complete", eval_loss=avg_loss, batches=num_batches)
        if self.tracker:
            self.tracker.log_metrics({"eval/loss": avg_loss}, step=self.global_step)
        return avg_loss
