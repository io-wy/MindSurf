"""Checkpoint primitives for reproducible MiniMind pretraining runs."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import torch


SCHEMA_VERSION = 1
RESUME_COMPATIBILITY_KEYS = (
    "data_path",
    "data_mix_json",
    "packed",
    "stream_packed",
    "shuffle_buffer",
    "seed",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "intermediate_size",
    "use_moe",
    "max_seq_len",
    "batch_size",
    "accumulation_steps",
    "dtype",
    "learning_rate",
    "weight_decay",
    "adam_beta1",
    "adam_beta2",
    "adam_eps",
    "lr_schedule",
    "min_lr_ratio",
    "lr_stable_ratio",
    "warmup_steps",
    "max_steps",
)


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def validate_resume_config(
    saved: dict[str, Any],
    current: dict[str, Any],
    keys: tuple[str, ...] = RESUME_COMPATIBILITY_KEYS,
) -> None:
    mismatches = [
        key
        for key in keys
        if key in saved and key in current and saved[key] != current[key]
    ]
    if mismatches:
        details = ", ".join(
            f"{key}: saved={saved[key]!r} current={current[key]!r}"
            for key in mismatches
        )
        raise ValueError(f"resume checkpoint is incompatible with current run: {details}")


def save_training_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    progress: dict[str, int],
    run_config: dict[str, Any],
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "progress": dict(progress),
        "run_config": dict(run_config),
        "rng_state": _capture_rng_state(),
    }
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    return destination


def load_training_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    # Stage the complete payload on CPU. Besides model tensors, it contains the
    # CPU RNG state required by torch.set_rng_state; mapping the whole payload
    # to CUDA makes that state invalid. load_state_dict moves trainable state to
    # each parameter's device after deserialization.
    del map_location
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported checkpoint schema_version={payload.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    _restore_rng_state(payload["rng_state"])
    return {
        "schema_version": payload["schema_version"],
        "progress": payload["progress"],
        "run_config": payload["run_config"],
    }
