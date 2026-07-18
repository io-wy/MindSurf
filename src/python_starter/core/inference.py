"""Checkpoint-aware local inference shared by CLI and API surfaces."""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from transformers import PreTrainedTokenizerBase

from python_starter.core.model import ModelConfig, TransformerLM
from python_starter.core.tokenizer import load_tokenizer
from python_starter.core.utils import get_device


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 identity for a checkpoint."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint_model(
    checkpoint_path: str | Path,
    device: str | torch.device = "auto",
) -> tuple[TransformerLM, dict[str, Any], torch.device]:
    """Load a trusted internal checkpoint using its embedded model config."""
    resolved_device = device if isinstance(device, torch.device) else get_device(device)
    checkpoint = cast(
        dict[str, Any],
        torch.load(checkpoint_path, map_location="cpu", weights_only=False),
    )
    raw_config = checkpoint.get("model_config")
    if not isinstance(raw_config, dict):
        raise ValueError("checkpoint is missing model_config")
    model = TransformerLM(ModelConfig.from_dict(raw_config))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(resolved_device).eval()
    return model, checkpoint, resolved_device


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """One decoded continuation and its runtime accounting."""

    text: str
    input_tokens: int
    output_tokens: int
    generation_time_ms: float


class InferenceEngine:
    """Thread-safe owner for one loaded checkpoint and tokenizer."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        tokenizer_path: str | Path,
        device: str = "auto",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.tokenizer_path = Path(tokenizer_path)
        self.model, self.checkpoint, self.device = load_checkpoint_model(
            self.checkpoint_path,
            device,
        )
        self.tokenizer: PreTrainedTokenizerBase = load_tokenizer(str(self.tokenizer_path))
        self.checkpoint_sha256 = sha256_file(self.checkpoint_path)
        self._lock = threading.Lock()

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> GenerationResult:
        """Generate a continuation without conflating it with the prompt."""
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if not prompt_ids:
            bos = self.tokenizer.bos_token_id
            if bos is None:
                raise ValueError("prompt produced no tokens and tokenizer has no BOS token")
            prompt_ids = [bos]
        prompt_limit = max(1, self.model.config.max_seq_len - 1)
        prompt_ids = prompt_ids[-prompt_limit:]
        input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)

        started = time.perf_counter()
        with self._lock, torch.inference_mode():
            output = self.model.generate(
                input_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        elapsed_ms = (time.perf_counter() - started) * 1000
        continuation_ids = output[0, len(prompt_ids) :].tolist()
        decoded = self.tokenizer.decode(continuation_ids, skip_special_tokens=True)
        if not isinstance(decoded, str):
            raise TypeError("tokenizer returned a batched decode result")
        return GenerationResult(
            text=decoded,
            input_tokens=len(prompt_ids),
            output_tokens=len(continuation_ids),
            generation_time_ms=elapsed_ms,
        )
