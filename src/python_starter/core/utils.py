"""Utility functions for ML training."""

from __future__ import annotations

import os
import random

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def get_device(preferred: str = "auto") -> torch.device:
    """Determine the best available device.

    Args:
        preferred: "auto", "cuda", "cpu", or "mps".

    Returns:
        torch.device instance.
    """
    if preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preferred)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count model parameters.

    Args:
        model: PyTorch model.
        trainable_only: If True, count only trainable parameters.

    Returns:
        Number of parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_number(n: int) -> str:
    """Format large numbers with K/M/B suffixes."""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.2f}K"
    return str(n)


# Dense bfloat16 tensor-core peak, excluding the 2x structured-sparsity figure
# vendors headline. Keyed by the name nvidia-smi reports.
DEVICE_PEAK_BF16_FLOPS = {
    "NVIDIA GeForce RTX 4090": 165.2e12,
    "NVIDIA GeForce RTX 5070 Laptop GPU": 61.4e12,
    "NVIDIA A100-SXM4-40GB": 312.0e12,
    "NVIDIA A100-SXM4-80GB": 312.0e12,
}


def model_flops_per_token(
    *,
    parameter_count: int,
    n_layer: int,
    n_embed: int,
    sequence_length: int,
) -> float:
    """Training FLOPs per token: 6N for the dense terms plus the attention term.

    Follows the PaLM accounting, so the number is comparable to published MFU
    figures rather than to a vendor throughput benchmark.
    """
    if min(parameter_count, n_layer, n_embed, sequence_length) <= 0:
        raise ValueError("model geometry must be positive")
    return 6.0 * parameter_count + 12.0 * n_layer * n_embed * sequence_length


def model_flops_utilization(
    *,
    flops_per_token: float,
    tokens_per_second: float,
    device_peak_flops: float,
) -> float:
    """Fraction of the device's dense peak the run actually sustained."""
    if device_peak_flops <= 0:
        raise ValueError("device_peak_flops must be positive")
    if min(flops_per_token, tokens_per_second) < 0:
        raise ValueError("throughput terms must be non-negative")
    return flops_per_token * tokens_per_second / device_peak_flops
