"""Alert rules for a live pretraining run.

The rules read the tracker's ``metrics.jsonl`` plus the free space on the
checkpoint filesystem, so a run can be watched from a second process without
touching the training loop. Every rule is a pure function of the parsed
samples and the frozen thresholds, which keeps the decision boundary testable
without a GPU.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any


@dataclass(frozen=True)
class AlertThresholds:
    """Frozen alert boundaries for one run.

    ``train/loss`` is one micro-batch, and on this corpus a single Chinese,
    code, or math block moves it between roughly 1.5 and 4.7 with no fault
    present. The spike rule therefore compares the median of two adjacent
    windows instead of the latest value against a trailing median, which would
    fire on ordinary batch noise.
    """

    loss_spike_ratio: float = 1.5
    loss_spike_window: int = 40
    stall_seconds: float = 600.0
    gpu_reserved_fraction_max: float = 0.95
    free_bytes_min: int = 20_000_000_000
    # Hardware limits. Temperature must be sustained rather than instantaneous:
    # a single sample above the line is a sampling artefact, a run of them is a
    # cooling problem.
    gpu_temperature_max_celsius: float = 85.0
    gpu_temperature_sustained_samples: int = 3
    gradient_norm_rise_samples: int = 100

    def as_dict(self) -> dict[str, Any]:
        return {
            "loss_spike_ratio": self.loss_spike_ratio,
            "loss_spike_window": self.loss_spike_window,
            "stall_seconds": self.stall_seconds,
            "gpu_reserved_fraction_max": self.gpu_reserved_fraction_max,
            "free_bytes_min": self.free_bytes_min,
            "gpu_temperature_max_celsius": self.gpu_temperature_max_celsius,
            "gpu_temperature_sustained_samples": self.gpu_temperature_sustained_samples,
            "gradient_norm_rise_samples": self.gradient_norm_rise_samples,
        }


@dataclass(frozen=True)
class Sample:
    """One logged training step."""

    step: int
    loss: float
    peak_reserved_bytes: float
    gradient_norm: float | None = None
    learning_rate: float | None = None


@dataclass
class MonitorState:
    """Everything the rules need beyond the samples themselves."""

    samples: list[Sample] = field(default_factory=list)
    # Hardware samples arrive on a separate channel from the tracker log; the
    # alert rules are the point where the two must finally meet.
    temperatures_celsius: list[float] = field(default_factory=list)
    uncorrectable_ecc_errors: int = 0
    seconds_since_last_step: float = 0.0
    free_bytes: int | None = None
    gpu_total_bytes: int | None = None


def parse_metrics(path: Path) -> list[Sample]:
    """Read tracker metric rows, ignoring params and non-step rows."""
    samples: list[Sample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("type") != "metrics" or "step" not in row:
                continue
            metrics = row.get("metrics", {})
            if "train/loss" not in metrics:
                continue
            samples.append(
                Sample(
                    step=int(row["step"]),
                    loss=float(metrics["train/loss"]),
                    peak_reserved_bytes=float(metrics.get("train/peak_cuda_reserved_bytes", 0.0)),
                    gradient_norm=_optional_float(metrics.get("train/gradient_norm")),
                    learning_rate=_optional_float(metrics.get("train/lr")),
                )
            )
    return samples


def evaluate_alerts(state: MonitorState, thresholds: AlertThresholds) -> list[dict[str, Any]]:
    """Return every alert that the current state trips, most severe first."""
    alerts: list[dict[str, Any]] = []
    samples = state.samples

    for sample in samples:
        if not math.isfinite(sample.loss):
            alerts.append(
                {
                    "alert": "non_finite_loss",
                    "severity": "critical",
                    "step": sample.step,
                    "detail": f"train/loss is {sample.loss}",
                }
            )
            break

    half = thresholds.loss_spike_window // 2
    if half and len(samples) >= 2 * half:
        older = [item.loss for item in samples[-2 * half : -half] if math.isfinite(item.loss)]
        recent = [item.loss for item in samples[-half:] if math.isfinite(item.loss)]
        if older and recent:
            baseline = median(older)
            current = median(recent)
            limit = baseline * thresholds.loss_spike_ratio
            if baseline > 0 and current > limit:
                alerts.append(
                    {
                        "alert": "loss_spike",
                        "severity": "critical",
                        "step": samples[-1].step,
                        "detail": (
                            f"median train/loss rose {baseline:.4f} -> {current:.4f} over the last "
                            f"{half} logged steps, limit {limit:.4f}"
                        ),
                    }
                )

    if state.seconds_since_last_step > thresholds.stall_seconds:
        alerts.append(
            {
                "alert": "no_progress",
                "severity": "critical",
                "step": samples[-1].step if samples else None,
                "detail": (
                    f"no logged step for {state.seconds_since_last_step:.0f}s, "
                    f"limit {thresholds.stall_seconds:.0f}s"
                ),
            }
        )

    if state.gpu_total_bytes and samples:
        fraction = samples[-1].peak_reserved_bytes / state.gpu_total_bytes
        if fraction > thresholds.gpu_reserved_fraction_max:
            alerts.append(
                {
                    "alert": "gpu_memory_pressure",
                    "severity": "warning",
                    "step": samples[-1].step,
                    "detail": (
                        f"peak reserved {fraction:.1%} of device memory, "
                        f"limit {thresholds.gpu_reserved_fraction_max:.1%}"
                    ),
                }
            )

    if state.free_bytes is not None and state.free_bytes < thresholds.free_bytes_min:
        alerts.append(
            {
                "alert": "disk_low",
                "severity": "warning",
                "step": samples[-1].step if samples else None,
                "detail": (
                    f"{state.free_bytes / 1e9:.1f} GB free, "
                    f"budget {thresholds.free_bytes_min / 1e9:.1f} GB"
                ),
            }
        )

    if state.uncorrectable_ecc_errors:
        alerts.append(
            {
                "alert": "hbm_uncorrectable_ecc",
                "severity": "critical",
                "step": samples[-1].step if samples else None,
                "detail": (
                    f"{state.uncorrectable_ecc_errors} uncorrectable ECC error(s); "
                    "memory corruption invalidates the run"
                ),
            }
        )

    sustained = thresholds.gpu_temperature_sustained_samples
    recent_temperatures = state.temperatures_celsius[-sustained:]
    if (
        sustained > 0
        and len(recent_temperatures) == sustained
        and all(value > thresholds.gpu_temperature_max_celsius for value in recent_temperatures)
    ):
        alerts.append(
                {
                    "alert": "gpu_temperature",
                    "severity": "warning",
                    "step": samples[-1].step if samples else None,
                    "detail": (
                        f"{sustained} consecutive samples above "
                        f"{thresholds.gpu_temperature_max_celsius:.0f}C, "
                        f"latest {recent_temperatures[-1]:.0f}C"
                    ),
                }
            )

    # A single large gradient is ordinary; a monotone climb is divergence
    # building up, which is what the trend rule is for.
    window = thresholds.gradient_norm_rise_samples
    norms = [item.gradient_norm for item in samples[-window:] if item.gradient_norm is not None]
    if (
        window > 1
        and len(norms) == window
        and all(later > earlier for earlier, later in zip(norms[:-1], norms[1:], strict=True))
    ):
        alerts.append(
                {
                    "alert": "gradient_norm_rising",
                    "severity": "critical",
                    "step": samples[-1].step,
                    "detail": (
                        f"gradient norm rose on {window} consecutive logged steps, "
                        f"{norms[0]:.4f} -> {norms[-1]:.4f}"
                    ),
                }
            )

    order = {"critical": 0, "warning": 1}
    alerts.sort(key=lambda item: order.get(str(item["severity"]), 2))
    return alerts


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_telemetry(path: Path) -> tuple[list[float], int]:
    """Read the GPU sampler's log: temperatures, and any uncorrectable ECC count.

    The sampler writes to its own file rather than the tracker log, so without
    this the hardware thresholds could never fire no matter how they were set.
    """
    temperatures: list[float] = []
    uncorrectable = 0
    if not path.is_file():
        return temperatures, uncorrectable
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            temperature = _optional_float(row.get("temperature_celsius"))
            if temperature is not None:
                temperatures.append(temperature)
            errors = _optional_float(row.get("ecc_uncorrectable_total"))
            if errors:
                uncorrectable = max(uncorrectable, int(errors))
    return temperatures, uncorrectable
