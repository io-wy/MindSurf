"""Training alert rule tests."""

from __future__ import annotations

import json
from pathlib import Path

from python_starter.infrastructure.training_monitor import (
    AlertThresholds,
    MonitorState,
    Sample,
    evaluate_alerts,
    parse_metrics,
)

# Real micro-batch losses logged every 1,000 steps by the frozen Official parent
# run. The spread is ordinary corpus heterogeneity, not a fault.
PARENT_SAMPLE_LOSSES = [2.866, 4.726, 2.389, 1.498, 3.171, 1.959, 2.144, 2.275, 2.546, 2.111]


def _state(losses: list[float], **values: object) -> MonitorState:
    samples = [
        Sample(step=(index + 1) * 10, loss=loss, peak_reserved_bytes=9_667_870_720.0)
        for index, loss in enumerate(losses)
    ]
    return MonitorState(samples=samples, **values)  # type: ignore[arg-type]


def _alert_names(state: MonitorState, thresholds: AlertThresholds | None = None) -> set[str]:
    return {item["alert"] for item in evaluate_alerts(state, thresholds or AlertThresholds())}


def test_ordinary_micro_batch_noise_does_not_alert() -> None:
    losses = PARENT_SAMPLE_LOSSES * 8

    assert _alert_names(_state(losses)) == set()


def test_non_finite_loss_alerts() -> None:
    losses = [*PARENT_SAMPLE_LOSSES, float("nan")]

    assert "non_finite_loss" in _alert_names(_state(losses))


def test_sustained_rise_alerts_but_single_bad_batch_does_not() -> None:
    steady = PARENT_SAMPLE_LOSSES * 4
    one_bad_batch = [*steady, 9.0]
    diverged = [*steady, *[value * 2 for value in PARENT_SAMPLE_LOSSES * 2]]

    assert "loss_spike" not in _alert_names(_state(one_bad_batch))
    assert "loss_spike" in _alert_names(_state(diverged))


def test_stall_and_disk_and_memory_alerts() -> None:
    losses = PARENT_SAMPLE_LOSSES * 4

    stalled = _state(losses, seconds_since_last_step=1200.0)
    starved = _state(losses, free_bytes=5_000_000_000)
    crowded = _state(losses, gpu_total_bytes=10_000_000_000)

    assert "no_progress" in _alert_names(stalled)
    assert "disk_low" in _alert_names(starved)
    assert "gpu_memory_pressure" in _alert_names(crowded)


def test_parse_metrics_skips_non_step_rows(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"type": "params", "params": {"max_steps": 10}}),
                json.dumps({"type": "metrics", "step": 10, "metrics": {"train/loss": 2.5}}),
                json.dumps({"type": "metrics", "step": 20, "metrics": {"eval/loss": 3.0}}),
                json.dumps({"type": "finish"}),
            ]
        ),
        encoding="utf-8",
    )

    samples = parse_metrics(path)

    assert [item.step for item in samples] == [10]
