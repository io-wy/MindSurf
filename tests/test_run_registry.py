"""Run lifecycle and idempotency tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from python_starter.experiments.run_registry import RunRegistry


def test_registry_records_lifecycle_and_rejects_duplicate(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "runs.json")
    registry.transition("run-1", "queued")
    registry.transition("run-1", "running")
    with pytest.raises(ValueError, match="duplicate"):
        registry.transition("run-1", "queued")
    registry.transition("run-1", "completed")
    record = registry.get("run-1")
    assert record is not None
    assert record["status"] == "completed"
    assert [event["status"] for event in record["history"]] == [
        "queued",
        "running",
        "completed",
    ]


def test_failed_run_can_be_retried_explicitly(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "runs.json")
    registry.transition("run-1", "queued")
    registry.transition("run-1", "failed")
    registry.transition("run-1", "queued", allow_retry=True)
    registry.transition("run-1", "resumed")
    assert registry.get("run-1")["status"] == "resumed"  # type: ignore[index]
