"""Run lifecycle and idempotency tests."""

from __future__ import annotations

import json
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


def test_abandoned_active_run_can_be_requeued_without_allow_retry(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "runs.json")
    registry.transition("train-abandoned", "queued")
    registry.transition("train-abandoned", "running")

    # Point the stored record at a pid that cannot be alive, as a killed run leaves behind.
    payload = json.loads((tmp_path / "runs.json").read_text(encoding="utf-8"))
    payload["runs"]["train-abandoned"]["pid"] = 2**31 - 1
    (tmp_path / "runs.json").write_text(json.dumps(payload), encoding="utf-8")

    record = registry.transition("train-abandoned", "queued")

    assert record["status"] == "queued"
    assert [item["status"] for item in record["history"]][-2:] == ["failed", "queued"]
    assert "abandoned" in record["history"][-2]["detail"]["reason"]


def test_live_active_run_still_blocks_a_duplicate(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "runs.json")
    registry.transition("train-live", "queued")
    registry.transition("train-live", "running")

    with pytest.raises(ValueError, match="duplicate active or completed run"):
        registry.transition("train-live", "queued")


def test_abandoned_run_is_retryable_but_live_run_is_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = RunRegistry(tmp_path / "runs.json")
    registry.transition("run-killed", "queued")
    registry.transition("run-killed", "running")

    # A crash or a kill leaves the record in an active status forever. Requeuing
    # must stay blocked while the owner lives and open up once it is gone.
    with pytest.raises(ValueError, match="duplicate"):
        registry.transition("run-killed", "queued")

    monkeypatch.setattr(
        "python_starter.experiments.run_registry._pid_alive", lambda pid: False
    )
    record = registry.transition("run-killed", "queued")

    assert record["status"] == "queued"
    assert [event["status"] for event in record["history"]] == [
        "queued",
        "running",
        "failed",
        "queued",
    ]
    assert record["history"][2]["detail"]["reason"].startswith("abandoned")


def test_completed_run_stays_blocked_even_when_owner_is_gone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = RunRegistry(tmp_path / "runs.json")
    registry.transition("run-done", "queued")
    registry.transition("run-done", "completed")

    monkeypatch.setattr(
        "python_starter.experiments.run_registry._pid_alive", lambda pid: False
    )

    with pytest.raises(ValueError, match="duplicate"):
        registry.transition("run-done", "queued")
