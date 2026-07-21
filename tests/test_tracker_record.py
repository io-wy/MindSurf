"""Experiment record completeness tests."""

from __future__ import annotations

import json
from pathlib import Path

from python_starter.experiments.tracker import (
    _config_fingerprint,
    _environment_snapshot,
)


def test_config_fingerprint_is_order_independent_and_change_sensitive() -> None:
    a = _config_fingerprint({"lr": 5e-4, "steps": 60000})
    b = _config_fingerprint({"steps": 60000, "lr": 5e-4})
    c = _config_fingerprint({"steps": 60000, "lr": 1e-4})

    assert a == b
    assert a != c


def test_environment_snapshot_records_what_cannot_be_recovered_later() -> None:
    snapshot = _environment_snapshot()

    # Each of these is unrecoverable after the fact and changes results.
    for key in ("git_head", "python", "platform", "processor_count", "torch"):
        assert key in snapshot
    assert "dependencies_sha256" in snapshot


def test_run_record_carries_environment_and_fingerprint(tmp_path: Path) -> None:
    from python_starter.experiments.tracker import ExperimentTracker
    from python_starter.infrastructure.config import Settings

    tracker = ExperimentTracker(Settings(), experiment_name="t", local_dir=tmp_path)
    tracker.start(run_name="run-a", config={"lr": 5e-4})
    tracker.finish()

    record = json.loads((tmp_path / "run-a" / "run.json").read_text(encoding="utf-8"))
    events = [
        json.loads(line)
        for line in (tmp_path / "run-a" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert record["config_fingerprint"]
    assert record["environment"]["python"]
    assert any(event.get("type") == "finish" and event.get("ended_at") for event in events)


def test_reusing_a_run_name_is_recorded_not_hidden(tmp_path: Path) -> None:
    from python_starter.experiments.tracker import ExperimentTracker
    from python_starter.infrastructure.config import Settings

    first = ExperimentTracker(Settings(), experiment_name="t", local_dir=tmp_path)
    first.start(run_name="same", config={"lr": 5e-4})
    first.finish()

    second = ExperimentTracker(Settings(), experiment_name="t", local_dir=tmp_path)
    second.start(run_name="same", config={"lr": 1e-4})

    events = [
        json.loads(line)
        for line in (tmp_path / "same" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    resume = [event for event in events if event.get("type") == "resume"]
    record = json.loads((tmp_path / "same" / "run.json").read_text(encoding="utf-8"))

    # run.json still describes the first run; the second is only discoverable
    # because the resume row carries a differing config fingerprint.
    assert len(resume) == 1
    assert resume[0]["config_fingerprint"] != record["config_fingerprint"]
