from __future__ import annotations

import json
import sys
from types import SimpleNamespace

from experiments.pretrain.experiment_tracking import ExperimentTracker, write_run_manifest


def test_tracker_always_writes_jsonl_and_optionally_mirrors_to_trackio(tmp_path, monkeypatch) -> None:
    calls = []
    fake_trackio = SimpleNamespace(
        init=lambda **kwargs: calls.append(("init", kwargs)),
        log=lambda metrics, step=None: calls.append(("log", metrics, step)),
        alert=lambda **kwargs: calls.append(("alert", kwargs)),
        finish=lambda: calls.append(("finish",)),
        AlertLevel=SimpleNamespace(INFO="info", WARN="warn", ERROR="error"),
    )
    monkeypatch.setitem(sys.modules, "trackio", fake_trackio)
    path = tmp_path / "metrics.jsonl"

    tracker = ExperimentTracker(
        path,
        run_name="tiny-run",
        config={"learning_rate": 5e-4},
        trackio_project="minimind-pretrain",
    )
    tracker.log({"loss": 1.25, "tokens_per_second": 1234.0}, step=4)
    tracker.alert("loss spike", "loss increased", level="warn", step=4)
    tracker.finish(status="completed", summary={"last_loss": 1.25})

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["event"] for row in rows] == ["start", "metrics", "alert", "finish"]
    assert rows[1]["step"] == 4
    assert rows[1]["metrics"]["loss"] == 1.25
    assert [call[0] for call in calls] == ["init", "log", "alert", "finish"]


def test_run_manifest_is_replaced_atomically_and_keeps_source_revision(tmp_path) -> None:
    path = tmp_path / "run_manifest.json"
    write_run_manifest(
        path,
        run_name="tiny-run",
        config={"max_steps": 2},
        source_revision="abc123",
        artifacts={"metrics": "metrics.jsonl"},
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["run_name"] == "tiny-run"
    assert payload["source_revision"] == "abc123"
    assert payload["artifacts"]["metrics"] == "metrics.jsonl"
    assert not list(tmp_path.glob("*.tmp"))
