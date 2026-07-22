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

    monkeypatch.setattr("python_starter.experiments.run_registry._pid_alive", lambda pid: False)
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

    monkeypatch.setattr("python_starter.experiments.run_registry._pid_alive", lambda pid: False)

    with pytest.raises(ValueError, match="duplicate"):
        registry.transition("run-done", "queued")


def test_candidate_record_binds_source_commit_and_training_summary(tmp_path: Path) -> None:
    from python_starter.experiments.registry import LocalCandidateRegistry

    checkpoint = tmp_path / "final_model.pt"
    checkpoint.write_bytes(b"weights")
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(
        json.dumps({"gate": {"internal_candidate_passed": True, "public_release_passed": False}}),
        encoding="utf-8",
    )
    summary = tmp_path / "training_summary.json"
    summary.write_text(json.dumps({"global_step": 60000, "seed": 20260511}), encoding="utf-8")
    preflight = tmp_path / "preflight.json"
    preflight.write_text(
        json.dumps({"status": "passed", "source": {"git_head": "deadbeef", "git_branch": "x"}}),
        encoding="utf-8",
    )

    record = LocalCandidateRegistry(tmp_path / "registry.json").register(
        name="candidate-1",
        checkpoint_path=checkpoint,
        evaluation_path=evaluation,
        training_summary_path=summary,
        preflight_path=preflight,
    )

    assert record["source_git_head"] == "deadbeef"
    assert record["training_summary_sha256"]
    assert record["preflight_sha256"]


def test_candidate_registers_on_a_later_gate_verdict_over_a_stale_evaluation(
    tmp_path: Path,
) -> None:
    """A gate v3 verdict overrides the gate the evaluation was born with."""
    from python_starter.experiments.registry import LocalCandidateRegistry

    checkpoint = tmp_path / "checkpoint_step_168000.pt"
    checkpoint.write_bytes(b"weights")
    evaluation = tmp_path / "full168k_seed20260511_80m.json"
    evaluation.write_text(
        json.dumps({"gate": {"internal_candidate_passed": False}}),
        encoding="utf-8",
    )
    verdict = tmp_path / "verdict_full168k_seed20260511.json"
    verdict.write_text(
        json.dumps(
            {
                "evaluation": "artifacts/evaluation/full168k_seed20260511_80m.json",
                "gate": "configs/evaluation/pretrain_gate_v3.json",
                "passed": True,
                "reference": "artifacts/evaluation/reeval_shuf60k_seed20260721_80m.json",
            }
        ),
        encoding="utf-8",
    )
    registry = LocalCandidateRegistry(tmp_path / "registry.json")

    record = registry.register(
        name="candidate-v3",
        checkpoint_path=checkpoint,
        evaluation_path=evaluation,
        verdict_path=verdict,
        limitations=["trained 168000 steps, not the planned 172000"],
    )

    assert record["gate_source"] == "verdict"
    assert record["verdict_sha256"]
    assert record["gate_config"] == "configs/evaluation/pretrain_gate_v3.json"
    # A license decision is not a gate v3 concept, so a v3 pass never promotes.
    assert record["public_release_eligible"] is False
    assert record["limitations"] == ["trained 168000 steps, not the planned 172000"]


def test_candidate_registration_rejects_a_verdict_for_another_evaluation(
    tmp_path: Path,
) -> None:
    from python_starter.experiments.registry import LocalCandidateRegistry

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"weights")
    evaluation = tmp_path / "candidate_a.json"
    evaluation.write_text(json.dumps({"gate": {}}), encoding="utf-8")
    verdict = tmp_path / "verdict_b.json"
    verdict.write_text(
        json.dumps({"evaluation": "artifacts/evaluation/candidate_b.json", "passed": True}),
        encoding="utf-8",
    )
    failing = tmp_path / "verdict_a.json"
    failing.write_text(
        json.dumps({"evaluation": "candidate_a.json", "passed": False}),
        encoding="utf-8",
    )
    registry = LocalCandidateRegistry(tmp_path / "registry.json")

    with pytest.raises(ValueError, match="does not judge this evaluation"):
        registry.register(
            name="mismatched",
            checkpoint_path=checkpoint,
            evaluation_path=evaluation,
            verdict_path=verdict,
        )
    with pytest.raises(ValueError, match="internally passing"):
        registry.register(
            name="failing",
            checkpoint_path=checkpoint,
            evaluation_path=evaluation,
            verdict_path=failing,
        )
