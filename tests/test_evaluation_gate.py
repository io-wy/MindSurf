"""Candidate evaluation and registry gate tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from python_starter.core.evaluation import DOMAIN_NAMES, evaluate_gate
from python_starter.experiments.registry import LocalCandidateRegistry


def _metrics() -> dict[str, Any]:
    return {
        "strict_val": {"loss": 2.0},
        "strict_test": {"loss": 2.0},
        "domain": dict.fromkeys(DOMAIN_NAMES, 2.0),
        "mcq": {"accuracy": 0.75},
        "fixed_prompts": {"mean_score": 0.75, "repetition_count": 1},
    }


def _thresholds() -> dict[str, Any]:
    return {
        "strict_val_max": 2.5,
        "strict_test_max": 2.5,
        "domain_max": dict.fromkeys(DOMAIN_NAMES, 2.5),
        "mcq_accuracy_min": 0.5,
        "fixed_score_min": 0.5,
        "repetition_count_max": 2,
    }


def test_public_release_stays_closed_without_license_readiness() -> None:
    gate = evaluate_gate(_metrics(), _thresholds(), license_ready=False)
    assert gate["internal_candidate_passed"] is True
    assert gate["public_release_passed"] is False


def test_missing_domain_evidence_fails_closed() -> None:
    metrics = _metrics()
    del metrics["domain"]["math_like"]
    gate = evaluate_gate(metrics, _thresholds(), license_ready=True)
    assert gate["internal_candidate_passed"] is False
    assert "domain.math_like" in gate["failures"]


def test_registry_rejects_failed_candidate(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(
        json.dumps({"gate": {"internal_candidate_passed": False}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="internally passing"):
        LocalCandidateRegistry(tmp_path / "registry.json").register(
            name="failed",
            checkpoint_path=checkpoint,
            evaluation_path=evaluation,
        )


def test_pretrain_gate_excludes_instruction_score() -> None:
    metrics = _metrics()
    metrics["fixed_prompts"]["mean_score"] = 0.0
    metrics["generation"] = {"repetition_count": 1}
    metrics["mcq"]["wilson_interval_95"] = [0.4, 0.9]
    thresholds = {
        **_thresholds(),
        "gate_kind": "pretrain",
        "mcq_wilson_lower_bound_min": 0.25,
    }
    assert evaluate_gate(metrics, thresholds, license_ready=False)["internal_candidate_passed"]


def test_posttrain_gate_is_reported_separately() -> None:
    thresholds = {
        "gate_kind": "posttrain",
        "fixed_score_min": 0.5,
        "repetition_count_max": 2,
    }
    passed = evaluate_gate(_metrics(), thresholds, license_ready=False)
    assert passed["posttrain_passed"] is True
    metrics = _metrics()
    metrics["fixed_prompts"]["mean_score"] = 0.2
    failed = evaluate_gate(metrics, thresholds, license_ready=False)
    assert failed["posttrain_passed"] is False
