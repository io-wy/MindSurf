"""Ratchet-and-canary gate tests.

Each test pins a property the v2 gate lacked, and the lack of which changed a
real verdict during this project.
"""

from __future__ import annotations

import pytest
from scripts.judge_gate_v3 import MissingMetricError, judge_criterion, read_metric

NOISE = {"strict_val": 0.0068, "mcq.accuracy": 0.099}
TOLERANCE = 3.0

LOWER = {"path": "strict_val.loss", "direction": "lower_is_better", "gating": True}
HIGHER = {"path": "mcq.accuracy", "direction": "higher_is_better", "gating": False}


def _judge(spec, candidate, reference=None, name="strict_val"):
    return judge_criterion(name, spec, candidate, reference, NOISE, TOLERANCE)


def test_a_small_regression_within_tolerance_passes() -> None:
    # 0.01 worse against a tolerated 3 x 0.0068 = 0.0204.
    result = _judge(LOWER, {"strict_val": {"loss": 1.73}}, {"strict_val": {"loss": 1.72}})

    assert result["verdict"] == "passed"
    assert result["margin"] == pytest.approx(-0.01)


def test_a_regression_beyond_tolerance_fails() -> None:
    result = _judge(LOWER, {"strict_val": {"loss": 1.80}}, {"strict_val": {"loss": 1.72}})

    assert result["verdict"] == "regressed"


def test_an_improvement_passes_with_positive_margin() -> None:
    result = _judge(LOWER, {"strict_val": {"loss": 1.60}}, {"strict_val": {"loss": 1.72}})

    assert result["verdict"] == "passed"
    assert result["margin"] > 0


def test_margin_sign_means_better_for_higher_is_better_metrics_too() -> None:
    """Downstream readers must not have to re-derive which way a metric points."""
    better = _judge(HIGHER, {"mcq": {"accuracy": 0.40}}, {"mcq": {"accuracy": 0.30}}, "mcq.accuracy")
    worse = _judge(HIGHER, {"mcq": {"accuracy": 0.20}}, {"mcq": {"accuracy": 0.30}}, "mcq.accuracy")

    assert better["margin"] > 0
    assert worse["margin"] < 0


def test_absent_reference_is_not_a_pass() -> None:
    """The absence of something to compare against is not evidence of quality."""
    result = _judge(LOWER, {"strict_val": {"loss": 1.73}}, None)

    assert result["verdict"] == "no_reference"


def test_canary_fires_regardless_of_reference() -> None:
    spec = dict(LOWER, canary_max=2.42)

    collapsed = _judge(spec, {"strict_val": {"loss": 9.9}}, {"strict_val": {"loss": 1.72}})
    collapsed_without_reference = _judge(spec, {"strict_val": {"loss": 9.9}}, None)

    assert collapsed["verdict"] == "canary_failed"
    assert collapsed_without_reference["verdict"] == "canary_failed"


def test_a_missing_metric_is_an_error_not_a_silent_pass() -> None:
    """Skipping an absent criterion is indistinguishable from passing it."""
    with pytest.raises(MissingMetricError):
        _judge(LOWER, {"strict_test": {"loss": 1.73}}, {"strict_val": {"loss": 1.72}})


def test_a_non_numeric_metric_is_rejected() -> None:
    with pytest.raises(MissingMetricError):
        read_metric({"strict_val": {"loss": "1.73"}}, "strict_val.loss")


def test_non_gating_criteria_carry_their_disqualification() -> None:
    """An invalid instrument may be reported but must never decide."""
    result = _judge(HIGHER, {"mcq": {"accuracy": 0.01}}, {"mcq": {"accuracy": 0.38}}, "mcq.accuracy")

    assert result["gating"] is False
    assert result["verdict"] == "regressed"
