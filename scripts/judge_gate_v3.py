"""Judge a candidate against the v3 ratchet-and-canary gate.

The v2 gate ranked candidates against absolute thresholds copied from a legacy
reference checkpoint. Once candidates cleared those by two orders of magnitude
more than the seed-to-seed noise, the loss criteria stopped separating anything
and the verdict had silently moved onto the two instruments that failed their
validity checks. This gate splits the two jobs: a ratchet decides whether a
candidate is good enough relative to a reference, and retired absolute
thresholds stay on as canaries that only fire on collapse.

Three properties matter more than the arithmetic:

- a criterion without gating qualification is computed and reported but can
  never change the verdict, so an invalid instrument cannot quietly decide;
- a missing reference produces ``no_reference``, never a pass, because the
  absence of something to compare against is not evidence of quality;
- a metric absent from the artifact is an error, because silently skipping a
  criterion is indistinguishable from passing it.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from python_starter.core.data_contract import write_json_atomic  # noqa: E402


class MissingMetricError(KeyError):
    """A criterion names a path the evaluation artifact does not contain."""


def read_metric(metrics: dict[str, Any], path: str) -> float:
    """Resolve a dotted path, refusing to treat absence as zero."""
    node: Any = metrics
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            raise MissingMetricError(f"evaluation artifact has no metric at {path!r}")
        node = node[part]
    if isinstance(node, bool) or not isinstance(node, (int, float)):
        raise MissingMetricError(f"metric at {path!r} is not numeric: {node!r}")
    return float(node)


def judge_criterion(
    name: str,
    spec: dict[str, Any],
    candidate: dict[str, Any],
    reference: dict[str, Any] | None,
    noise_floors: dict[str, Any],
    tolerance: float,
) -> dict[str, Any]:
    """Evaluate one criterion: canary first, then ratchet."""
    path = str(spec["path"])
    lower_is_better = str(spec.get("direction", "lower_is_better")) == "lower_is_better"
    value = read_metric(candidate, path)
    gating = bool(spec.get("gating", False))
    noise = float(noise_floors.get(name, 0.0))

    result: dict[str, Any] = {
        "value": value,
        "direction": spec.get("direction", "lower_is_better"),
        "gating": gating,
        "noise_floor": noise,
        "note": spec.get("note"),
    }

    canary_max = spec.get("canary_max")
    if canary_max is not None and value > float(canary_max):
        result.update(
            {
                "verdict": "canary_failed",
                "canary_max": float(canary_max),
                "detail": f"{value:.4f} exceeds the collapse bound {float(canary_max):.4f}",
            }
        )
        return result
    if canary_max is not None:
        result["canary_max"] = float(canary_max)

    if reference is None:
        result.update({"verdict": "no_reference", "detail": "no reference to ratchet against"})
        return result

    reference_value = read_metric(reference, path)
    # Positive margin always means "better than the reference", whichever way
    # the metric points, so downstream reading never has to re-derive it.
    margin = reference_value - value if lower_is_better else value - reference_value
    allowed = tolerance * noise
    result.update(
        {
            "reference": reference_value,
            "margin": margin,
            "allowed_regression": allowed,
            "verdict": "passed" if margin >= -allowed else "regressed",
            "detail": (
                f"{value:.4f} against reference {reference_value:.4f}; "
                f"margin {margin:+.4f}, tolerated regression {allowed:.4f}"
            ),
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--gate", type=Path, default=ROOT / "configs/evaluation/pretrain_gate_v3.json")
    parser.add_argument(
        "--reference",
        default="none",
        help="Reference evaluation artifact, or 'none' when no reference exists yet",
    )
    parser.add_argument("--reference-source", default="unspecified")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    gate = json.loads(args.gate.read_text(encoding="utf-8"))
    candidate = json.loads(args.evaluation.read_text(encoding="utf-8"))["metrics"]
    reference = (
        json.loads(Path(args.reference).read_text(encoding="utf-8"))["metrics"]
        if str(args.reference).lower() != "none"
        else None
    )

    tolerance = float(gate["tolerance_noise_multiple"])
    noise_floors = gate["noise_floors"]
    criteria = {
        name: judge_criterion(name, spec, candidate, reference, noise_floors, tolerance)
        for name, spec in gate["criteria"].items()
    }

    gating = {name: item for name, item in criteria.items() if item["gating"]}
    failures = sorted(
        name for name, item in gating.items() if item["verdict"] in {"canary_failed", "regressed"}
    )
    unjudged = sorted(name for name, item in gating.items() if item["verdict"] == "no_reference")
    passed = not failures and not unjudged

    verdict = {
        "schema_version": 1,
        "judged_at": datetime.now(UTC).isoformat(),
        "gate": str(args.gate),
        "evaluation": str(args.evaluation),
        "reference": None if reference is None else str(args.reference),
        "reference_source": args.reference_source,
        "tolerance_noise_multiple": tolerance,
        "passed": passed,
        "failures": failures,
        "unjudged_for_lack_of_reference": unjudged,
        "reported_not_gating": sorted(
            name for name, item in criteria.items() if not item["gating"]
        ),
        "criteria": criteria,
    }
    write_json_atomic(args.output, verdict)
    print(
        json.dumps(
            {
                "passed": passed,
                "failures": failures,
                "unjudged": unjudged,
                "reported_not_gating": verdict["reported_not_gating"],
            },
            ensure_ascii=False,
        )
    )
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
