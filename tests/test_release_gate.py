"""Publication gate rule tests."""

from __future__ import annotations

import re

import pytest
from scripts.release_gate import CONTENT_RULES, PATH_RULES, RULE_EXEMPT_PATHS

CONTENT_BY_NAME = {name: pattern for name, pattern, _ in CONTENT_RULES}
PATH_BY_NAME = {name: pattern for name, pattern, _ in PATH_RULES}


def _hits(rule: str, line: str) -> bool:
    return bool(re.search(CONTENT_BY_NAME[rule], line, re.IGNORECASE))


@pytest.mark.parametrize(
    "rule,line",
    [
        ("agent_process_record", "Co-Authored-By: Claude Opus 4.8"),
        ("agent_process_record", "generated with codex"),
        ("personal_path", '"/home/oscar/mindsurf-workspace/configs/x.json"'),
        ("personal_path", "D:\\\\UserData\\\\Desktop\\\\minimind"),
        ("internal_address", "HostName 192.168.6.3"),
        ("internal_address", "ssh 10.0.12.7"),
        ("credential", "-----BEGIN OPENSSH PRIVATE KEY-----"),
        ("credential", "token = ghp_abcdefghijklmnopqrstuvwxyz0123456789"),
    ],
)
def test_forbidden_content_is_caught(rule: str, line: str) -> None:
    assert _hits(rule, line)


@pytest.mark.parametrize(
    "rule,line",
    [
        # A compliant experiment report must survive the gate: it is the
        # evidence the completion checklist requires.
        ("personal_path", "strict mean 2.3725 -> 1.9492 over 737,280,000 tokens"),
        ("personal_path", "see configs/evaluation/pretrain_gate_80m.json"),
        ("internal_address", "seen tokens 122,880,000 across 10,000 steps"),
        ("agent_process_record", "the maintainer reviewed 32 sampled items"),
    ],
)
def test_compliant_documentation_is_not_flagged(rule: str, line: str) -> None:
    assert not _hits(rule, line)


def test_dependency_version_is_not_read_as_an_internal_address() -> None:
    line = 'name = "nvidia-curand-cu12"\nversion = "10.3.5.147"'

    # The bare pattern cannot tell a dotted quad from a version, so the rule is
    # exempted for generated lock files rather than weakened everywhere.
    assert _hits("internal_address", line)
    assert RULE_EXEMPT_PATHS["internal_address"].search("uv.lock")
    assert RULE_EXEMPT_PATHS["internal_address"].search("frontend/package-lock.json")
    assert not RULE_EXEMPT_PATHS["internal_address"].search("docs/experiments/report.md")


@pytest.mark.parametrize(
    "name,expected",
    [
        ("models/checkpoints/final_model.pt", True),
        (".codex/state.json", True),
        ("docs/task-brief-2026.md", True),
        ("configs/evaluation/pretrain_gate_80m.json", False),
        ("docs/experiments/2026-07-21-pretraining-budget-attribution.md", False),
    ],
)
def test_path_rules(name: str, expected: bool) -> None:
    matched = any(re.search(pattern, name, re.IGNORECASE) for pattern in PATH_BY_NAME.values())

    assert matched is expected


def test_every_exempt_file_exists() -> None:
    """An exemption for a path that no longer exists is a silent hole."""
    from pathlib import Path

    from scripts.release_gate import EXEMPT, ROOT

    missing = sorted(name for name in EXEMPT if not (ROOT / Path(name)).is_file())

    assert missing == []
