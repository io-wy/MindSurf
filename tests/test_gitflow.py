"""Repository-level checks for the documented Gitflow branch model."""

from __future__ import annotations

from pathlib import Path

import pytest

WORKFLOW_DIR = Path(__file__).parent.parent / ".github" / "workflows"
GITFLOW_WORKFLOWS = (
    "build-docker.yml",
    "ci.yml",
    "quality.yml",
    "test.yml",
)


@pytest.mark.parametrize("workflow_name", GITFLOW_WORKFLOWS)
def test_ci_workflows_target_main_and_develop(workflow_name: str) -> None:
    """CI must protect the two permanent branches used by this repository."""
    content = (WORKFLOW_DIR / workflow_name).read_text(encoding="utf-8")

    assert "master" not in content
    assert "main" in content
    assert "develop" in content
