"""Regression checks for reproducible Docker dependency layers."""

from __future__ import annotations

from pathlib import Path

import pytest

DOCKER_DIR = Path(__file__).parent.parent / "docker"


@pytest.mark.parametrize("dockerfile_name", ("Dockerfile.cpu", "Dockerfile.gpu"))
def test_dependency_layer_does_not_build_project_before_sources_are_copied(
    dockerfile_name: str,
) -> None:
    """The dependency-only layer must not require README.md or src/."""
    content = (DOCKER_DIR / dockerfile_name).read_text(encoding="utf-8")
    sync_commands = [line for line in content.splitlines() if line.startswith("RUN uv sync")]

    assert len(sync_commands) == 1
    assert "--no-install-project" in sync_commands[0]
