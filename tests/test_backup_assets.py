"""Durability inventory: what it copies, and what it only claims."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "backup_assets.py"


def _run(
    repo: Path, backup_root: Path, output: Path, *extra: str, expect_failure: bool = False
) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo-root",
            str(repo),
            "--backup-root",
            str(backup_root),
            "--output",
            str(output),
            "--no-defaults",
            "--asset",
            "weights=models/weights.pt",
            *extra,
        ],
        capture_output=True,
    )
    # A failed asset has to reach the caller's exit code, not only the JSON:
    # an inventory nobody reads is how a lost asset stays unnoticed.
    assert (completed.returncode != 0) is expect_failure, completed.stderr.decode()
    return cast(dict[str, Any], json.loads(output.read_text(encoding="utf-8")))


def test_copying_backup_stores_the_asset_under_its_digest(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "models").mkdir(parents=True)
    (repo / "models" / "weights.pt").write_bytes(b"weights")
    backup_root = tmp_path / "backup"

    inventory = _run(repo, backup_root, tmp_path / "copied.json")

    assert inventory["missing_assets"] == []
    assert inventory["failed_assets"] == []
    assert inventory["in_place"] is False
    stored = Path(inventory["assets"][0]["backup_path"])
    assert stored.is_file()
    assert stored.read_bytes() == b"weights"
    assert backup_root in stored.parents


def test_in_place_inventory_copies_nothing_and_says_so(tmp_path: Path) -> None:
    """The claim has to shrink to match what an in-place record actually proves."""
    repo = tmp_path / "repo"
    (repo / "models").mkdir(parents=True)
    source = repo / "models" / "weights.pt"
    source.write_bytes(b"weights")
    backup_root = tmp_path / "backup"

    inventory = _run(repo, backup_root, tmp_path / "in_place.json", "--in-place")

    assert inventory["in_place"] is True
    assert inventory["failed_assets"] == []
    assert inventory["assets"][0]["verified"] is True
    assert Path(inventory["assets"][0]["backup_path"]) == source.resolve()
    assert not backup_root.exists()
    assert "not that a second copy exists" in inventory["durability_scope"]


def test_a_truncated_asset_fails_verification(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "models").mkdir(parents=True)
    (repo / "models" / "weights.pt").write_bytes(b"weights")
    backup_root = tmp_path / "backup"
    output = tmp_path / "first.json"
    inventory = _run(repo, backup_root, output)

    Path(inventory["assets"][0]["backup_path"]).write_bytes(b"trunc")

    reverified = _run(
        repo, backup_root, tmp_path / "second.json", "--verify-only", expect_failure=True
    )
    assert reverified["failed_assets"] == ["weights"]
