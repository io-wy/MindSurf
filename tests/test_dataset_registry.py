"""Shared dataset-index tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from python_starter.core.dataset_registry import DatasetRegistry


def _write_fixture(root: Path) -> Path:
    (root / "configs").mkdir()
    (root / "artifacts").mkdir()
    spec = root / "configs/spec.json"
    audit = root / "artifacts/audit.json"
    spec.write_text(
        json.dumps({"dataset_id": "owner/data", "revision": "frozen"}),
        encoding="utf-8",
    )
    audit.write_text(
        json.dumps(
            {
                "dataset_id": "owner/data",
                "dataset_revision": "frozen",
                "status": "passed",
            }
        ),
        encoding="utf-8",
    )
    index = root / "index.json"
    index.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "aliases": {"fixture": "fixture_v1"},
                "datasets": {
                    "fixture_v1": {
                        "dataset_id": "owner/data",
                        "revision": "frozen",
                        "hydra_config": "fixture_v1",
                        "identity": "fixture_v1",
                        "spec": "configs/spec.json",
                        "audit": "artifacts/audit.json",
                        "training_view_manifest": "artifacts/view.json",
                        "train": "data/train.jsonl",
                        "validation": "data/val.jsonl",
                        "test": "data/test.jsonl",
                        "tokenizer": "data/tokenizer",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return index


def test_registry_resolves_alias_and_validates_identity(tmp_path: Path) -> None:
    registry = DatasetRegistry(tmp_path, _write_fixture(tmp_path))
    record = registry.resolve("fixture")
    assert record.name == "fixture_v1"
    assert record.train == tmp_path / "data/train.jsonl"
    assert len(registry.sha256) == 64
    registry.validate_identity(record)


def test_registry_rejects_unknown_and_identity_drift(tmp_path: Path) -> None:
    registry = DatasetRegistry(tmp_path, _write_fixture(tmp_path))
    with pytest.raises(KeyError, match="unknown dataset"):
        registry.resolve("missing")
    (tmp_path / "artifacts/audit.json").write_text(
        json.dumps(
            {
                "dataset_id": "other/data",
                "dataset_revision": "frozen",
                "status": "passed",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="disagrees with audit"):
        registry.validate_identity(registry.resolve("fixture"))
