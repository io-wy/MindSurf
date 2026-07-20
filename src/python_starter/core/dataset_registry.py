"""Resolve frozen pretraining datasets from one repository-local index."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from python_starter.core.data_contract import sha256_file


@dataclass(frozen=True, slots=True)
class DatasetRecord:
    """Paths and immutable identity for one audited training dataset."""

    name: str
    dataset_id: str
    revision: str
    hydra_config: str
    identity: str
    spec: Path
    audit: Path
    training_view_manifest: Path
    train: Path
    validation: Path
    test: Path
    tokenizer: Path

    @classmethod
    def from_mapping(
        cls,
        *,
        root: Path,
        name: str,
        value: dict[str, Any],
    ) -> DatasetRecord:
        required = (
            "dataset_id",
            "revision",
            "hydra_config",
            "identity",
            "spec",
            "audit",
            "training_view_manifest",
            "train",
            "validation",
            "test",
            "tokenizer",
        )
        missing = [key for key in required if not isinstance(value.get(key), str)]
        if missing:
            raise ValueError(f"dataset {name!r} is missing string fields: {missing}")
        return cls(
            name=name,
            dataset_id=value["dataset_id"],
            revision=value["revision"],
            hydra_config=value["hydra_config"],
            identity=value["identity"],
            spec=root / value["spec"],
            audit=root / value["audit"],
            training_view_manifest=root / value["training_view_manifest"],
            train=root / value["train"],
            validation=root / value["validation"],
            test=root / value["test"],
            tokenizer=root / value["tokenizer"],
        )


class DatasetRegistry:
    """Fail-closed loader for the shared dataset index."""

    def __init__(self, root: Path, index_path: Path) -> None:
        self.root = root.resolve()
        self.index_path = index_path.resolve()
        payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported dataset index schema")
        datasets = payload.get("datasets")
        aliases = payload.get("aliases", {})
        if not isinstance(datasets, dict) or not datasets:
            raise ValueError("dataset index must define datasets")
        if not isinstance(aliases, dict):
            raise ValueError("dataset aliases must be a mapping")
        self._records = {
            name: DatasetRecord.from_mapping(root=self.root, name=name, value=value)
            for name, value in datasets.items()
            if isinstance(name, str) and isinstance(value, dict)
        }
        if len(self._records) != len(datasets):
            raise ValueError("dataset index contains invalid records")
        self._aliases = aliases

    @property
    def sha256(self) -> str:
        return sha256_file(self.index_path)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._records))

    def resolve(self, value: str) -> DatasetRecord:
        name = self._aliases.get(value, value)
        if not isinstance(name, str) or name not in self._records:
            choices = ", ".join((*self.names(), *sorted(self._aliases)))
            raise KeyError(f"unknown dataset {value!r}; expected one of: {choices}")
        return self._records[name]

    def validate_identity(self, record: DatasetRecord) -> None:
        spec = json.loads(record.spec.read_text(encoding="utf-8"))
        audit = json.loads(record.audit.read_text(encoding="utf-8"))
        if spec.get("dataset_id") != record.dataset_id or spec.get("revision") != record.revision:
            raise ValueError(f"dataset index identity disagrees with spec for {record.name}")
        if (
            audit.get("dataset_id") != record.dataset_id
            or audit.get("dataset_revision") != record.revision
        ):
            raise ValueError(f"dataset index identity disagrees with audit for {record.name}")
        if audit.get("status") != "passed":
            raise ValueError(f"dataset audit is not passing for {record.name}")
