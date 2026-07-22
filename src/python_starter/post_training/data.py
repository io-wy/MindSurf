"""Load and validate local SFT and preference JSONL data."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class PostTrainingDataError(ValueError):
    """Raised when post-training data does not match the expected contract."""


@dataclass(frozen=True)
class PostTrainingDatasets:
    """Training and optional evaluation datasets passed to TRL."""

    train: Any
    eval: Any = None


def _read_jsonl(path: str | Path) -> list[tuple[int, dict[str, Any]]]:
    source = Path(path)
    if not source.is_file():
        raise PostTrainingDataError(f"Dataset file does not exist: {source}")

    records: list[tuple[int, dict[str, Any]]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PostTrainingDataError(
                    f"Invalid JSON at {source.name}:{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(value, dict):
                raise PostTrainingDataError(
                    f"Expected a JSON object at {source.name}:{line_number}"
                )
            records.append((line_number, value))

    if not records:
        raise PostTrainingDataError(f"Dataset file is empty: {source}")
    return records


def _require_fields(
    record: dict[str, Any],
    fields: tuple[str, ...],
    source: Path,
    line_number: int,
) -> None:
    missing = [field for field in fields if field not in record]
    if missing:
        names = ", ".join(missing)
        raise PostTrainingDataError(
            f"Missing required field(s) {names} at {source.name}:{line_number}"
        )


def load_sft_records(path: str | Path) -> list[dict[str, Any]]:
    """Load SFT JSONL and normalize ``response`` to ``completion``."""
    source = Path(path)
    normalized: list[dict[str, Any]] = []

    for line_number, record in _read_jsonl(source):
        item = dict(record)
        if "messages" in item or "text" in item:
            normalized.append(item)
            continue

        if "response" in item and "completion" not in item:
            item["completion"] = item.pop("response")
        _require_fields(item, ("prompt", "completion"), source, line_number)
        normalized.append(item)

    return normalized


def load_preference_records(path: str | Path) -> list[dict[str, Any]]:
    """Load explicit-prompt DPO preference data."""
    source = Path(path)
    normalized: list[dict[str, Any]] = []

    for line_number, record in _read_jsonl(source):
        _require_fields(record, ("prompt", "chosen", "rejected"), source, line_number)
        normalized.append(dict(record))

    return normalized


def load_post_training_datasets(
    stage: str,
    train_path: str | Path,
    eval_path: str | Path | None = None,
    dataset_cls: type[Any] | None = None,
) -> PostTrainingDatasets:
    """Load validated records into Hugging Face ``Dataset`` objects."""
    normalized_stage = stage.strip().lower()
    if normalized_stage == "sft":
        loader = load_sft_records
    elif normalized_stage == "dpo":
        loader = load_preference_records
    else:
        raise PostTrainingDataError(f"Unsupported stage {stage!r}; expected one of: dpo, sft")

    if dataset_cls is None:
        from datasets import Dataset

        dataset_cls = Dataset

    train_dataset = dataset_cls.from_list(loader(train_path))
    eval_dataset = None
    if eval_path is not None:
        eval_dataset = dataset_cls.from_list(loader(eval_path))
    return PostTrainingDatasets(train=train_dataset, eval=eval_dataset)
