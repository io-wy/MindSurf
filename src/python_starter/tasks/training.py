"""Celery entry point for the same audited Hydra training CLI used locally."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from python_starter.infrastructure.logging import get_logger
from python_starter.tasks.celery_app import celery_app

logger = get_logger(__name__)
ROOT = Path(__file__).resolve().parents[3]


def _hydra_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def _flatten_overrides(
    values: Mapping[str, Any],
    prefix: str = "",
) -> list[str]:
    overrides: list[str] = []
    for key, value in sorted(values.items()):
        if not key.replace("_", "").isalnum():
            raise ValueError(f"invalid Hydra override key: {key}")
        qualified = f"{prefix}.{key}" if prefix else key
        if isinstance(value, Mapping):
            overrides.extend(_flatten_overrides(value, qualified))
        elif isinstance(value, (str, int, float, bool)) or value is None:
            overrides.append(f"{qualified}={_hydra_value(value)}")
        else:
            raise ValueError(f"unsupported override value for {qualified}")
    return overrides


@celery_app.task(bind=True)  # type: ignore[untyped-decorator]
def run_training_task(
    self: Any,
    experiment_name: str,
    config_overrides: dict[str, Any] | None = None,
    dataset_path: str | None = None,
) -> dict[str, Any]:
    """Execute one validated training run in a worker subprocess."""
    task_id = str(self.request.id)
    log_dir = ROOT / "artifacts" / "jobs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{task_id}.log"

    command = [
        sys.executable,
        str(ROOT / "scripts" / "train.py"),
        f"run_name={_hydra_value(experiment_name)}",
    ]
    command.extend(_flatten_overrides(config_overrides or {}))
    if dataset_path:
        resolved_dataset = Path(dataset_path).expanduser().resolve(strict=True)
        command.append(f"data.train_path={_hydra_value(resolved_dataset.as_posix())}")

    self.update_state(
        state="STARTED",
        meta={"experiment": experiment_name, "log_path": str(log_path)},
    )
    logger.info("training_task_started", task_id=task_id, experiment=experiment_name)
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"training command failed with exit code {completed.returncode}; see {log_path}"
        )
    return {
        "status": "completed",
        "experiment": experiment_name,
        "log_path": str(log_path),
    }
