from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def detect_source_revision(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def write_run_manifest(
    path: Path,
    *,
    run_name: str,
    config: dict[str, Any],
    source_revision: str | None,
    artifacts: dict[str, str],
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "run_name": run_name,
        "created_at": _utc_now(),
        "source_revision": source_revision,
        "config": config,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "artifacts": artifacts,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)
    return payload


class ExperimentTracker:
    def __init__(
        self,
        path: Path,
        *,
        run_name: str,
        config: dict[str, Any],
        trackio_project: str | None = None,
        trackio_space_id: str | None = None,
    ) -> None:
        self.path = path
        self.run_name = run_name
        self.trackio = None
        self.finished = False
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if trackio_project:
            try:
                import trackio
            except ImportError as exc:
                raise RuntimeError(
                    "Trackio logging was requested but trackio is not installed; "
                    "install experiments/pretrain/requirements-infra.txt"
                ) from exc
            init_kwargs = {
                "project": trackio_project,
                "name": run_name,
                "config": config,
            }
            if trackio_space_id:
                init_kwargs["space_id"] = trackio_space_id
            trackio.init(**init_kwargs)
            self.trackio = trackio

        self._append({"event": "start", "run_name": run_name, "config": config})

    def _append(self, row: dict[str, Any]) -> None:
        record = {"timestamp": _utc_now(), **row}
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def log(self, metrics: dict[str, int | float], *, step: int) -> None:
        self._append({"event": "metrics", "step": step, "metrics": metrics})
        if self.trackio is not None:
            self.trackio.log(metrics, step=step)

    def alert(self, title: str, text: str, *, level: str = "warn", step: int | None = None) -> None:
        normalized = level.lower()
        if normalized not in {"info", "warn", "error"}:
            raise ValueError(f"unsupported alert level: {level}")
        self._append(
            {
                "event": "alert",
                "step": step,
                "level": normalized,
                "title": title,
                "text": text,
            }
        )
        if self.trackio is not None:
            trackio_level = {
                "info": self.trackio.AlertLevel.INFO,
                "warn": self.trackio.AlertLevel.WARN,
                "error": self.trackio.AlertLevel.ERROR,
            }[normalized]
            self.trackio.alert(title=title, text=text, level=trackio_level)

    def finish(self, *, status: str, summary: dict[str, Any]) -> None:
        if self.finished:
            return
        self._append({"event": "finish", "status": status, "summary": summary})
        if self.trackio is not None:
            self.trackio.finish()
        self.finished = True
