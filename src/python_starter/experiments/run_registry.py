"""Atomic local lifecycle registry for idempotent training and evaluation runs."""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

from python_starter.infrastructure.gpu_capacity import _pid_alive

TERMINAL_STATUSES = {"completed", "failed", "rejected"}
VALID_STATUSES = {"queued", "running", "failed", "resumed", "completed", "rejected"}
ACTIVE_STATUSES = {"queued", "running", "resumed"}


class RunRegistry:
    """Persist one current record and an append-only transition history per run ID."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock_path = path.with_suffix(f"{path.suffix}.lock")

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + 30
        while True:
            try:
                self.lock_path.mkdir()
                break
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"timed out waiting for run registry: {self.lock_path}"
                    ) from None
                time.sleep(0.05)
        try:
            yield
        finally:
            self.lock_path.rmdir()

    def _read(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {"schema_version": 1, "runs": {}}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1 or not isinstance(payload.get("runs"), dict):
            raise ValueError("invalid run registry")
        return cast(dict[str, Any], payload)

    def _write(self, payload: dict[str, Any]) -> None:
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.path)

    def transition(
        self,
        run_id: str,
        status: str,
        *,
        detail: dict[str, Any] | None = None,
        allow_retry: bool = False,
    ) -> dict[str, Any]:
        if not run_id or status not in VALID_STATUSES:
            raise ValueError("run_id and status must be valid")
        with self._locked():
            payload = self._read()
            previous = payload["runs"].get(run_id)
            history: list[dict[str, Any]] = (
                list(previous.get("history", [])) if isinstance(previous, dict) else []
            )
            if previous and status == "queued":
                previous_status = previous.get("status")
                if previous_status == "completed":
                    raise ValueError(f"duplicate active or completed run: {run_id}")
                if previous_status in ACTIVE_STATUSES and not allow_retry:
                    # An active record whose process is gone was abandoned, not
                    # duplicated. Blocking it would force allow_retry, which also
                    # permits clobbering a genuinely live run.
                    owner_pid = previous.get("pid")
                    if isinstance(owner_pid, int) and _pid_alive(owner_pid):
                        raise ValueError(f"duplicate active or completed run: {run_id}")
                    history.append(
                        {
                            "status": "failed",
                            "unix_time": time.time(),
                            "pid": owner_pid,
                            "detail": {
                                "reason": "abandoned: owner process is no longer running",
                                "previous_status": previous_status,
                            },
                        }
                    )
            event = {
                "status": status,
                "unix_time": time.time(),
                "pid": os.getpid(),
                "detail": detail or {},
            }
            history.append(event)
            record = {"run_id": run_id, **event, "history": history}
            payload["runs"][run_id] = record
            self._write(payload)
            return record

    def get(self, run_id: str) -> dict[str, Any] | None:
        with self._locked():
            value = self._read()["runs"].get(run_id)
            return value if isinstance(value, dict) else None
