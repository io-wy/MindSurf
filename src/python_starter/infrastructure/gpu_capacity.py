"""Capacity-aware admission and atomic leases for a shared single GPU."""

from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast


@dataclass(frozen=True, slots=True)
class GpuProcess:
    pid: int
    memory_mib: int
    username: str | None = None


@dataclass(frozen=True, slots=True)
class GpuSnapshot:
    index: int
    name: str
    total_mib: int
    used_mib: int
    free_mib: int
    utilization_percent: int
    processes: tuple[GpuProcess, ...]


def _username(pid: int) -> str | None:
    try:
        return (
            Path(f"/proc/{pid}/status").read_text(encoding="utf-8").split("Uid:", 1)[1].split()[0]
        )
    except (FileNotFoundError, IndexError, PermissionError):
        return None


def query_gpu_snapshot(index: int = 0) -> GpuSnapshot:
    """Read one coherent-enough capacity snapshot from nvidia-smi."""
    gpu = subprocess.run(
        [
            "nvidia-smi",
            f"--id={index}",
            "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    values = [value.strip() for value in gpu.split(",")]
    if len(values) != 6:
        raise RuntimeError(f"unexpected nvidia-smi GPU row: {gpu!r}")
    process_output = subprocess.run(
        [
            "nvidia-smi",
            f"--id={index}",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    processes = []
    for row in process_output.splitlines():
        if not row.strip():
            continue
        pid_value, memory_value = (value.strip() for value in row.split(",", 1))
        pid = int(pid_value)
        processes.append(GpuProcess(pid=pid, memory_mib=int(memory_value), username=_username(pid)))
    return GpuSnapshot(
        index=int(values[0]),
        name=values[1],
        total_mib=int(values[2]),
        used_mib=int(values[3]),
        free_mib=int(values[4]),
        utilization_percent=int(values[5]),
        processes=tuple(processes),
    )


def capacity_decision(
    snapshot: GpuSnapshot,
    *,
    required_mib: int,
    safety_margin_mib: int,
    reservations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Calculate admission without double-counting leased processes."""
    if required_mib <= 0 or safety_margin_mib < 0:
        raise ValueError("GPU memory requirement must be positive and safety margin non-negative")
    observed = {process.pid: process.memory_mib for process in snapshot.processes}
    leased_child_pids = {
        int(value["child_pid"]) for value in reservations if isinstance(value.get("child_pid"), int)
    }
    external_used = max(
        0,
        snapshot.used_mib
        - sum(memory for pid, memory in observed.items() if pid in leased_child_pids),
    )
    reserved = sum(
        max(
            int(value["required_mib"]),
            observed.get(int(value.get("child_pid") or -1), 0),
        )
        for value in reservations
    )
    demand = external_used + reserved + required_mib + safety_margin_mib
    return {
        "admitted": demand <= snapshot.total_mib,
        "gpu_index": snapshot.index,
        "total_mib": snapshot.total_mib,
        "external_used_mib": external_used,
        "active_reserved_mib": reserved,
        "required_mib": required_mib,
        "safety_margin_mib": safety_margin_mib,
        "projected_total_mib": demand,
        "remaining_mib": snapshot.total_mib - demand,
        "processes": [asdict(process) for process in snapshot.processes],
    }


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


class GpuLeaseStore:
    """Serialize capacity decisions and retain reservations across launch races."""

    def __init__(self, state_path: Path) -> None:
        self.state_path = state_path
        self.lock_path = state_path.with_suffix(f"{state_path.suffix}.lock")

    @contextmanager
    def _locked(self, timeout_seconds: float = 30.0) -> Iterator[None]:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                self.lock_path.mkdir()
                break
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"timed out waiting for GPU lease lock: {self.lock_path}"
                    ) from None
                time.sleep(0.05)
        try:
            yield
        finally:
            self.lock_path.rmdir()

    def _read(self) -> dict[str, Any]:
        if not self.state_path.is_file():
            return {"schema_version": 1, "leases": []}
        payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1 or not isinstance(payload.get("leases"), list):
            raise ValueError("invalid GPU lease state")
        return cast(dict[str, Any], payload)

    def _write(self, payload: dict[str, Any]) -> None:
        temporary = self.state_path.with_name(f".{self.state_path.name}.{uuid.uuid4().hex}.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.state_path)

    @staticmethod
    def _live_leases(payload: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            value
            for value in payload["leases"]
            if isinstance(value, dict)
            and isinstance(value.get("owner_pid"), int)
            and _pid_alive(int(value["owner_pid"]))
        ]

    def try_acquire(
        self,
        *,
        required_mib: int,
        safety_margin_mib: int,
        run_id: str,
        gpu_index: int = 0,
    ) -> tuple[str | None, dict[str, Any]]:
        with self._locked():
            payload = self._read()
            leases = self._live_leases(payload)
            snapshot = query_gpu_snapshot(gpu_index)
            # A reservation only constrains the card it was taken on. Counting
            # every live lease against every card makes the second arm of a
            # two-card host wait on memory that is reserved elsewhere.
            decision = capacity_decision(
                snapshot,
                required_mib=required_mib,
                safety_margin_mib=safety_margin_mib,
                reservations=[
                    value for value in leases if int(value.get("gpu_index", 0)) == gpu_index
                ],
            )
            if decision["admitted"]:
                lease_id = uuid.uuid4().hex
                leases.append(
                    {
                        "lease_id": lease_id,
                        "run_id": run_id,
                        "owner_pid": os.getpid(),
                        "child_pid": None,
                        "required_mib": required_mib,
                        "gpu_index": gpu_index,
                        "created_unix": time.time(),
                    }
                )
                payload["leases"] = leases
                self._write(payload)
                return lease_id, decision
            payload["leases"] = leases
            self._write(payload)
            return None, decision

    def set_child_pid(self, lease_id: str, child_pid: int) -> None:
        with self._locked():
            payload = self._read()
            leases = self._live_leases(payload)
            for value in leases:
                if value.get("lease_id") == lease_id:
                    value["child_pid"] = child_pid
                    self._write({**payload, "leases": leases})
                    return
            raise KeyError(f"unknown live GPU lease: {lease_id}")

    def release(self, lease_id: str) -> None:
        with self._locked():
            payload = self._read()
            payload["leases"] = [
                value for value in self._live_leases(payload) if value.get("lease_id") != lease_id
            ]
            self._write(payload)
