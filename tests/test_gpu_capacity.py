"""Shared-GPU capacity admission tests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import python_starter.infrastructure.gpu_capacity as gpu_capacity
from python_starter.infrastructure.gpu_capacity import (
    GpuLeaseStore,
    GpuProcess,
    GpuSnapshot,
    capacity_decision,
    query_gpu_snapshot,
)


def _snapshot(processes: tuple[GpuProcess, ...] = ()) -> GpuSnapshot:
    used = sum(process.memory_mib for process in processes)
    return GpuSnapshot(
        index=0,
        name="test-gpu",
        total_mib=24_564,
        used_mib=used,
        free_mib=24_564 - used,
        utilization_percent=0,
        processes=processes,
    )


def test_two_11g_runs_fit_with_margin() -> None:
    decision = capacity_decision(
        _snapshot((GpuProcess(pid=10, memory_mib=9700),)),
        required_mib=11_000,
        safety_margin_mib=1536,
        reservations=[{"required_mib": 11_000, "child_pid": 10}],
    )
    assert decision["admitted"] is True
    assert decision["projected_total_mib"] == 23_536


def test_external_process_and_third_run_are_rejected() -> None:
    processes = (
        GpuProcess(pid=10, memory_mib=9700),
        GpuProcess(pid=20, memory_mib=4000, username="other"),
    )
    decision = capacity_decision(
        _snapshot(processes),
        required_mib=11_000,
        safety_margin_mib=1536,
        reservations=[{"required_mib": 11_000, "child_pid": 10}],
    )
    assert decision["admitted"] is False
    assert decision["external_used_mib"] == 4000


def test_snapshot_parsing_and_invalid_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    outputs = iter(
        [
            "0, RTX 4090, 24564, 9700, 14864, 75\n",
            "123, 9700\n",
        ]
    )
    monkeypatch.setattr(
        "python_starter.infrastructure.gpu_capacity.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(stdout=next(outputs)),
    )
    monkeypatch.setattr(gpu_capacity, "_username", lambda pid: f"user-{pid}")
    snapshot = query_gpu_snapshot()
    assert snapshot.processes == (GpuProcess(pid=123, memory_mib=9700, username="user-123"),)
    with pytest.raises(ValueError, match="positive"):
        capacity_decision(snapshot, required_mib=0, safety_margin_mib=0, reservations=[])


def test_lease_store_acquire_update_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gpu_capacity, "query_gpu_snapshot", lambda _: _snapshot())
    monkeypatch.setattr(gpu_capacity, "_pid_alive", lambda _: True)
    store = GpuLeaseStore(tmp_path / "leases.json")
    lease_id, decision = store.try_acquire(
        required_mib=11_000,
        safety_margin_mib=1536,
        run_id="run",
    )
    assert lease_id is not None
    assert decision["admitted"] is True
    store.set_child_pid(lease_id, 123)
    payload = json.loads((tmp_path / "leases.json").read_text(encoding="utf-8"))
    assert payload["leases"][0]["child_pid"] == 123
    store.release(lease_id)
    assert json.loads((tmp_path / "leases.json").read_text(encoding="utf-8"))["leases"] == []


def test_lease_on_another_card_does_not_block_this_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reservation constrains only the card it was taken on."""
    from python_starter.infrastructure import gpu_capacity

    store = gpu_capacity.GpuLeaseStore(tmp_path / "leases.json")

    def snapshot_for(index: int = 0) -> gpu_capacity.GpuSnapshot:
        return gpu_capacity.GpuSnapshot(
            index=index,
            name="NVIDIA GeForce RTX 4090",
            total_mib=23028,
            used_mib=0,
            free_mib=23028,
            utilization_percent=0,
            processes=(),
        )

    monkeypatch.setattr(gpu_capacity, "query_gpu_snapshot", snapshot_for)

    first, first_decision = store.try_acquire(
        required_mib=11000, safety_margin_mib=1536, run_id="arm-0", gpu_index=0
    )
    second, second_decision = store.try_acquire(
        required_mib=11000, safety_margin_mib=1536, run_id="arm-1", gpu_index=1
    )
    third, third_decision = store.try_acquire(
        required_mib=11000, safety_margin_mib=1536, run_id="arm-2", gpu_index=0
    )

    assert first is not None and first_decision["admitted"]
    assert second is not None and second_decision["admitted"]
    # Two 11 GiB arms plus margin do not fit on one 23 GiB card.
    assert third is None and not third_decision["admitted"]
