from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import torch
import pytest


def test_training_checkpoint_module_exists() -> None:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "experiments"
        / "pretrain"
        / "training_checkpoint.py"
    )

    assert importlib.util.spec_from_file_location("training_checkpoint", module_path) is not None
    assert module_path.exists()


def test_checkpoint_roundtrip_restores_training_and_rng_state(tmp_path: Path) -> None:
    from experiments.pretrain.training_checkpoint import (
        load_training_checkpoint,
        save_training_checkpoint,
    )

    random.seed(17)
    torch.manual_seed(23)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    optimizer.zero_grad(set_to_none=True)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    saved_weight = model.weight.detach().clone()
    checkpoint_path = tmp_path / "training_state.pt"
    progress = {
        "global_step": 3,
        "optimizer_step": 3,
        "consumed_blocks": 12,
        "epoch": 0,
    }
    run_config = {"batch_size": 4, "max_seq_len": 128, "seed": 17}

    save_training_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scaler=None,
        progress=progress,
        run_config=run_config,
    )
    expected_python_random = random.random()
    expected_torch_random = torch.rand(1)

    with torch.no_grad():
        model.weight.fill_(99)
    random.seed(999)
    torch.manual_seed(999)

    restored = load_training_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scaler=None,
    )

    assert checkpoint_path.exists()
    assert not checkpoint_path.with_suffix(".pt.tmp").exists()
    assert restored["schema_version"] == 1
    assert restored["progress"] == progress
    assert restored["run_config"] == run_config
    assert torch.equal(model.weight, saved_weight)
    assert random.random() == expected_python_random
    assert torch.equal(torch.rand(1), expected_torch_random)


def test_resume_rejects_incompatible_training_shape() -> None:
    from experiments.pretrain.training_checkpoint import validate_resume_config

    saved = {
        "hidden_size": 768,
        "num_hidden_layers": 8,
        "batch_size": 8,
        "max_seq_len": 512,
    }
    current = {**saved, "max_seq_len": 1024}

    with pytest.raises(ValueError, match="max_seq_len"):
        validate_resume_config(saved, current)


def test_checkpoint_stages_serialized_state_on_cpu_before_restore(tmp_path: Path) -> None:
    from experiments.pretrain.training_checkpoint import (
        load_training_checkpoint,
        save_training_checkpoint,
    )

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    checkpoint_path = tmp_path / "state.pt"
    save_training_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scaler=None,
        progress={"global_step": 0, "optimizer_step": 0, "consumed_blocks": 0, "epoch": 0},
        run_config={},
    )

    restored = load_training_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        scaler=None,
        map_location="meta",
    )

    assert restored["progress"]["global_step"] == 0
