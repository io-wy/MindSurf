"""Trainer tests."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, IterableDataset

from python_starter.core.model import ModelConfig, TransformerLM
from python_starter.core.trainer import Trainer, TrainerConfig, get_wsd_schedule


class _CursorDataset(IterableDataset[dict[str, torch.Tensor]]):
    def __init__(self, samples: torch.Tensor, limit: int | None = None) -> None:
        self.samples = samples
        self.limit = limit
        self.skip_blocks = 0

    def set_skip_blocks(self, value: int) -> None:
        self.skip_blocks = value

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        samples = self.samples[self.skip_blocks :]
        if self.limit is not None:
            samples = samples[: self.limit]
        for sample in samples:
            yield {"input_ids": sample, "labels": sample.clone()}


def _config(output_dir: Path) -> TrainerConfig:
    return TrainerConfig(
        output_dir=output_dir,
        max_steps=4,
        batch_size=2,
        accumulation_steps=1,
        warmup_steps=1,
        stable_ratio=0.5,
        save_every=2,
        logging_every=1,
        eval_every=0,
        device="cpu",
    )


def test_wsd_schedule_has_warmup_stable_and_decay() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter], lr=1.0)
    scheduler = get_wsd_schedule(
        optimizer,
        num_warmup_steps=2,
        num_training_steps=10,
        stable_ratio=0.5,
        min_lr_ratio=0.1,
    )

    values = []
    for _ in range(10):
        values.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    assert values[0] < values[1]
    assert values[2] == pytest.approx(1.0)
    assert values[5] == pytest.approx(1.0)
    assert values[-1] < values[6]


def test_atomic_checkpoint_contains_full_resume_state(tmp_path: Path) -> None:
    model = TransformerLM(ModelConfig(vocab_size=32, n_embed=16, n_layer=1, n_head=4))
    trainer = Trainer(model, _config(tmp_path))

    path = trainer.save_checkpoint("test.pt")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    assert checkpoint["schema_version"] == 2
    assert checkpoint["model_config"] == model.config.to_dict()
    assert checkpoint["progress"]["consumed_blocks"] == 0
    assert checkpoint["progress"]["consumed_tokens"] == 0
    assert not list(tmp_path.glob("*.tmp"))


def test_resume_matches_uninterrupted_training(tmp_path: Path) -> None:
    torch.manual_seed(11)
    samples = torch.randint(0, 32, (8, 8))
    model_config = ModelConfig(
        vocab_size=32,
        n_embed=16,
        n_layer=1,
        n_head=4,
        max_seq_len=8,
        dropout=0.0,
    )
    initial = TransformerLM(model_config)
    initial_state = {name: value.clone() for name, value in initial.state_dict().items()}

    baseline_model = TransformerLM(model_config)
    baseline_model.load_state_dict(initial_state)
    baseline = Trainer(baseline_model, _config(tmp_path / "baseline"))
    baseline.train(DataLoader(_CursorDataset(samples), batch_size=2, num_workers=0))

    partial_model = TransformerLM(model_config)
    partial_model.load_state_dict(initial_state)
    partial = Trainer(partial_model, _config(tmp_path / "partial"))
    with pytest.raises(RuntimeError, match="exhausted"):
        partial.train(DataLoader(_CursorDataset(samples, limit=4), batch_size=2, num_workers=0))
    checkpoint_path = tmp_path / "partial" / "checkpoint_step_2.pt"

    resumed_model = TransformerLM(model_config)
    resumed = Trainer(resumed_model, _config(tmp_path / "resumed"))
    resumed.load_checkpoint(checkpoint_path)
    resumed.train(DataLoader(_CursorDataset(samples), batch_size=2, num_workers=0))

    assert resumed.global_step == baseline.global_step == 4
    assert resumed.consumed_blocks == baseline.consumed_blocks == 8
    for name, value in baseline._model_for_state().state_dict().items():
        torch.testing.assert_close(value, resumed._model_for_state().state_dict()[name])


def test_training_with_validation_saves_best_checkpoint(tmp_path: Path) -> None:
    torch.manual_seed(4)
    model = TransformerLM(
        ModelConfig(
            vocab_size=32,
            n_embed=16,
            n_layer=1,
            n_head=4,
            max_seq_len=8,
        )
    )
    config = TrainerConfig(
        output_dir=tmp_path,
        max_steps=1,
        batch_size=2,
        warmup_steps=0,
        stable_ratio=0.0,
        eval_every=1,
        eval_batches=1,
        save_every=1,
        logging_every=1,
        device="cpu",
    )
    trainer = Trainer(model, config)
    samples = torch.randint(0, 32, (4, 8))
    train_loader = DataLoader(_CursorDataset(samples[:2]), batch_size=2, num_workers=0)
    eval_loader = DataLoader(_CursorDataset(samples[2:]), batch_size=2, num_workers=0)

    trainer.train(train_loader, eval_loader)

    assert trainer.best_eval_loss < float("inf")
    assert (tmp_path / "best_model.pt").is_file()
    assert (tmp_path / "final_model.pt").is_file()
    summary = (tmp_path / "training_summary.json").read_text(encoding="utf-8")
    assert '"consumed_tokens": 16' in summary
    assert '"global_step": 1' in summary
    assert '"parameter_count":' in summary
    assert '"tokens_per_second":' in summary


def test_trainer_and_schedule_validation() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.AdamW([parameter])
    with pytest.raises(ValueError, match="training_steps"):
        get_wsd_schedule(optimizer, 0, 0)
    with pytest.raises(ValueError, match="warmup"):
        get_wsd_schedule(optimizer, 2, 2)
    with pytest.raises(ValueError, match="stable_ratio"):
        get_wsd_schedule(optimizer, 0, 2, stable_ratio=2.0)
    with pytest.raises(ValueError, match="min_lr_ratio"):
        get_wsd_schedule(optimizer, 0, 2, min_lr_ratio=-1.0)
    with pytest.raises(ValueError, match="max_steps"):
        TrainerConfig(max_steps=0)
    with pytest.raises(ValueError, match="accumulation"):
        TrainerConfig(accumulation_steps=0)
    with pytest.raises(ValueError, match="batch_size"):
        TrainerConfig(batch_size=0)
    with pytest.raises(ValueError, match="save_every"):
        TrainerConfig(save_every=0)
