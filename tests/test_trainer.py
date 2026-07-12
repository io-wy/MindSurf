"""Trainer tests."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from python_starter.core.model import ModelConfig, TransformerLM
from python_starter.core.trainer import Trainer, TrainerConfig


class _DummyDataset(Dataset):
    """Minimal dataset matching the expected batch format."""

    def __init__(self, num_samples: int, seq_len: int, vocab_size: int) -> None:
        self.input_ids = torch.randint(0, vocab_size, (num_samples, seq_len))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "labels": self.input_ids[idx].clone(),
        }


def _make_dummy_loader(batch_size: int = 2, seq_len: int = 16, num_batches: int = 3) -> DataLoader:
    """Create a dummy dataloader for testing."""
    dataset = _DummyDataset(num_batches * batch_size, seq_len, vocab_size=100)
    return DataLoader(dataset, batch_size=batch_size)


def test_trainer_save_load(tmp_path: Path) -> None:
    config = ModelConfig(vocab_size=100, n_embed=64, n_layer=2, n_head=4)
    model = TransformerLM(config)
    trainer_cfg = TrainerConfig(
        output_dir=str(tmp_path), num_epochs=1, batch_size=2, device="cpu"
    )
    trainer = Trainer(model, trainer_cfg)

    trainer.save_checkpoint("test.pt")
    assert (tmp_path / "test.pt").exists()

    trainer.load_checkpoint(tmp_path / "test.pt")
    assert trainer.global_step == 0
