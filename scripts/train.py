"""Unified training entry point driven by Hydra configuration.

Usage:
    uv run scripts/train.py training=pretrain model=minimind
    uv run scripts/train.py training=sft model=minimind_small data=default
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, cast

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset, IterableDataset

# Allow importing src/python_starter as top-level package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from python_starter.core.data_contract import sha256_file, verify_training_view_manifest
from python_starter.core.dataset import (
    JsonlPackedDataset,
    PackedBlockDataset,
    SFTDataset,
    collate_fn,
)
from python_starter.core.model import ModelConfig, TransformerLM
from python_starter.core.tokenizer import load_tokenizer
from python_starter.core.trainer import Trainer, TrainerConfig
from python_starter.core.utils import count_parameters, format_number, set_seed
from python_starter.experiments.tracker import ExperimentTracker
from python_starter.infrastructure.config import get_settings
from python_starter.infrastructure.logging import get_logger

logger = get_logger(__name__)


def _plain_mapping(config: DictConfig) -> dict[str, Any]:
    value = OmegaConf.to_container(config, resolve=True)
    if not isinstance(value, dict):
        raise TypeError("expected a mapping configuration")
    return cast(dict[str, Any], value)


def _load_audit(path: Path, dataset_id: str, revision: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"dataset audit is required before training: {path}. "
            "Run scripts/audit_pretrain_dataset.py first."
        )
    audit = json.loads(path.read_text(encoding="utf-8"))
    required_gates = (
        "schema_valid",
        "metadata_contract_valid",
        "file_identity_valid",
        "validation_test_disjoint",
        "train_holdout_disjoint",
    )
    if audit.get("status") != "passed" or not all(
        audit.get("gates", {}).get(name) is True for name in required_gates
    ):
        raise ValueError("dataset audit has not passed all internal-training gates")
    if audit.get("dataset_id") != dataset_id or audit.get("dataset_revision") != revision:
        raise ValueError("dataset audit identity does not match the training configuration")
    return cast(dict[str, Any], audit)


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    """Run training with Hydra configuration."""
    settings = get_settings()
    resolved_config = _plain_mapping(cfg)

    logger.info("training_config", config=resolved_config)

    seed = cfg.get("seed", 42)
    set_seed(seed)

    tokenizer_path = Path(to_absolute_path(str(cfg.data.tokenizer_name)))
    tokenizer = load_tokenizer(str(tokenizer_path))

    model_cfg = ModelConfig(**_plain_mapping(cfg.model))
    model = TransformerLM(model_cfg)
    parameter_count = count_parameters(model)
    expected_parameters = cfg.get("expected_parameters")
    if expected_parameters is not None and parameter_count != int(expected_parameters):
        raise ValueError(
            f"model parameter count mismatch: expected {expected_parameters}, got {parameter_count}"
        )
    logger.info(
        "model_built",
        params=format_number(parameter_count),
        config=model_cfg.to_dict(),
    )

    dataset_type = cfg.get("dataset_type", "pretrain")
    train_dataset: Dataset[dict[str, torch.Tensor]] | IterableDataset[dict[str, torch.Tensor]]
    val_dataset: Dataset[dict[str, torch.Tensor]] | IterableDataset[dict[str, torch.Tensor]] | None
    if dataset_type == "sft":
        train_dataset = SFTDataset(
            to_absolute_path(str(cfg.data.train_path)),
            tokenizer,
            max_length=cfg.data.max_length,
        )
        val_dataset = None
        if cfg.data.get("val_path") and Path(to_absolute_path(str(cfg.data.val_path))).exists():
            val_dataset = SFTDataset(
                to_absolute_path(str(cfg.data.val_path)),
                tokenizer,
                max_length=cfg.data.max_length,
            )
    else:
        audit_path = Path(
            to_absolute_path(
                str(cfg.data.get("audit_path", "artifacts/data/mindsurf_team_v1/audit.json"))
            )
        )
        audit = _load_audit(
            audit_path,
            dataset_id=str(cfg.data.dataset_id),
            revision=str(cfg.data.dataset_revision),
        )
        resolved_config["dataset_audit"] = audit
        train_path = Path(to_absolute_path(str(cfg.data.train_path)))
        manifest_path = Path(to_absolute_path(str(cfg.data.training_view_manifest)))
        training_view = verify_training_view_manifest(
            manifest_path,
            train_path=train_path,
            dataset_id=str(cfg.data.dataset_id),
            revision=str(cfg.data.dataset_revision),
            source_sha256=str(audit["splits"]["train"]["sha256"]),
        )
        resolved_config["training_view"] = training_view
        pretokenized = cfg.data.get("pretokenized_path")
        if pretokenized:
            # Pre-tokenised blocks are byte-for-byte what the inline packer
            # would have produced, so the block cursor keeps its meaning and
            # exact resume is unaffected. shuffle_buffer is refused rather than
            # ignored: the blocks are fixed at build time, so accepting the
            # setting would silently drop a requested randomisation.
            if int(cfg.data.get("shuffle_buffer", 0)) > 1:
                raise ValueError(
                    "shuffle_buffer cannot apply to a pre-tokenised view; "
                    "shuffle the source view before pre-tokenising"
                )
            pretokenized_path = Path(to_absolute_path(str(pretokenized)))
            manifest = json.loads(
                Path(to_absolute_path(str(cfg.data.pretokenized_manifest))).read_text(
                    encoding="utf-8"
                )
            )
            declared = manifest.get("output", {})
            if sha256_file(pretokenized_path) != declared.get("sha256"):
                raise ValueError(
                    f"pre-tokenised block array does not match its manifest: {pretokenized_path}"
                )
            resolved_config["pretokenized"] = manifest.get("pretokenized")
            train_dataset = PackedBlockDataset(
                pretokenized_path,
                max_length=cfg.data.max_length,
                epochs=int(cfg.data.get("epochs", 1)),
            )
            # Fail here rather than after hours of training. A budget that
            # overruns the corpus raises "training data exhausted" only when the
            # loader actually runs dry, which for a one-epoch run means at the
            # very end: a 172,000-step request against 171,892 available steps
            # cost two GPUs seven hours before it surfaced.
            available_blocks = train_dataset.blocks_per_epoch * train_dataset.epochs
            required_blocks = int(cfg.training.max_steps) * int(cfg.training.batch_size) * int(
                cfg.training.get("accumulation_steps", 1)
            )
            if required_blocks > available_blocks:
                affordable = available_blocks // (
                    int(cfg.training.batch_size)
                    * int(cfg.training.get("accumulation_steps", 1))
                )
                raise ValueError(
                    f"training budget exceeds the corpus: {cfg.training.max_steps} steps need "
                    f"{required_blocks} blocks but {available_blocks} are available "
                    f"({train_dataset.blocks_per_epoch} per epoch x {train_dataset.epochs}). "
                    f"Reduce max_steps to at most {affordable}, or raise data.epochs."
                )
        else:
            train_dataset = JsonlPackedDataset(
                train_path,
                tokenizer,
                max_length=cfg.data.max_length,
                text_key=str(cfg.data.get("text_field", "text")),
                shuffle_buffer=int(cfg.data.get("shuffle_buffer", 0)),
                seed=int(seed),
                epochs=int(cfg.data.get("epochs", 1)),
            )
        val_path = Path(to_absolute_path(str(cfg.data.val_path)))
        val_dataset = (
            JsonlPackedDataset(
                val_path,
                tokenizer,
                max_length=cfg.data.max_length,
                text_key=str(cfg.data.get("text_field", "text")),
            )
            if val_path.is_file()
            else None
        )

    train_loader_generator = torch.Generator()
    train_loader_generator.manual_seed(int(seed))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=train_loader_generator,
    )
    val_loader = None
    if val_dataset:
        val_loader_generator = torch.Generator()
        val_loader_generator.manual_seed(int(seed) + 1)
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            generator=val_loader_generator,
        )

    trainer_cfg = TrainerConfig(**_plain_mapping(cfg.training))
    tracker = ExperimentTracker(
        settings,
        experiment_name=cfg.get("experiment_name", "default"),
        local_dir=trainer_cfg.output_dir / "tracking",
    )
    tracker.start(
        run_name=cfg.get("run_name", None),
        config=resolved_config,
    )

    try:
        trainer = Trainer(
            model,
            trainer_cfg,
            tracker=tracker,
            run_config=resolved_config,
        )
        resume_from = cfg.get("resume_from")
        init_from = cfg.get("init_from")
        if resume_from and init_from:
            raise ValueError("resume_from and init_from are mutually exclusive")
        if resume_from:
            trainer.load_checkpoint(to_absolute_path(str(resume_from)))
        if init_from:
            parent = Path(to_absolute_path(str(init_from)))
            resolved_config["parent_checkpoint"] = {
                "path": str(parent),
                "sha256": sha256_file(parent),
                "mode": "model_weights_only",
            }
            trainer.run_config = resolved_config
            trainer.load_model_weights(parent)
        trainer.train(train_loader, val_loader)
    finally:
        tracker.finish()
    logger.info("training_finished")


if __name__ == "__main__":
    main()
