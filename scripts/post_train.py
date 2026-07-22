"""Isolated TRL entry point for SFT and DPO.

Examples:
    uv run --extra post-training accelerate launch \
        --config_file configs/accelerate/cpu.yaml \
        scripts/post_train.py \
        model.name_or_path=sshleifer/tiny-gpt2 \
        data.train_path=/path/to/sft.jsonl

    uv run --extra post-training accelerate launch \
        --config_file configs/accelerate/single_gpu.yaml \
        scripts/post_train.py \
        post_training=dpo \
        model.name_or_path=/path/to/sft-checkpoint/merged \
        data.train_path=/path/to/dpo.jsonl
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from python_starter.core.utils import set_seed
from python_starter.infrastructure.logging import get_logger
from python_starter.post_training.data import load_post_training_datasets
from python_starter.post_training.trainer import (
    create_post_training_runtime,
    run_post_training,
)

logger = get_logger(__name__)


def _as_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Resolved {name} configuration must be a mapping")
    return value


@hydra.main(version_base=None, config_path="../configs", config_name="post_training")
def main(cfg: DictConfig) -> None:
    """Run one isolated SFT or DPO job."""
    resolved = OmegaConf.to_container(cfg, resolve=True)
    config = _as_mapping(resolved, "post-training")
    data_config = _as_mapping(config.get("data"), "data")
    stage = str(config.get("stage", ""))

    logger.info("post_training_config", config=resolved)
    set_seed(int(config.get("seed", 42)))

    datasets = load_post_training_datasets(
        stage=stage,
        train_path=str(data_config["train_path"]),
        eval_path=data_config.get("eval_path"),
    )
    runtime = create_post_training_runtime(config, datasets)
    run_post_training(runtime)
    logger.info(
        "post_training_finished",
        stage=stage,
        output_dir=str(runtime.output_dir),
        merged_output_dir=(
            str(runtime.merged_output_dir) if runtime.merged_output_dir is not None else None
        ),
    )


if __name__ == "__main__":
    main()
