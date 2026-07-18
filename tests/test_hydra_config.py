"""Hydra composition regression tests for grouped experiment configuration."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir


def test_default_and_dataset_arm_configs_remain_nested() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        official = compose(config_name="default")
        team = compose(config_name="default", overrides=["data=mindsurf_team_v1"])

    assert official.model.n_embed == 768
    assert official.training.max_steps == 10_000
    assert official.data.dataset_id == "gongjy/minimind_dataset"
    assert team.data.dataset_id == "wyywnab/mindsurf_pretrain_dataset"
    assert "_group_" not in official
