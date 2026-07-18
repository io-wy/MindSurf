"""Inference, tracking, registry, and task helper tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from python_starter.core.inference import (
    InferenceEngine,
    load_checkpoint_model,
    sha256_file,
)
from python_starter.core.model import ModelConfig, TransformerLM
from python_starter.experiments.registry import LocalCandidateRegistry
from python_starter.experiments.tracker import ExperimentTracker
from python_starter.infrastructure.config import Settings
from python_starter.tasks.training import _flatten_overrides, _hydra_value


class _Tokenizer:
    eos_token_id = 1
    bos_token_id = 2

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [3 + (ord(value) % 12) for value in text]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return "completion"


def _checkpoint(path: Path) -> tuple[Path, ModelConfig]:
    config = ModelConfig(
        vocab_size=16,
        n_embed=16,
        n_layer=1,
        n_head=4,
        max_seq_len=16,
    )
    model = TransformerLM(config)
    torch.save(
        {
            "model_config": config.to_dict(),
            "model_state_dict": model.state_dict(),
            "progress": {"global_step": 2},
        },
        path,
    )
    return path, config


def test_checkpoint_loading_and_inference_engine(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    checkpoint, config = _checkpoint(tmp_path / "model.pt")
    model, payload, device = load_checkpoint_model(checkpoint, "cpu")
    assert model.config == config
    assert payload["progress"]["global_step"] == 2
    assert device.type == "cpu"
    assert len(sha256_file(checkpoint)) == 64

    monkeypatch.setattr(
        "python_starter.core.inference.load_tokenizer",
        lambda *_: _Tokenizer(),
    )
    engine = InferenceEngine(checkpoint, tmp_path, "cpu")
    result = engine.generate(
        "hello",
        max_new_tokens=2,
        temperature=0,
        top_p=1.0,
    )
    assert result.text == "completion"
    assert result.input_tokens == 5
    assert result.output_tokens == 2
    assert result.generation_time_ms >= 0

    invalid = tmp_path / "invalid.pt"
    torch.save({"model_state_dict": {}}, invalid)
    with pytest.raises(ValueError, match="model_config"):
        load_checkpoint_model(invalid, "cpu")


def test_local_tracker_is_durable_and_resumable(tmp_path: Path) -> None:
    settings = Settings(
        env="test",
        mlflow_tracking_uri=None,
        wandb_api_key=None,
    )
    tracker = ExperimentTracker(settings, "experiment", local_dir=tmp_path)
    tracker.start("run", {"seed": 7})
    tracker.log_params({"batch": 2})
    tracker.log_metrics({"loss": 1.5}, step=1)
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("artifact", encoding="utf-8")
    tracker.log_artifact(str(artifact))
    tracker.finish()

    run_dir = tmp_path / "run"
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert manifest["config"]["seed"] == 7
    events = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [event["type"] for event in events] == [
        "params",
        "metrics",
        "artifact",
        "finish",
    ]

    resumed = ExperimentTracker(settings, "experiment", local_dir=tmp_path)
    resumed.start("run")
    resumed.finish()
    resumed_events = (run_dir / "metrics.jsonl").read_text(encoding="utf-8")
    assert '"type": "resume"' in resumed_events


def test_successful_local_registry_and_public_gate(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    evaluation = tmp_path / "evaluation.json"
    evaluation.write_text(
        json.dumps(
            {
                "gate": {
                    "internal_candidate_passed": True,
                    "public_release_passed": False,
                }
            }
        ),
        encoding="utf-8",
    )
    registry = LocalCandidateRegistry(tmp_path / "registry.json")
    record = registry.register(
        name="candidate",
        checkpoint_path=checkpoint,
        evaluation_path=evaluation,
    )
    assert record["stage"] == "candidate"
    with pytest.raises(ValueError, match="not eligible"):
        registry.promote_public(str(record["checkpoint_sha256"]))
    with pytest.raises(ValueError, match="already registered"):
        registry.register(
            name="duplicate",
            checkpoint_path=checkpoint,
            evaluation_path=evaluation,
        )
    with pytest.raises(KeyError):
        registry.promote_public("0" * 64)

    registry_data = json.loads((tmp_path / "registry.json").read_text(encoding="utf-8"))
    registry_data["models"][0]["public_release_eligible"] = True
    (tmp_path / "registry.json").write_text(json.dumps(registry_data), encoding="utf-8")
    promoted = registry.promote_public(str(record["checkpoint_sha256"]))
    assert promoted["stage"] == "public"


def test_celery_hydra_override_helpers_fail_closed() -> None:
    assert _hydra_value(True) == "true"
    assert _hydra_value(None) == "null"
    assert _hydra_value(3) == "3"
    assert _hydra_value("text") == '"text"'
    assert _flatten_overrides({"training": {"max_steps": 2}, "debug": False}) == [
        "debug=false",
        "training.max_steps=2",
    ]
    with pytest.raises(ValueError, match="invalid"):
        _flatten_overrides({"bad-key": 1})
    with pytest.raises(ValueError, match="unsupported"):
        _flatten_overrides({"value": [1, 2]})
