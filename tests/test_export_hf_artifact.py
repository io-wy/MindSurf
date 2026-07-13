from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModel, AutoModelForCausalLM

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM, MiniMindModel


def test_exported_artifact_loads_through_auto_model(tmp_path: Path) -> None:
    from experiments.pretrain.scripts.export_hf_artifact import export_checkpoint

    config = MiniMindConfig(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        max_position_embeddings=128,
    )
    source_model = MiniMindForCausalLM(config).eval()
    weight_path = tmp_path / "tiny.pth"
    torch.save(source_model.state_dict(), weight_path)
    output_dir = tmp_path / "artifact"

    manifest = export_checkpoint(
        weight_path=weight_path,
        output_dir=output_dir,
        tokenizer_path=Path("model"),
        config=config,
        source_revision="test-revision",
    )
    loaded = AutoModelForCausalLM.from_pretrained(
        output_dir,
        trust_remote_code=True,
        local_files_only=True,
    ).eval()
    loaded_base = AutoModel.from_pretrained(
        output_dir,
        trust_remote_code=True,
        local_files_only=True,
    ).eval()
    input_ids = torch.tensor([[1, 2, 3]])

    with torch.no_grad():
        expected = source_model(input_ids).logits
        actual = loaded(input_ids).logits
        expected_hidden = source_model.model(input_ids)[0]
        actual_hidden = loaded_base(input_ids)[0]

    assert torch.equal(actual, expected)
    assert torch.equal(actual_hidden, expected_hidden)
    assert MiniMindModel._supports_attention_backend is True
    assert (output_dir / "model.safetensors").exists()
    assert (output_dir / "tokenizer.json").exists()
    assert manifest["source_revision"] == "test-revision"
    assert manifest["parameter_count"] == sum(parameter.numel() for parameter in source_model.parameters())
    assert manifest["source_weight"]["sha256"]
    artifact_config = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
    assert artifact_config["auto_map"]["AutoModel"].endswith(".MiniMindModel")
    assert json.loads((output_dir / "artifact_manifest.json").read_text(encoding="utf-8")) == manifest


def test_export_preserves_source_weight_dtype(tmp_path: Path) -> None:
    from experiments.pretrain.scripts.export_hf_artifact import export_checkpoint

    config = MiniMindConfig(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        max_position_embeddings=128,
    )
    source_model = MiniMindForCausalLM(config).half().eval()
    weight_path = tmp_path / "tiny-fp16.pth"
    torch.save(source_model.state_dict(), weight_path)
    output_dir = tmp_path / "artifact-fp16"

    export_checkpoint(
        weight_path=weight_path,
        output_dir=output_dir,
        tokenizer_path=Path("model"),
        config=config,
        source_revision="test-revision",
    )
    stored_tensors = load_file(output_dir / "model.safetensors")

    assert next(iter(stored_tensors.values())).dtype == torch.float16
