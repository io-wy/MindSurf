"""Pin the upstream-anchor port.

These checks stand in for the one-off proof that the port reproduces upstream's
own logits bit-for-bit (max |difference| 0.0 in float32, upstream commit
83e52f6). That proof needs upstream's source, which is not vendored here, so
what is pinned permanently is the machinery the proof depended on: the norms
really disappear, the key remap is exhaustive in both directions, and the
inferred configuration reproduces upstream's published parameter counts.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from python_starter.core.model import ModelConfig, TransformerLM  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "evaluate_external_anchor", ROOT / "scripts" / "evaluate_external_anchor.py"
)
assert _SPEC is not None and _SPEC.loader is not None
anchor = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(anchor)

# Upstream's released shapes: (layers, hidden, mlp hidden, kv rows) and the
# parameter count upstream advertises for each release.
UPSTREAM_RELEASES = {
    "pretrain_512": ((8, 512, 1408, 128), 25_829_888),
    "pretrain_768": ((16, 768, 2048, 192), 104_030_976),
}


def upstream_state_dict(
    n_layer: int, n_embed: int, hidden_dim: int, kv_rows: int, vocab_size: int = 6400
) -> dict[str, torch.Tensor]:
    """Build a state dict with upstream's key names and shapes, on the meta device."""

    def tensor(*shape: int) -> torch.Tensor:
        return torch.empty(shape, device="meta")

    state_dict = {
        "model.embed_tokens.weight": tensor(vocab_size, n_embed),
        "model.norm.weight": tensor(n_embed),
        "lm_head.weight": tensor(vocab_size, n_embed),
    }
    for index in range(n_layer):
        state_dict.update(
            {
                f"model.layers.{index}.input_layernorm.weight": tensor(n_embed),
                f"model.layers.{index}.post_attention_layernorm.weight": tensor(n_embed),
                f"model.layers.{index}.self_attn.q_proj.weight": tensor(n_embed, n_embed),
                f"model.layers.{index}.self_attn.k_proj.weight": tensor(kv_rows, n_embed),
                f"model.layers.{index}.self_attn.v_proj.weight": tensor(kv_rows, n_embed),
                f"model.layers.{index}.self_attn.o_proj.weight": tensor(n_embed, n_embed),
                f"model.layers.{index}.mlp.gate_proj.weight": tensor(hidden_dim, n_embed),
                f"model.layers.{index}.mlp.up_proj.weight": tensor(hidden_dim, n_embed),
                f"model.layers.{index}.mlp.down_proj.weight": tensor(n_embed, hidden_dim),
            }
        )
    return state_dict


def test_qk_norm_false_removes_the_norms() -> None:
    """A disabled QK norm must hold no parameters, not merely be skipped."""
    config = ModelConfig(n_embed=64, n_layer=2, n_head=4, max_seq_len=32, qk_norm=False)
    state_dict = TransformerLM(config).state_dict()
    assert not [key for key in state_dict if "q_norm" in key or "k_norm" in key]

    enabled = TransformerLM(ModelConfig(n_embed=64, n_layer=2, n_head=4, max_seq_len=32))
    assert "blocks.0.attn.q_norm.weight" in enabled.state_dict()


def test_qk_norm_changes_the_function() -> None:
    """Guard against the flag becoming decorative: the two models must differ."""
    torch.manual_seed(0)
    kwargs = {"vocab_size": 64, "n_embed": 64, "n_layer": 2, "n_head": 4, "max_seq_len": 32}
    with_norm = TransformerLM(ModelConfig(**kwargs)).eval()
    without_norm = TransformerLM(ModelConfig(**kwargs, qk_norm=False)).eval()
    shared = {
        key: value
        for key, value in with_norm.state_dict().items()
        if ".q_norm." not in key and ".k_norm." not in key
    }
    without_norm.load_state_dict(shared, strict=True)

    input_ids = torch.randint(0, 64, (1, 16))
    with torch.inference_mode():
        assert not torch.allclose(with_norm(input_ids)[0], without_norm(input_ids)[0])


def test_default_config_omits_qk_norm_from_checkpoints() -> None:
    """Existing checkpoints must keep comparing equal to a freshly built config."""
    assert "qk_norm" not in ModelConfig().to_dict()
    assert ModelConfig(qk_norm=False).to_dict()["qk_norm"] is False
    assert ModelConfig.from_dict(ModelConfig().to_dict()).qk_norm is True


@pytest.mark.parametrize("release", sorted(UPSTREAM_RELEASES))
def test_remap_is_exhaustive_both_ways(release: str) -> None:
    """Every upstream tensor lands somewhere, and every slot of ours is filled."""
    (n_layer, n_embed, hidden_dim, kv_rows), _ = UPSTREAM_RELEASES[release]
    state_dict = upstream_state_dict(n_layer, n_embed, hidden_dim, kv_rows)

    remapped = anchor.remap_state_dict(state_dict)
    assert len(remapped) == len(state_dict)

    with torch.device("meta"):
        model = TransformerLM(anchor.infer_config(state_dict))
    missing, unexpected = model.load_state_dict(remapped, strict=False, assign=True)
    assert not missing and not unexpected


def test_remap_rejects_an_unknown_key() -> None:
    """A key we do not understand must fail loudly, never be dropped."""
    state_dict = upstream_state_dict(1, 64, 128, 32)
    state_dict["model.layers.0.self_attn.q_norm.weight"] = torch.empty(8, device="meta")
    with pytest.raises(ValueError, match="unrecognised upstream checkpoint key"):
        anchor.remap_state_dict(state_dict)


@pytest.mark.parametrize("release", sorted(UPSTREAM_RELEASES))
def test_inferred_config_reproduces_upstream_parameter_count(release: str) -> None:
    """The port must be the size upstream says it is, not merely load cleanly."""
    (n_layer, n_embed, hidden_dim, kv_rows), expected = UPSTREAM_RELEASES[release]
    config = anchor.infer_config(upstream_state_dict(n_layer, n_embed, hidden_dim, kv_rows))

    assert (config.n_layer, config.n_embed, config.hidden_dim) == (n_layer, n_embed, hidden_dim)
    assert config.n_kv_head == kv_rows // (n_embed // config.n_head)
    assert config.qk_norm is False

    with torch.device("meta"):
        model = TransformerLM(config)
    # Tied weights alias one tensor, which parameters() already deduplicates.
    assert sum(parameter.numel() for parameter in model.parameters()) == expected


def test_tokenizer_guard_constant_matches_the_released_weights() -> None:
    """The anchor is void unless the evaluation tokenizer is the trained one.

    Our tokenizer is upstream's post-"minimind-3" vocabulary; the released
    MiniMind2 weights were trained on the one before it. Same 6400-token budget,
    different mapping, so this constant must never be quietly set to ours.
    """
    assert anchor.UPSTREAM_TOKENIZER_SHA256 == (
        "d98595c6aef70d95f72748582fb9b4f53d76dd58c1ae1dd702ad7c84e1caf5e4"
    )
    assert anchor.UPSTREAM_TOKENIZER_SHA256 != (
        "71f32c68cf63a15355a8fc171b7594b3d41870fe0ddb54fc6aefa55f73a4a668"
    )
