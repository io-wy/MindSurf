"""Equivalence of the Hugging Face export."""

from __future__ import annotations

import pytest
import torch
from scripts.export_hf import build_state_dict, convert, max_logit_difference

from python_starter.core.model import ModelConfig, TransformerLM

pytest.importorskip("transformers")

CONFIG = ModelConfig(vocab_size=64, n_embed=32, n_layer=2, n_head=4, max_seq_len=32, hidden_dim=64)


def _source() -> TransformerLM:
    torch.manual_seed(0)
    model = TransformerLM(CONFIG)
    # Untrained weights are near-symmetric, which would hide a swapped pair of
    # matrices; random ones make any misrouting show up in the logits.
    for parameter in model.parameters():
        torch.nn.init.normal_(parameter, mean=0.0, std=0.05)
    return model.eval().float()


def test_export_reproduces_the_source_logits_exactly() -> None:
    model = _source()

    exported = convert(CONFIG, model.state_dict())

    assert max_logit_difference(model, exported) == 0.0


def test_swapped_feedforward_matrices_are_caught() -> None:
    """gate_proj and up_proj differ only by which side takes the silu."""
    model = _source()
    state = dict(model.state_dict())
    state["blocks.0.ffn.w1.weight"], state["blocks.0.ffn.w2.weight"] = (
        state["blocks.0.ffn.w2.weight"],
        state["blocks.0.ffn.w1.weight"],
    )

    exported = convert(CONFIG, state)

    assert max_logit_difference(model, exported) > 0.0


def test_every_parameter_is_mapped() -> None:
    """A parameter left out of the rename would silently stay at its init."""
    model = _source()

    mapped = build_state_dict(model.state_dict(), CONFIG.n_layer)

    source_names = set(model.state_dict())
    # lm_head is tied to the embedding, so the two names carry one tensor.
    assert len(mapped) == len(source_names)


def test_a_model_without_qk_norm_is_refused() -> None:
    """Qwen3 has no place to put weights that were never trained."""
    config = ModelConfig(
        vocab_size=64, n_embed=32, n_layer=2, n_head=4, max_seq_len=32, qk_norm=False
    )
    model = TransformerLM(config).eval()

    with pytest.raises(ValueError, match="QK normalisation"):
        convert(config, model.state_dict())
