"""Model architecture tests."""

from __future__ import annotations

import torch

from python_starter.core.model import ModelConfig, TransformerLM


def test_model_forward() -> None:
    config = ModelConfig(
        vocab_size=100, n_embed=64, n_layer=2, n_head=4, max_seq_len=32
    )
    model = TransformerLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    logits, loss = model(input_ids, input_ids)
    assert logits.shape == (2, 16, config.vocab_size)
    assert loss is not None


def test_model_generate() -> None:
    config = ModelConfig(
        vocab_size=100, n_embed=64, n_layer=2, n_head=4, max_seq_len=32
    )
    model = TransformerLM(config)
    model.eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 5))
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=10)
    assert output.shape[1] == 15


def test_model_parameter_count() -> None:
    from python_starter.core.utils import count_parameters

    config = ModelConfig(
        vocab_size=100, n_embed=64, n_layer=2, n_head=4, max_seq_len=32
    )
    model = TransformerLM(config)
    params = count_parameters(model)
    assert params > 0
