"""Model architecture tests."""

from __future__ import annotations

import pytest
import torch

from python_starter.core.model import ModelConfig, TransformerLM


def test_model_forward() -> None:
    config = ModelConfig(vocab_size=100, n_embed=64, n_layer=2, n_head=4, max_seq_len=32)
    model = TransformerLM(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    logits, loss = model(input_ids, input_ids)
    assert logits.shape == (2, 16, config.vocab_size)
    assert loss is not None


def test_model_generate() -> None:
    config = ModelConfig(vocab_size=100, n_embed=64, n_layer=2, n_head=4, max_seq_len=32)
    model = TransformerLM(config)
    model.eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 5))
    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=10, temperature=0)
    assert output.shape[1] == 15


def test_model_parameter_count() -> None:
    from python_starter.core.utils import count_parameters

    config = ModelConfig(vocab_size=100, n_embed=64, n_layer=2, n_head=4, max_seq_len=32)
    model = TransformerLM(config)
    params = count_parameters(model)
    assert params > 0


def test_selected_80m_parameter_count_is_frozen() -> None:
    from python_starter.core.utils import count_parameters

    config = ModelConfig(
        vocab_size=6400,
        n_embed=768,
        n_layer=8,
        n_head=8,
        n_kv_head=8,
        max_seq_len=384,
        hidden_dim=3584,
    )
    assert count_parameters(TransformerLM(config)) == 89_864_448


def test_attention_is_causal() -> None:
    torch.manual_seed(7)
    config = ModelConfig(vocab_size=32, n_embed=32, n_layer=1, n_head=4, max_seq_len=8)
    model = TransformerLM(config).eval()
    left = torch.tensor([[1, 2, 3, 4]])
    right = torch.tensor([[1, 9, 8, 7]])

    left_logits, _ = model(left)
    right_logits, _ = model(right)

    torch.testing.assert_close(left_logits[:, 0], right_logits[:, 0])


def test_grouped_query_sampling_and_config_validation() -> None:
    config = ModelConfig(
        vocab_size=32,
        n_embed=32,
        n_layer=1,
        n_head=4,
        n_kv_head=2,
        max_seq_len=8,
    )
    model = TransformerLM(config)
    output = model.generate(
        torch.tensor([[1, 2]]),
        max_new_tokens=2,
        temperature=1.0,
        top_p=0.5,
    )
    assert output.shape == (1, 4)
    with pytest.raises(ValueError, match="temperature"):
        model.generate(torch.tensor([[1]]), temperature=-1)
    with pytest.raises(ValueError, match="top_p"):
        model.generate(torch.tensor([[1]]), top_p=0)
    with pytest.raises(ValueError, match="exceeds"):
        model(torch.ones((1, 9), dtype=torch.long))
    with pytest.raises(ValueError, match="n_embed"):
        TransformerLM(ModelConfig(n_embed=30, n_head=4))
    with pytest.raises(ValueError, match="n_head"):
        TransformerLM(ModelConfig(n_embed=32, n_head=4, n_kv_head=3))
