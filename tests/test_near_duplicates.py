"""MinHash near-duplicate detection tests."""

from __future__ import annotations

import numpy as np
import pytest

from python_starter.core.near_duplicates import (
    MinHashConfig,
    band_keys,
    duplicate_indices,
    hash_shingle,
    normalize_for_shingles,
    permutation_parameters,
    shingle_hashes,
    signature,
)


def _signatures(texts: list[str], config: MinHashConfig) -> np.ndarray:
    a, b = permutation_parameters(config)
    return np.vstack([signature(shingle_hashes(text, config.shingle_size), a, b) for text in texts])


def test_shingle_hash_is_stable_across_calls() -> None:
    # Python's hash() is salted per process; a signature built with it would
    # not match one built in a worker process.
    assert hash_shingle("样品甲") == hash_shingle("样品甲")


def test_normalisation_folds_width_and_spacing() -> None:
    assert normalize_for_shingles("ＡＢ  １２") == normalize_for_shingles("ab 12")


def test_near_duplicate_is_caught_where_exact_hashing_fails() -> None:
    config = MinHashConfig()
    base = "深度学习模型的训练需要大量的算力和数据，" * 6
    edited = base + "本文最后更新于二零二六年七月。"

    assert base != edited  # exact digest deduplication keeps both
    dropped = duplicate_indices(_signatures([base, edited], config), config)

    assert dropped == {1}


def test_unrelated_documents_are_not_merged() -> None:
    config = MinHashConfig()
    texts = [
        "深度学习模型的训练需要大量的算力和数据。" * 6,
        "今天的天气很好，适合去公园散步和野餐。" * 6,
        "def train(model, loader): return model.fit(loader)" * 6,
    ]

    assert duplicate_indices(_signatures(texts, config), config) == set()


def test_the_earliest_occurrence_survives() -> None:
    config = MinHashConfig()
    text = "重复文本用于验证去重保留最早出现的那一条。" * 6
    dropped = duplicate_indices(_signatures([text, text, text], config), config)

    assert dropped == {1, 2}


def test_empty_documents_do_not_collapse_into_one_group() -> None:
    config = MinHashConfig()

    assert duplicate_indices(_signatures(["", "", ""], config), config) == set()


def test_band_geometry_puts_the_threshold_where_documented() -> None:
    config = MinHashConfig()

    assert config.rows_per_band == 8
    assert config.detection_probability(0.9) > 0.98
    assert config.detection_probability(0.5) < 0.05


def test_band_keys_shape_and_determinism() -> None:
    config = MinHashConfig()
    signatures = _signatures(["文本一" * 20, "文本二" * 20], config)

    first = band_keys(signatures, config)
    second = band_keys(signatures, config)

    assert first.shape == (2, config.bands)
    assert np.array_equal(first, second)


def test_invalid_geometry_is_rejected() -> None:
    with pytest.raises(ValueError, match="divide evenly"):
        MinHashConfig(permutations=64, bands=7)
