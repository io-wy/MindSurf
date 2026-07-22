"""Pre-tokenised block dataset equivalence tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from python_starter.core.dataset import JsonlPackedDataset, PackedBlockDataset

ROOT = Path(__file__).resolve().parents[1]


class _CharacterTokenizer:
    eos_token_id = 99

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [ord(value) - ord("a") + 1 for value in text]


def _write_corpus(path: Path, rows: int = 40) -> None:
    path.write_text(
        "\n".join(json.dumps({"text": "abcdefghij"[: (i % 9) + 2]}) for i in range(rows)),
        encoding="utf-8",
    )


def _pack_inline(path: Path, max_length: int) -> list[list[int]]:
    dataset = JsonlPackedDataset(path, _CharacterTokenizer(), max_length=max_length)  # type: ignore[arg-type]
    return [item["input_ids"].tolist() for item in dataset]


def _pack_offline(path: Path, out: Path, max_length: int) -> list[list[int]]:
    """Mirror the pretokenizer's packing without spawning a tokenizer process."""
    tokenizer = _CharacterTokenizer()
    tokens: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        tokens.extend(tokenizer.encode(json.loads(line)["text"]))
        tokens.append(tokenizer.eos_token_id)
    block_size = max_length + 1
    usable = (len(tokens) // block_size) * block_size
    np.asarray(tokens[:usable], dtype=np.uint16).tofile(out)
    dataset = PackedBlockDataset(out, max_length=max_length)
    return [item["input_ids"].tolist() for item in dataset]


def test_memmap_blocks_match_the_inline_packer_exactly(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(corpus)

    inline = _pack_inline(corpus, max_length=7)
    offline = _pack_offline(corpus, tmp_path / "blocks.bin", max_length=7)

    assert offline == inline


def test_cursor_semantics_match_after_a_resume(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(corpus)
    blocks = tmp_path / "blocks.bin"
    _pack_offline(corpus, blocks, max_length=7)

    full = [item["input_ids"].tolist() for item in PackedBlockDataset(blocks, max_length=7)]
    resumed = PackedBlockDataset(blocks, max_length=7)
    resumed.set_skip_blocks(3)

    assert [item["input_ids"].tolist() for item in resumed] == full[3:]


def test_epochs_wrap_the_cursor_without_re_reading_from_zero(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(corpus)
    blocks = tmp_path / "blocks.bin"
    _pack_offline(corpus, blocks, max_length=7)

    one = [item["input_ids"].tolist() for item in PackedBlockDataset(blocks, max_length=7)]
    three = [
        item["input_ids"].tolist() for item in PackedBlockDataset(blocks, max_length=7, epochs=3)
    ]

    assert three == one * 3
    resumed = PackedBlockDataset(blocks, max_length=7, epochs=3)
    resumed.set_skip_blocks(len(one) + 1)
    assert [item["input_ids"].tolist() for item in resumed] == three[len(one) + 1 :]


def test_a_truncated_array_is_rejected_rather_than_silently_misaligned(tmp_path: Path) -> None:
    blocks = tmp_path / "blocks.bin"
    # One token short of a whole block: reshaping would silently shift every
    # sample, so this must fail loudly.
    np.asarray(list(range(8 * 5 - 1)), dtype=np.uint16).tofile(blocks)

    with pytest.raises(ValueError, match="whole number"):
        PackedBlockDataset(blocks, max_length=7)


def test_labels_are_the_inputs_shifted_by_one(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    _write_corpus(corpus)
    blocks = tmp_path / "blocks.bin"
    _pack_offline(corpus, blocks, max_length=7)

    first = next(iter(PackedBlockDataset(blocks, max_length=7)))

    assert first["labels"].tolist()[:-1] == first["input_ids"].tolist()[1:]
    assert len(first["input_ids"]) == 7


def test_block_capacity_is_exposed_for_budget_checking(tmp_path: Path) -> None:
    """The number train.py needs to refuse an over-long budget up front.

    A 172,000-step request against a corpus holding 171,892 steps failed only
    when the loader ran dry, which for a single-epoch run is at the very end:
    two GPUs spent seven hours to surface an arithmetic error.
    """
    blocks = tmp_path / "blocks.bin"
    np.asarray(list(range(8 * 5)), dtype=np.uint16).tofile(blocks)

    dataset = PackedBlockDataset(blocks, max_length=7)

    assert dataset.blocks_per_epoch == 5
    assert PackedBlockDataset(blocks, max_length=7, epochs=3).blocks_per_epoch == 5
