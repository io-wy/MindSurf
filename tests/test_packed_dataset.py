"""Streaming packed-dataset tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from python_starter.core.dataset import JsonlPackedDataset


class _CharacterTokenizer:
    eos_token_id = 99

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [ord(value) - ord("a") + 1 for value in text]


def _dataset(path: Path, **values: Any) -> JsonlPackedDataset:
    return JsonlPackedDataset(
        path,
        _CharacterTokenizer(),  # type: ignore[arg-type]
        max_length=3,
        **values,
    )


def test_jsonl_records_are_packed_across_boundaries(tmp_path: Path) -> None:
    path = tmp_path / "sample.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"text": "ab"}),
                json.dumps({"text": "cde"}),
            ]
        ),
        encoding="utf-8",
    )

    blocks = list(_dataset(path))

    assert blocks[0]["input_ids"].tolist() == [1, 2, 99]
    assert blocks[0]["labels"].tolist() == [2, 99, 3]


def test_absolute_block_cursor_reproduces_suffix(tmp_path: Path) -> None:
    path = tmp_path / "sample.jsonl"
    path.write_text(
        "\n".join(json.dumps({"text": "abcdefgh"}) for _ in range(3)),
        encoding="utf-8",
    )

    all_blocks = list(_dataset(path))
    resumed_blocks = list(_dataset(path, skip_blocks=2))

    assert len(all_blocks) > 2
    assert [item["input_ids"].tolist() for item in resumed_blocks] == [
        item["input_ids"].tolist() for item in all_blocks[2:]
    ]
