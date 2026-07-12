from __future__ import annotations

import json
from itertools import islice
from pathlib import Path
from types import SimpleNamespace

import torch

from experiments.pretrain.scripts.train_pretrain_optimized import (
    StreamingMixedPackedPretrainDataset,
    StreamingPackedPretrainDataset,
)


class TinyTokenizer:
    eos_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool = False) -> SimpleNamespace:
        del add_special_tokens
        return SimpleNamespace(input_ids=[ord(char) % 31 + 1 for char in text])


def test_stream_resume_replays_shuffle_state_before_skipping(tmp_path: Path) -> None:
    data_path = tmp_path / "train.jsonl"
    rows = [{"text": f"row-{index}-abcdefgh"} for index in range(12)]
    data_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    common = {
        "data_path": data_path,
        "tokenizer": TinyTokenizer(),
        "max_length": 8,
        "shuffle_buffer": 4,
        "seed": 123,
    }

    uninterrupted = list(islice(StreamingPackedPretrainDataset(**common), 9))
    resumed = list(
        islice(
            StreamingPackedPretrainDataset(**common, skip_blocks=5),
            4,
        )
    )

    assert len(uninterrupted) == 9
    assert all(torch.equal(actual[0], expected[0]) for actual, expected in zip(resumed, uninterrupted[5:]))


def test_mixed_stream_resume_replays_source_and_shuffle_state(tmp_path: Path) -> None:
    sources = []
    for source_index in range(2):
        source_path = tmp_path / f"source-{source_index}.jsonl"
        rows = [
            {"text": f"source-{source_index}-row-{row_index}-abcdefgh"}
            for row_index in range(8)
        ]
        source_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        sources.append({"path": source_path.name, "weight": source_index + 1})
    mix_path = tmp_path / "mix.json"
    mix_path.write_text(json.dumps({"sources": sources}), encoding="utf-8")
    common = {
        "mix_path": mix_path,
        "tokenizer": TinyTokenizer(),
        "max_length": 8,
        "shuffle_buffer": 4,
        "seed": 321,
    }

    uninterrupted = list(islice(StreamingMixedPackedPretrainDataset(**common), 9))
    resumed = list(
        islice(
            StreamingMixedPackedPretrainDataset(**common, skip_blocks=5),
            4,
        )
    )

    assert all(torch.equal(actual[0], expected[0]) for actual, expected in zip(resumed, uninterrupted[5:]))
