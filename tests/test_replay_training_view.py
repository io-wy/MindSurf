"""Official/targeted replay-view tests."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.build_replay_training_view import _select_rows, _write_interleaved


def test_replay_selection_is_deterministic_unique_and_interleaved(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    source.write_text(
        "".join(json.dumps({"text": f"row {index}"}) + "\n" for index in range(20)),
        encoding="utf-8",
    )
    first = _select_rows(
        source=source,
        rows=8,
        seed=7,
        excluded_hashes=set(),
        source_label="official",
    )
    second = _select_rows(
        source=source,
        rows=8,
        seed=7,
        excluded_hashes=set(),
        source_label="official",
    )
    assert first == second
    assert len({value[0] for value in first}) == 8

    output = tmp_path / "mixed.jsonl"
    _write_interleaved(
        output=output,
        official=first,
        targeted=[
            ("target-a", {"text": "target a", "replay_arm": "targeted"}),
            ("target-b", {"text": "target b", "replay_arm": "targeted"}),
        ],
        official_per_targeted=4,
    )
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["replay_arm"] for row in rows] == [
        "official",
        "official",
        "official",
        "official",
        "targeted",
        "official",
        "official",
        "official",
        "official",
        "targeted",
    ]
