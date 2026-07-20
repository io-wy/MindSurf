"""Targeted continuation-view selection tests."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.build_targeted_stem_view import _select_rows


def test_selection_is_deterministic_balanced_and_excludes_holdout(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    rows = [{"text": f"alpha {index}", "source_key": "a"} for index in range(10)] + [
        {"text": f"beta {index}", "source_key": "b"} for index in range(10)
    ]
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    components = [
        {"name": "one", "source_key": "a", "rows": 3},
        {"name": "two", "source_key": "b", "rows": 3},
    ]
    first, skipped = _select_rows(
        source=source,
        components=components,
        seed=7,
        holdout_hashes=set(),
    )
    second, _ = _select_rows(
        source=source,
        components=components,
        seed=7,
        holdout_hashes=set(),
    )
    assert first == second
    assert {name: len(values) for name, values in first.items()} == {"one": 3, "two": 3}
    assert skipped == 0
