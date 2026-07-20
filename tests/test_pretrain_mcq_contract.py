"""Stage-correct MCQ benchmark construction tests."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.build_pretrain_mcq_v2 import build_items, contamination_audit, validate_items


def test_benchmark_is_balanced_and_unique() -> None:
    items = build_items()
    validation = validate_items(items)
    assert validation["items"] == 192
    assert set(validation["categories"].values()) == {32}
    for counts in validation["answer_positions"].values():
        assert counts == {"0": 8, "1": 8, "2": 8, "3": 8}


def test_contamination_audit_detects_exact_normalized_prompt(tmp_path: Path) -> None:
    training = tmp_path / "train.jsonl"
    prompt = str(build_items()[0]["prompt"])
    training.write_text(json.dumps({"text": prompt}, ensure_ascii=False) + "\n", encoding="utf-8")
    result = contamination_audit(build_items(), [training])
    assert result["passed"] is False
