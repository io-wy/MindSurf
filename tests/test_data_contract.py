from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from python_starter.core.data_contract import (
    FileIdentity,
    audit_split,
    normalized_text,
    verify_file,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> FileIdentity:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    content = path.read_bytes()
    return FileIdentity(
        path=path.name,
        sha256=hashlib.sha256(content).hexdigest(),
        size=len(content),
        rows=len(rows),
    )


def test_normalized_text_is_cross_platform_stable() -> None:
    assert normalized_text("Ａ  \r\n B") == "A B"


def test_audit_detects_reserved_overlap(tmp_path: Path) -> None:
    holdout_path = tmp_path / "validation.jsonl"
    holdout_identity = _write_jsonl(
        holdout_path,
        [{"text": "same text", "source_key": "public/source", "id": 1}],
    )
    _, holdout_hashes = audit_split(holdout_path, expected=holdout_identity)

    train_path = tmp_path / "train.jsonl"
    train_identity = _write_jsonl(
        train_path,
        [
            {"text": "same  text", "source_key": "public/source", "id": 2},
            {"text": "unique", "source_key": "public/source", "id": 3},
        ],
    )
    result, _ = audit_split(
        train_path,
        expected=train_identity,
        reserved_hashes=holdout_hashes,
    )
    assert result["reserved_overlap_rows"] == 1


def test_verify_file_rejects_wrong_identity(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    identity = _write_jsonl(
        path,
        [{"text": "ok", "source_key": "public/source", "id": 1}],
    )
    wrong = FileIdentity(
        path=identity.path,
        sha256="0" * 64,
        size=identity.size,
        rows=identity.rows,
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_file(path, wrong)
