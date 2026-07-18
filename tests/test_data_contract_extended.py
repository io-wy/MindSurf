"""Extended dataset-contract and dataset-class tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from python_starter.core.data_contract import (
    FileIdentity,
    audit_split,
    classify_language,
    download_tokenizer,
    iter_jsonl,
    length_bucket,
    load_dataset_spec,
    materialize_lfs_files,
    normalized_text,
    published_text_digest,
    sha256_file,
    verify_file,
    verify_training_view_manifest,
    write_json_atomic,
)
from python_starter.core.dataset import (
    JsonlPackedDataset,
    SFTDataset,
    TextDataset,
    collate_fn,
)


class _Tokenizer:
    eos_token_id = 9
    eos_token = "<eos>"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [1 + (ord(value) % 7) for value in text]


def _identity(path: Path, rows: int | None = None) -> FileIdentity:
    return FileIdentity(
        path=path.name,
        sha256=sha256_file(path),
        size=path.stat().st_size,
        rows=rows,
    )


def test_spec_file_and_normalization_contracts(tmp_path: Path) -> None:
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "repository_url": "https://example.invalid/repo",
                "revision": "abc",
                "files": {},
            }
        ),
        encoding="utf-8",
    )
    assert load_dataset_spec(spec)["revision"] == "abc"
    spec.write_text(json.dumps({"schema_version": 2}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        load_dataset_spec(spec)

    assert normalized_text("Ａ  \n B") == "A B"
    assert published_text_digest("Ａ  \n B") != published_text_digest("A B")
    assert classify_language("中文") == "zh"
    assert classify_language("text") == "en"
    assert classify_language("中a") == "mixed"
    assert classify_language("123") == "other"
    assert length_bucket(64) == "le_64"
    assert length_bucket(9000) == "gt_8192"


def test_file_verification_jsonl_and_audit_failures(tmp_path: Path) -> None:
    data = tmp_path / "split.jsonl"
    rows = [
        {"text": "hello", "source_key": "web", "id": 1},
        {"text": "hello", "source_key": "web", "id": 1},
    ]
    data.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    identity = _identity(data, rows=2)
    verify_file(data, identity)
    result, hashes = audit_split(data, expected=identity)
    assert result["duplicate_texts"] == 1
    assert result["duplicate_source_ids"] == 1
    assert len(hashes) == 1

    limited, _ = audit_split(data, expected=identity, max_rows=1)
    assert limited["rows"] == 1
    assert limited["sha256"] is None

    wrong = FileIdentity(data.name, "0" * 64, identity.size)
    with pytest.raises(ValueError, match="SHA-256"):
        verify_file(data, wrong)
    wrong_size = FileIdentity(data.name, identity.sha256, identity.size + 1)
    with pytest.raises(ValueError, match="Size"):
        verify_file(data, wrong_size)

    invalid = tmp_path / "invalid.jsonl"
    invalid.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected a JSON object"):
        list(iter_jsonl(invalid))


def test_training_view_manifest_verification(tmp_path: Path) -> None:
    training_view = tmp_path / "train.jsonl"
    training_view.write_text('{"text":"hello"}\n', encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "schema_version": 1,
        "dataset_id": "team/dataset",
        "dataset_revision": "revision",
        "source": {"sha256": "source-sha"},
        "output": {
            "path": str(training_view),
            "sha256": sha256_file(training_view),
            "size": training_view.stat().st_size,
            "rows": 1,
        },
        "gates": {
            "source_identity_valid": True,
            "nfkc_duplicate_free": True,
            "holdout_disjoint": True,
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    verified = verify_training_view_manifest(
        manifest_path,
        train_path=training_view,
        dataset_id="team/dataset",
        revision="revision",
        source_sha256="source-sha",
    )
    assert verified["output"]["rows"] == 1

    manifest["gates"] = {
        "source_identity_valid": True,
        "nfkc_duplicate_free": True,
        "holdout_disjoint": False,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="safety gates"):
        verify_training_view_manifest(
            manifest_path,
            train_path=training_view,
            dataset_id="team/dataset",
            revision="revision",
            source_sha256="source-sha",
        )


def test_atomic_json_lfs_materialization_and_tokenizer_download(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    output_json = tmp_path / "nested" / "value.json"
    write_json_atomic(output_json, {"z": 1, "a": 2})
    assert json.loads(output_json.read_text(encoding="utf-8")) == {"a": 2, "z": 1}

    payload = b"dataset"
    sha = hashlib.sha256(payload).hexdigest()
    identity = FileIdentity("strict/file.jsonl", sha, len(payload), 1)
    cache = tmp_path / "cache"
    object_path = cache / ".git" / "lfs" / "objects" / sha[:2] / sha[2:4] / sha
    object_path.parent.mkdir(parents=True)
    object_path.write_bytes(payload)
    monkeypatch.setattr(
        "python_starter.core.data_contract.run_checked",
        lambda *_, **__: None,
    )
    materialize_lfs_files(
        repository_url="https://example.invalid/repo",
        revision="abc",
        identities={"train": identity},
        output_root=tmp_path / "output",
        repository_cache=cache,
    )
    assert (tmp_path / "output" / identity.path).read_bytes() == payload

    tokenizer_payload = b"{}"
    tokenizer_sha = hashlib.sha256(tokenizer_payload).hexdigest()

    def fake_retrieve(url: str, destination: Path) -> None:
        assert "raw.githubusercontent.com" in url
        Path(destination).write_bytes(tokenizer_payload)

    monkeypatch.setattr(
        "python_starter.core.data_contract.urllib.request.urlretrieve",
        fake_retrieve,
    )
    download_tokenizer(
        {
            "repository_url": "https://github.com/example/repo",
            "revision": "abc",
            "files": {
                "tokenizer.json": {
                    "path": "model/tokenizer.json",
                    "sha256": tokenizer_sha,
                    "size": len(tokenizer_payload),
                }
            },
        },
        tmp_path / "tokenizer",
    )
    assert (tmp_path / "tokenizer" / "tokenizer.json").read_bytes() == tokenizer_payload


def test_text_sft_stream_shuffle_and_collation(tmp_path: Path) -> None:
    tokenizer = _Tokenizer()
    text = tmp_path / "text.txt"
    text.write_text("abcdefghijklmnopqrstuvwxyz", encoding="utf-8")
    dataset = TextDataset(text, tokenizer, max_length=4)  # type: ignore[arg-type]
    assert len(dataset) > 1
    assert dataset[0]["input_ids"].shape == torch.Size([4])

    short = tmp_path / "short.txt"
    short.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="No samples"):
        TextDataset(short, tokenizer, max_length=8)  # type: ignore[arg-type]

    sft_path = tmp_path / "sft.jsonl"
    sft_path.write_text(
        json.dumps({"prompt": "question", "response": "answer"}) + "\n",
        encoding="utf-8",
    )
    sft = SFTDataset(sft_path, tokenizer, max_length=10)  # type: ignore[arg-type]
    assert len(sft) == 1
    assert (sft[0]["labels"] == -100).any()

    packed_path = tmp_path / "packed.jsonl"
    packed_path.write_text(
        "\n".join(json.dumps({"text": value}) for value in ("abcdef", "ghijkl", "mnopqr")),
        encoding="utf-8",
    )
    packed = JsonlPackedDataset(
        packed_path,
        tokenizer,  # type: ignore[arg-type]
        max_length=3,
        shuffle_buffer=2,
        seed=7,
        max_blocks=2,
    )
    packed.set_skip_blocks(1)
    blocks = list(packed)
    assert len(blocks) == 2
    batch = collate_fn(
        [
            blocks[0],
            {
                "input_ids": blocks[1]["input_ids"][:2],
                "labels": blocks[1]["labels"][:2],
            },
        ]
    )
    assert batch["input_ids"].shape == (2, 3)
    assert batch["labels"][1, 2] == -100
