from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from scripts.build_asset_manifest import build_manifest


def test_build_manifest_records_present_and_missing_assets(tmp_path: Path) -> None:
    payload = tmp_path / "dataset.jsonl"
    payload.write_text('{"text":"sample"}\n', encoding="utf-8")
    definitions = [
        {
            "id": "strict-sample",
            "kind": "dataset",
            "logical_path": "dataset.jsonl",
            "storage_path": "dataset.jsonl",
            "producing_run": None,
            "restore": "Copy the verified file to dataset.jsonl.",
        },
        {
            "id": "stage12-ext80-replay",
            "kind": "checkpoint",
            "logical_path": "stage12.pth",
            "storage_path": "stage12.pth",
            "producing_run": "pretrain_stage12_ext80_replay_from_stage11_s512_lr3e7",
            "restore": "Recover from an independently verified backup.",
        },
    ]
    source = tmp_path / "sources.json"
    source.write_text(json.dumps(definitions), encoding="utf-8")

    manifest = build_manifest(source, tmp_path)

    assets = {entry["id"]: entry for entry in manifest["assets"]}
    assert assets["strict-sample"]["availability"] == "present"
    assert assets["strict-sample"]["size_bytes"] == payload.stat().st_size
    assert assets["strict-sample"]["sha256"] == hashlib.sha256(payload.read_bytes()).hexdigest()
    assert assets["stage12-ext80-replay"]["availability"] == "missing"
    assert assets["stage12-ext80-replay"]["size_bytes"] is None
    assert assets["stage12-ext80-replay"]["sha256"] is None


def test_cli_writes_deterministic_manifest_with_canonical_lf(tmp_path: Path) -> None:
    source = tmp_path / "sources.json"
    output = tmp_path / "manifest.json"
    source.write_text("[]\n", encoding="utf-8")
    command = [
        sys.executable,
        "scripts/build_asset_manifest.py",
        "--sources",
        str(source),
        "--root",
        str(tmp_path),
        "--output",
        str(output),
    ]

    subprocess.run(command, check=True)
    first = output.read_bytes()
    subprocess.run(command, check=True)

    assert output.read_bytes() == first
    assert first == (
        b'{\n  "assets": [],\n  "generated_from": "'
        + source.as_posix().encode()
        + b'",\n  "schema_version": 1\n}\n'
    )
    assert b"\r\n" not in first
