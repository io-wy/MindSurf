from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "id",
    "kind",
    "logical_path",
    "storage_path",
    "producing_run",
    "restore",
}
CHUNK_SIZE = 1024 * 1024


def _validate_definition(value: object, index: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"asset definition {index} must be an object")
    fields = set(value)
    if fields != REQUIRED_FIELDS:
        missing = sorted(REQUIRED_FIELDS - fields)
        extra = sorted(fields - REQUIRED_FIELDS)
        raise ValueError(
            f"asset definition {index} has invalid fields; missing={missing}, extra={extra}"
        )
    for field in REQUIRED_FIELDS - {"producing_run"}:
        if not isinstance(value[field], str) or not value[field]:
            raise ValueError(f"asset definition {index} field {field!r} must be a non-empty string")
    if value["producing_run"] is not None and (
        not isinstance(value["producing_run"], str) or not value["producing_run"]
    ):
        raise ValueError(
            f"asset definition {index} field 'producing_run' must be null or a non-empty string"
        )
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(source_file: Path, root: Path) -> dict[str, object]:
    definitions = json.loads(source_file.read_text(encoding="utf-8"))
    if not isinstance(definitions, list):
        raise ValueError("asset source must contain a JSON array")

    assets: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for index, raw_definition in enumerate(definitions):
        definition = _validate_definition(raw_definition, index)
        asset_id = definition["id"]
        if asset_id in seen_ids:
            raise ValueError(f"duplicate asset id: {asset_id}")
        seen_ids.add(asset_id)

        local_path = root / definition["logical_path"]
        present = local_path.is_file()
        assets.append(
            {
                **definition,
                "availability": "present" if present else "missing",
                "size_bytes": local_path.stat().st_size if present else None,
                "sha256": _sha256(local_path) if present else None,
            }
        )

    return {
        "schema_version": 1,
        "generated_from": source_file.as_posix(),
        "assets": sorted(assets, key=lambda asset: asset["id"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a deterministic asset manifest.")
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = build_manifest(args.sources, args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
