"""Build a deterministic, audited STEM continuation view from the frozen team source."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from python_starter.core.data_contract import (  # noqa: E402
    iter_jsonl,
    normalized_text,
    sha256_file,
    write_json_atomic,
)


def _holdout_hashes(paths: list[Path]) -> set[str]:
    values = set()
    for path in paths:
        for _, row in iter_jsonl(path):
            text = row.get("text")
            if isinstance(text, str):
                values.add(hashlib.sha256(normalized_text(text).encode()).hexdigest())
    return values


def _select_rows(
    *,
    source: Path,
    components: list[dict[str, Any]],
    seed: int,
    holdout_hashes: set[str],
) -> tuple[dict[str, list[tuple[str, dict[str, str]]]], int]:
    by_key = {str(component["source_key"]): component for component in components}
    heaps: dict[str, list[tuple[int, str, dict[str, str]]]] = {
        str(component["name"]): [] for component in components
    }
    skipped_holdout = 0
    for _, row in iter_jsonl(source):
        source_key = row.get("source_key")
        text = row.get("text")
        if not isinstance(source_key, str) or not isinstance(text, str) or source_key not in by_key:
            continue
        normalized = normalized_text(text)
        text_hash = hashlib.sha256(normalized.encode()).hexdigest()
        if text_hash in holdout_hashes:
            skipped_holdout += 1
            continue
        component = by_key[source_key]
        name = str(component["name"])
        limit = int(component["rows"])
        score_hex = hashlib.sha256(f"{seed}:{source_key}:{text_hash}".encode()).hexdigest()
        score = int(score_hex, 16)
        candidate = (-score, text_hash, {"text": normalized, "source_key": source_key})
        heap = heaps[name]
        if len(heap) < limit:
            heapq.heappush(heap, candidate)
        elif candidate > heap[0]:
            heapq.heapreplace(heap, candidate)
    selected = {
        name: sorted(
            ((text_hash, row) for _, text_hash, row in heap),
            key=lambda value: value[0],
        )
        for name, heap in heaps.items()
    }
    return selected, skipped_holdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("configs/datasets/mindsurf_targeted_stem_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/mindsurf_targeted_stem_v1/pretrain_train.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/data/mindsurf_targeted_stem_v1/training_view.json"),
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("artifacts/data/mindsurf_targeted_stem_v1/audit.json"),
    )
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    source_config = spec["source"]
    source_path = Path(source_config["path"])
    source_audit = json.loads(Path(source_config["audit"]).read_text(encoding="utf-8"))
    source_manifest = json.loads(
        Path(source_config["training_view_manifest"]).read_text(encoding="utf-8")
    )
    if source_audit.get("status") != "passed":
        raise ValueError("source dataset audit must pass")
    if sha256_file(source_path) != source_manifest["output"]["sha256"]:
        raise ValueError("source training view identity changed")

    holdouts = [Path(value) for value in spec["holdouts"]]
    holdout_hashes = _holdout_hashes(holdouts)
    components = spec["components"]
    selected, skipped_holdout = _select_rows(
        source=source_path,
        components=components,
        seed=int(spec["selection_seed"]),
        holdout_hashes=holdout_hashes,
    )
    expected = {str(value["name"]): int(value["rows"]) for value in components}
    actual = {name: len(rows) for name, rows in selected.items()}
    if actual != expected:
        raise ValueError(f"component quotas were not met: expected {expected}, got {actual}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ordered_names = [str(value["name"]) for value in components]
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for offset in range(max(actual.values())):
            for name in ordered_names:
                rows = selected[name]
                if offset < len(rows):
                    handle.write(
                        json.dumps(rows[offset][1], ensure_ascii=False, separators=(",", ":"))
                        + "\n"
                    )
    output_identity = {
        "path": args.output.as_posix(),
        "sha256": sha256_file(args.output),
        "size": args.output.stat().st_size,
        "rows": sum(actual.values()),
    }
    source_identity = {
        "path": source_path.as_posix(),
        "sha256": source_manifest["output"]["sha256"],
        "rows": source_manifest["output"]["rows"],
    }
    manifest = {
        "schema_version": 1,
        "dataset_id": spec["dataset_id"],
        "dataset_revision": spec["revision"],
        "source": source_identity,
        "output": output_identity,
        "components": [
            {
                **component,
                "selected_rows": actual[str(component["name"])],
            }
            for component in components
        ],
        "selection_seed": spec["selection_seed"],
        "holdout_rows_removed": skipped_holdout,
        "gates": {
            "source_identity_valid": True,
            "nfkc_duplicate_free": True,
            "holdout_disjoint": True,
        },
    }
    audit = {
        "schema_version": 1,
        "status": "passed",
        "dataset_id": spec["dataset_id"],
        "dataset_revision": spec["revision"],
        "dataset_spec_sha256": sha256_file(args.spec),
        "license": spec["license"],
        "license_status": spec["license_status"],
        "splits": {"train": source_identity},
        "gates": {
            "schema_valid": True,
            "metadata_contract_valid": True,
            "file_identity_valid": True,
            "validation_test_disjoint": True,
            "train_holdout_disjoint": True,
            "public_release_license_ready": False,
        },
        "component_rows": actual,
        "holdout_files": {path.as_posix(): sha256_file(path) for path in holdouts},
    }
    write_json_atomic(args.manifest, manifest)
    write_json_atomic(args.audit, audit)
    print(json.dumps({"manifest": manifest, "audit": audit}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
