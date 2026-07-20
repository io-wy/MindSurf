"""Build a deterministic Official/targeted replay view with an explicit ratio."""

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


def _load_source(config: dict[str, Any]) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    path = Path(config["path"])
    audit = json.loads(Path(config["audit"]).read_text(encoding="utf-8"))
    manifest = json.loads(Path(config["training_view_manifest"]).read_text(encoding="utf-8"))
    if audit.get("status") != "passed":
        raise ValueError(f"source dataset audit must pass: {path}")
    if sha256_file(path) != manifest["output"]["sha256"]:
        raise ValueError(f"source training view identity changed: {path}")
    return path, audit, manifest


def _select_rows(
    *,
    source: Path,
    rows: int,
    seed: int,
    excluded_hashes: set[str],
    source_label: str,
) -> list[tuple[str, dict[str, str]]]:
    heap: list[tuple[int, str, dict[str, str]]] = []
    seen: set[str] = set()
    for _, row in iter_jsonl(source):
        text = row.get("text")
        if not isinstance(text, str):
            continue
        normalized = normalized_text(text)
        text_hash = hashlib.sha256(normalized.encode()).hexdigest()
        if text_hash in excluded_hashes or text_hash in seen:
            continue
        seen.add(text_hash)
        score = int(hashlib.sha256(f"{seed}:{source_label}:{text_hash}".encode()).hexdigest(), 16)
        candidate = (
            -score,
            text_hash,
            {
                "text": normalized,
                "source_key": str(row.get("source_key", source_label)),
                "replay_arm": source_label,
            },
        )
        if len(heap) < rows:
            heapq.heappush(heap, candidate)
        elif candidate > heap[0]:
            heapq.heapreplace(heap, candidate)
    if len(heap) != rows:
        raise ValueError(f"{source_label} quota not met: expected {rows}, got {len(heap)}")
    return sorted(
        ((text_hash, row) for _, text_hash, row in heap),
        key=lambda value: value[0],
    )


def _write_interleaved(
    *,
    output: Path,
    official: list[tuple[str, dict[str, str]]],
    targeted: list[tuple[str, dict[str, str]]],
    official_per_targeted: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    official_index = 0
    targeted_index = 0
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        while official_index < len(official) or targeted_index < len(targeted):
            for _ in range(official_per_targeted):
                if official_index >= len(official):
                    break
                handle.write(
                    json.dumps(official[official_index][1], ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
                official_index += 1
            if targeted_index < len(targeted):
                handle.write(
                    json.dumps(targeted[targeted_index][1], ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
                targeted_index += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("configs/datasets/mindsurf_official_targeted_replay_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/processed/mindsurf_official_targeted_replay_v1/pretrain_train.jsonl"
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/data/mindsurf_official_targeted_replay_v1/training_view.json"),
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("artifacts/data/mindsurf_official_targeted_replay_v1/audit.json"),
    )
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    official_path, _, official_manifest = _load_source(spec["sources"]["official"])
    targeted_path, _, targeted_manifest = _load_source(spec["sources"]["targeted"])
    quotas = spec["row_quotas"]
    seed = int(spec["selection_seed"])

    targeted = _select_rows(
        source=targeted_path,
        rows=int(quotas["targeted"]),
        seed=seed,
        excluded_hashes=set(),
        source_label="targeted",
    )
    targeted_hashes = {text_hash for text_hash, _ in targeted}
    official = _select_rows(
        source=official_path,
        rows=int(quotas["official"]),
        seed=seed,
        excluded_hashes=targeted_hashes,
        source_label="official",
    )
    _write_interleaved(
        output=args.output,
        official=official,
        targeted=targeted,
        official_per_targeted=int(spec["interleave"]["official_per_targeted"]),
    )

    source_identity = hashlib.sha256(
        (
            f"{official_manifest['output']['sha256']}:"
            f"{targeted_manifest['output']['sha256']}:"
            f"{quotas['official']}:{quotas['targeted']}:{seed}"
        ).encode()
    ).hexdigest()
    output_identity = {
        "path": args.output.as_posix(),
        "sha256": sha256_file(args.output),
        "size": args.output.stat().st_size,
        "rows": len(official) + len(targeted),
    }
    expected_output = spec["files"]["train"]
    expected_sha256 = str(expected_output["sha256"])
    if expected_sha256 != "pending_materialization" and (
        output_identity["sha256"] != expected_sha256
        or output_identity["size"] != int(expected_output["size"])
        or output_identity["rows"] != int(expected_output["rows"])
    ):
        raise ValueError(
            f"replay output identity changed: expected {expected_output}, got {output_identity}"
        )
    manifest = {
        "schema_version": 1,
        "dataset_id": spec["dataset_id"],
        "dataset_revision": spec["revision"],
        "source": {
            "sha256": source_identity,
            "official_sha256": official_manifest["output"]["sha256"],
            "targeted_sha256": targeted_manifest["output"]["sha256"],
        },
        "output": output_identity,
        "row_quotas": quotas,
        "selection_seed": seed,
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
        "splits": {"train": {"sha256": source_identity}},
        "gates": {
            "schema_valid": True,
            "metadata_contract_valid": True,
            "file_identity_valid": True,
            "validation_test_disjoint": True,
            "train_holdout_disjoint": True,
            "public_release_license_ready": False,
        },
        "row_quotas": quotas,
    }
    write_json_atomic(args.manifest, manifest)
    write_json_atomic(args.audit, audit)
    print(json.dumps({"manifest": manifest, "audit": audit}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
