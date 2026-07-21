"""Emit a dataset spec for the full official pretraining corpus.

The frozen 80M work trained on `pretrain_t2t_mini.jsonl`, 1.24 GB of the
8.28 GB `pretrain_t2t.jsonl` that the same upstream revision publishes. That
choice was inherited from an earlier baseline and never justified anywhere in
the repository, and it is the binding constraint behind every result so far.

This spec keeps the already-frozen 4,000-row strict holdout as validation and
test, and points `train` at the full corpus. The training-view builder removes
holdout rows from `train` by NFKC text digest, so the resulting view is "full
corpus minus the frozen holdout" and every previously computed threshold and
comparison stays valid. `holdout_rows_removed` in the manifest then also
answers whether the mini file was a subset of the full one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from python_starter.core.data_contract import sha256_file, write_json_atomic  # noqa: E402


def _measure(path: Path) -> dict[str, Any]:
    rows = 0
    with path.open("rb") as handle:
        for line in handle:
            if line.strip():
                rows += 1
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "size": path.stat().st_size,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--base-spec",
        type=Path,
        default=ROOT / "configs/datasets/minimind_official_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "configs/datasets/minimind_official_full_v1.json",
    )
    parser.add_argument("--identity", default="minimind_official_full_v1")
    args = parser.parse_args()

    base = json.loads(args.base_spec.read_text(encoding="utf-8"))
    corpus = args.corpus if args.corpus.is_absolute() else args.root / args.corpus
    measured = _measure(corpus)

    spec = dict(base)
    spec["files"] = {
        "source": measured,
        # The builder reads `train` and strips the holdout rows from it.
        "train": dict(measured, path=corpus.relative_to(args.root).as_posix()),
        "validation": base["files"]["validation"],
        "test": base["files"]["test"],
    }
    spec["derived_from"] = {
        "base_spec": args.base_spec.relative_to(ROOT).as_posix(),
        "base_identity": "minimind_official_v1",
        "note": (
            "Same upstream dataset id and revision; the corpus file differs. The "
            "strict holdout is inherited unchanged so gate thresholds and prior "
            "comparisons remain valid."
        ),
    }
    write_json_atomic(args.output, spec)
    print(json.dumps({"identity": args.identity, "files": spec["files"]}, indent=1))


if __name__ == "__main__":
    main()
