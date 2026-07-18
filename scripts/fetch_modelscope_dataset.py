"""Fetch the pinned MindSurf team dataset and tokenizer without duplicate LFS storage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from python_starter.core.data_contract import (
    FileIdentity,
    download_tokenizer,
    load_dataset_spec,
    materialize_lfs_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        default="configs/datasets/mindsurf_team_v1.json",
        help="Pinned dataset source specification",
    )
    parser.add_argument(
        "--output",
        default="data/raw/mindsurf_team_v1",
        help="Materialized dataset root",
    )
    parser.add_argument(
        "--repo-cache",
        default="data/raw/.modelscope/mindsurf_pretrain_dataset_full",
        help="Local metadata and LFS object cache",
    )
    parser.add_argument(
        "--include",
        default="train,validation,test,metadata",
        help="Comma-separated logical dataset file names",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = load_dataset_spec(args.spec)
    requested = [name.strip() for name in args.include.split(",") if name.strip()]
    unknown = sorted(set(requested) - set(spec["files"]))
    if unknown:
        raise ValueError(f"Unknown dataset file names: {unknown}")
    identities = {name: FileIdentity.from_mapping(spec["files"][name]) for name in requested}
    materialize_lfs_files(
        repository_url=str(spec["repository_url"]),
        revision=str(spec["revision"]),
        identities=identities,
        output_root=args.output,
        repository_cache=args.repo_cache,
    )
    download_tokenizer(spec["tokenizer"], Path(args.output) / "tokenizer")
    print(f"Verified {len(identities)} dataset files and tokenizer at {args.output}")


if __name__ == "__main__":
    main()
