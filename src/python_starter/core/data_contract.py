"""Dataset identity, download verification, and strict-split auditing."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import unicodedata
import urllib.request
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z]")


@dataclass(frozen=True)
class FileIdentity:
    """Expected immutable identity for one external file."""

    path: str
    sha256: str
    size: int
    rows: int | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> FileIdentity:
        """Build a validated identity from JSON-compatible input."""
        rows = value.get("rows")
        return cls(
            path=str(value["path"]),
            sha256=str(value["sha256"]).lower(),
            size=int(value["size"]),
            rows=int(rows) if rows is not None else None,
        )


def load_dataset_spec(path: str | Path) -> dict[str, Any]:
    """Load and minimally validate a dataset source specification."""
    spec_path = Path(path)
    data = json.loads(spec_path.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise ValueError(f"Unsupported dataset spec version in {spec_path}")
    if not data.get("repository_url") or not data.get("revision"):
        raise ValueError(f"Dataset repository and revision are required in {spec_path}")
    if not isinstance(data.get("files"), dict):
        raise ValueError(f"Dataset files mapping is required in {spec_path}")
    return cast(dict[str, Any], data)


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return the SHA-256 of a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: str | Path, identity: FileIdentity) -> None:
    """Verify size and SHA-256 against an immutable source identity."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    observed_size = file_path.stat().st_size
    if observed_size != identity.size:
        raise ValueError(
            f"Size mismatch for {file_path}: expected {identity.size}, got {observed_size}"
        )
    observed_sha = sha256_file(file_path)
    if observed_sha != identity.sha256:
        raise ValueError(
            f"SHA-256 mismatch for {file_path}: expected {identity.sha256}, got {observed_sha}"
        )


def verify_training_view_manifest(
    manifest_path: str | Path,
    *,
    train_path: str | Path,
    dataset_id: str,
    revision: str,
    source_sha256: str,
) -> dict[str, Any]:
    """Verify the immutable identity and safety gates of a derived training view."""
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"training-view manifest is required before training: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported training-view manifest version")
    if manifest.get("dataset_id") != dataset_id or manifest.get("dataset_revision") != revision:
        raise ValueError("training-view manifest dataset identity does not match")
    source = manifest.get("source")
    output = manifest.get("output")
    if not isinstance(source, dict) or not isinstance(output, dict):
        raise ValueError("training-view manifest source and output identities are required")
    if source.get("sha256") != source_sha256:
        raise ValueError("training-view source does not match the audited training split")
    required_gates = ("source_identity_valid", "nfkc_duplicate_free", "holdout_disjoint")
    if not all(manifest.get("gates", {}).get(name) is True for name in required_gates):
        raise ValueError("training-view manifest has not passed all safety gates")

    configured_path = Path(train_path).resolve()
    declared_path = Path(str(output.get("path", ""))).resolve()
    if configured_path != declared_path:
        raise ValueError(
            f"training-view path mismatch: configured {configured_path}, declared {declared_path}"
        )
    rows = int(output.get("rows", 0))
    if rows <= 0:
        raise ValueError("training-view output row count must be positive")
    identity = FileIdentity(
        path=str(output["path"]),
        sha256=str(output["sha256"]),
        size=int(output["size"]),
        rows=rows,
    )
    verify_file(configured_path, identity)
    return cast(dict[str, Any], manifest)


def normalized_text(text: str) -> str:
    """Return the canonical text form used for cross-split identity."""
    return " ".join(unicodedata.normalize("NFKC", text).split())


def text_digest(text: str) -> bytes:
    """Return a strict NFKC-normalized SHA-1 identity for overlap checks."""
    return hashlib.sha1(normalized_text(text).encode("utf-8")).digest()


def published_text_digest(text: str) -> bytes:
    """Return the whitespace-normalized SHA-1 used in published split metadata."""
    return hashlib.sha1(" ".join(text.split()).encode("utf-8")).digest()


def classify_language(text: str) -> str:
    """Classify a row into coarse CJK/Latin buckets for corpus auditing."""
    cjk = len(_CJK_RE.findall(text))
    latin = len(_LATIN_RE.findall(text))
    if cjk and latin:
        return "mixed"
    if cjk:
        return "zh"
    if latin:
        return "en"
    return "other"


def length_bucket(length: int) -> str:
    """Return a stable character-length bucket label."""
    for boundary in (64, 128, 256, 512, 1024, 2048, 4096, 8192):
        if length <= boundary:
            return f"le_{boundary}"
    return "gt_8192"


def iter_jsonl(path: str | Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield one decoded JSON object per non-empty line."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield line_number, value


def audit_split(
    path: str | Path,
    *,
    expected: FileIdentity,
    reserved_hashes: set[bytes] | None = None,
    collect_hashes: bool = True,
    max_rows: int | None = None,
    require_source_id: bool = True,
) -> tuple[dict[str, Any], set[bytes]]:
    """Audit schema, identity, language, source, duplicate, and overlap counts."""
    file_path = Path(path)
    if max_rows is None:
        verify_file(file_path, expected)

    seen_texts: set[bytes] = set()
    seen_ids: set[bytes] = set()
    sources: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    lengths: Counter[str] = Counter()
    duplicate_texts = 0
    duplicate_ids = 0
    reserved_overlaps = 0
    rows = 0
    total_characters = 0

    for line_number, row in iter_jsonl(file_path):
        text = row.get("text")
        source = row.get("source_key")
        row_id = row.get("id")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{file_path}:{line_number}: missing non-empty text")
        if require_source_id:
            if not isinstance(source, str) or not source:
                raise ValueError(f"{file_path}:{line_number}: missing source_key")
            if not isinstance(row_id, (str, int)):
                raise ValueError(f"{file_path}:{line_number}: id must be string or integer")
        else:
            source = str(source) if source is not None else "official"
            row_id = row_id if row_id is not None else line_number

        digest = text_digest(text)
        if digest in seen_texts:
            duplicate_texts += 1
        elif collect_hashes:
            seen_texts.add(digest)

        identity_digest = hashlib.sha1(f"{source}\0{row_id}".encode()).digest()
        if identity_digest in seen_ids:
            duplicate_ids += 1
        else:
            seen_ids.add(identity_digest)

        if reserved_hashes is not None and digest in reserved_hashes:
            reserved_overlaps += 1

        sources[source] += 1
        languages[classify_language(text)] += 1
        lengths[length_bucket(len(text))] += 1
        total_characters += len(text)
        rows += 1
        if max_rows is not None and rows >= max_rows:
            break

    if max_rows is None and expected.rows is not None and rows != expected.rows:
        raise ValueError(f"Row mismatch for {file_path}: expected {expected.rows}, got {rows}")

    result = {
        "path": str(file_path.as_posix()),
        "rows": rows,
        "bytes": file_path.stat().st_size,
        "sha256": expected.sha256 if max_rows is None else None,
        "characters": total_characters,
        "duplicate_texts": duplicate_texts,
        "duplicate_source_ids": duplicate_ids,
        "reserved_overlap_rows": reserved_overlaps,
        "language_rows": dict(sorted(languages.items())),
        "length_rows": dict(sorted(lengths.items())),
        "source_rows": dict(sorted(sources.items())),
    }
    return result, seen_texts


def write_json_atomic(path: str | Path, value: Mapping[str, Any]) -> None:
    """Atomically write deterministic UTF-8 JSON."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, output)


def run_checked(
    args: Iterable[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> None:
    """Run a subprocess and raise with its exact exit status."""
    subprocess.run(list(args), cwd=cwd, env=env, check=True)


def _lfs_object_path(repository: Path, sha256: str) -> Path:
    return repository / ".git" / "lfs" / "objects" / sha256[:2] / sha256[2:4] / sha256


def materialize_lfs_files(
    *,
    repository_url: str,
    revision: str,
    identities: Mapping[str, FileIdentity],
    output_root: str | Path,
    repository_cache: str | Path,
) -> None:
    """Fetch selected Git LFS objects and hard-link them into the data directory."""
    output = Path(output_root)
    cache = Path(repository_cache)
    existing = {name: output / identity.path for name, identity in identities.items()}
    if all(path.is_file() for path in existing.values()):
        for name, path in existing.items():
            verify_file(path, identities[name])
        return
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        run_checked(
            [
                "git",
                "clone",
                "--no-checkout",
                repository_url,
                str(cache),
            ]
        )
    run_checked(["git", "fetch", "origin", revision, "--depth", "1"], cwd=cache)
    include = ",".join(identity.path for identity in identities.values())
    checkout_env = dict(os.environ)
    checkout_env["GIT_LFS_SKIP_SMUDGE"] = "1"
    run_checked(
        [
            "git",
            "checkout",
            revision,
            "--",
            ".gitattributes",
            *(identity.path for identity in identities.values()),
        ],
        cwd=cache,
        env=checkout_env,
    )
    run_checked(
        ["git", "lfs", "fetch", "origin", revision, f"--include={include}", "--exclude="],
        cwd=cache,
    )

    for identity in identities.values():
        source = _lfs_object_path(cache, identity.sha256)
        if not source.is_file():
            raise FileNotFoundError(f"Fetched LFS object is missing: {source}")
        verify_file(source, identity)
        destination = output / identity.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            verify_file(destination, identity)
            continue
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
        verify_file(destination, identity)


def download_tokenizer(
    tokenizer_spec: Mapping[str, Any],
    output_dir: str | Path,
) -> None:
    """Download a tokenizer bundle from a pinned GitHub revision and verify it."""
    repository_url = str(tokenizer_spec["repository_url"]).removesuffix(".git")
    revision = str(tokenizer_spec["revision"])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name, raw_identity in tokenizer_spec["files"].items():
        identity = FileIdentity.from_mapping(raw_identity)
        destination = output / str(name)
        if destination.exists():
            verify_file(destination, identity)
            continue
        repository_path = identity.path
        if repository_url.startswith("https://github.com/"):
            relative = repository_url.removeprefix("https://github.com/")
            url = f"https://raw.githubusercontent.com/{relative}/{revision}/{repository_path}"
        else:
            raise ValueError(f"Unsupported tokenizer repository: {repository_url}")
        temporary = destination.with_suffix(destination.suffix + ".download")
        urllib.request.urlretrieve(url, temporary)
        verify_file(temporary, identity)
        os.replace(temporary, destination)
