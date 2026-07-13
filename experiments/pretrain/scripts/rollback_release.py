from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def rollback_release(current: Path, previous: Path) -> dict[str, Any]:
    if not current.is_file():
        raise FileNotFoundError(f"current release pointer missing: {current}")
    if not previous.is_file():
        raise FileNotFoundError(f"previous release pointer missing: {previous}")
    current_payload = json.loads(current.read_text(encoding="utf-8"))
    previous_payload = json.loads(previous.read_text(encoding="utf-8"))
    artifact_dir = Path(previous_payload["artifact_dir"])
    if artifact_dir.is_absolute() or ".." in artifact_dir.parts:
        raise ValueError(f"unsafe previous artifact path: {artifact_dir}")
    _atomic_write(current, previous_payload)
    _atomic_write(previous, current_payload)
    return previous_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Swap current and previous MiniMind release pointers")
    parser.add_argument("--current", type=Path, default=Path("out/releases/current.json"))
    parser.add_argument("--previous", type=Path, default=Path("out/releases/previous.json"))
    args = parser.parse_args()
    payload = rollback_release(args.current, args.previous)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
