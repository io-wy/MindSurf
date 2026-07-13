from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.pretrain.scripts.release_gate import evaluate_release


def promote_release(spec_path: Path, root: Path, output: Path) -> dict[str, Any]:
    root = root.resolve()
    spec_path = spec_path.resolve()
    report = evaluate_release(spec_path, root)
    if not report["passed"]:
        raise RuntimeError("release gate failed: " + "; ".join(report["failures"]))
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    payload = {
        "schema_version": 1,
        "release_id": spec["release_id"],
        "artifact_dir": spec["artifact_dir"],
        "release_spec": spec_path.relative_to(root).as_posix(),
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_sha256": spec["assets"]["checkpoint"]["sha256"],
        "model_sha256": spec["assets"]["model"]["sha256"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file():
        previous = output.with_name("previous.json")
        previous_temporary = previous.with_suffix(previous.suffix + ".tmp")
        previous_temporary.write_text(
            output.read_text(encoding="utf-8"),
            encoding="utf-8",
            newline="\n",
        )
        os.replace(previous_temporary, previous)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, output)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Atomically promote a gated MiniMind release")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("out/releases/current.json"))
    args = parser.parse_args()
    payload = promote_release(args.spec, args.root, args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
