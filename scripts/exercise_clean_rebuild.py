"""Rebuild the project in an empty directory and verify every restored identity.

Proves that git, the dependency lock, and the verified asset backup are jointly
sufficient to reconstruct a run's inputs, without borrowing anything from the
working checkout. The scratch tree is disposable; the evidence JSON is not.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from python_starter.core.data_contract import sha256_file, write_json_atomic  # noqa: E402


def _run(
    arguments: list[str], cwd: Path, timeout: float = 3600.0
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _step(
    records: list[dict[str, Any]], name: str, completed: subprocess.CompletedProcess[str]
) -> bool:
    ok = completed.returncode == 0
    records.append(
        {
            "step": name,
            "returncode": completed.returncode,
            "ok": ok,
            "stderr_tail": completed.stderr.strip().splitlines()[-5:],
        }
    )
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--remote", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--uv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--tests",
        default="tests/test_packed_dataset.py,tests/test_training_monitor.py,tests/test_dataset_registry.py",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Leave the scratch tree in place instead of removing it",
    )
    args = parser.parse_args()

    if args.workdir.exists() and any(args.workdir.iterdir()):
        raise SystemExit(f"{args.workdir} is not empty; the drill must start from nothing")
    args.workdir.mkdir(parents=True, exist_ok=True)

    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    checkout = args.workdir / "repo"
    steps: list[dict[str, Any]] = []
    started_at = datetime.now(UTC).isoformat()

    ok = _step(
        steps,
        "git_clone",
        _run(
            [
                "git",
                "clone",
                "--branch",
                args.branch,
                "--single-branch",
                args.remote,
                str(checkout),
            ],
            cwd=args.workdir,
        ),
    )
    head = ""
    if ok:
        completed = _run(["git", "rev-parse", "HEAD"], cwd=checkout)
        head = completed.stdout.strip()

    if ok:
        ok = _step(
            steps,
            "uv_sync_frozen",
            # --extra dev, not a bare sync: the plain lock install has no pytest,
            # so a rebuilt environment could not verify itself.
            _run([str(args.uv), "sync", "--frozen", "--extra", "dev"], cwd=checkout),
        )

    restored: list[dict[str, Any]] = []
    if ok:
        for asset in inventory["assets"]:
            backup = Path(str(asset["backup_path"]))
            target = checkout / str(asset["source"])
            entry: dict[str, Any] = {
                "logical_name": asset["logical_name"],
                "source": asset["source"],
                "expected_sha256": asset["sha256"],
            }
            if not backup.is_file():
                entry.update({"restored": False, "reason": "backup copy missing"})
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(backup, target)
                digest = sha256_file(target)
                entry.update({"restored": True, "observed_sha256": digest})
                entry["matches"] = digest == asset["sha256"]
            restored.append(entry)
        ok = all(item.get("matches") for item in restored)
        steps.append(
            {"step": "restore_assets", "ok": ok, "returncode": 0 if ok else 1, "stderr_tail": []}
        )

    if ok:
        ok = _step(
            steps,
            "tests",
            _run(
                [
                    str(checkout / ".venv/bin/python"),
                    "-m",
                    "pytest",
                    *args.tests.split(","),
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    "--no-cov",
                ],
                cwd=checkout,
            ),
        )

    manifest_identity: dict[str, Any] = {}
    if ok:
        manifest = checkout / "artifacts/data/minimind_official_v1/training_view.json"
        view = checkout / "data/processed/minimind_official_v1/pretrain_train_nfkc_dedup.jsonl"
        if manifest.is_file() and view.is_file():
            recorded = json.loads(manifest.read_text(encoding="utf-8"))
            observed = sha256_file(view)
            expected = str(recorded.get("output", {}).get("sha256") or "")
            manifest_identity = {
                "training_view_sha256_expected": expected,
                "training_view_sha256_observed": observed,
                "matches": bool(expected) and expected == observed,
            }
            ok = bool(manifest_identity["matches"])
        else:
            manifest_identity = {"matches": False, "reason": "manifest or view missing"}
            ok = False
        steps.append(
            {"step": "identity_check", "ok": ok, "returncode": 0 if ok else 1, "stderr_tail": []}
        )

    write_json_atomic(
        args.output,
        {
            "schema_version": 1,
            "started_at": started_at,
            "completed_at": datetime.now(UTC).isoformat(),
            "remote": args.remote,
            "branch": args.branch,
            "rebuilt_head": head,
            "workdir": str(args.workdir),
            "inventory": str(args.inventory),
            "steps": steps,
            "restored_assets": restored,
            "training_view_identity": manifest_identity,
            "rebuild_passed": ok,
        },
    )

    if not args.keep and checkout.exists():
        shutil.rmtree(args.workdir, ignore_errors=True)

    print(f"rebuild_passed={ok} head={head[:12]}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
