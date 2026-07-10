import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile


ROOT = Path(__file__).resolve().parents[3]


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    with NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        tmp = Path(handle.name)
    tmp.replace(path)


@contextmanager
def file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except ImportError:
            yield


def main() -> None:
    parser = argparse.ArgumentParser(description="Append one event to a serving snapshot log and refresh manifest hashes.")
    parser.add_argument("--snapshot_dir", required=True)
    parser.add_argument("--event", required=True)
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("--status", default="")
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    snapshot_dir = resolve(args.snapshot_dir)
    manifest_path = snapshot_dir / "manifest.json"
    current_path = snapshot_dir / "current.json"
    log_path = snapshot_dir / "append-000001.jsonl"

    with file_lock(snapshot_dir / "append-000001.lock"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        last_seq = 0
        if log_path.exists():
            for line in log_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    last_seq = max(last_seq, int(json.loads(line)["seq"]))

        artifacts = []
        for item in args.artifact:
            path = resolve(item)
            artifacts.append({"path": rel(path), "sha256": sha256_file(path), "bytes": path.stat().st_size})

        event = {
            "seq": last_seq + 1,
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": args.event,
            "artifacts": artifacts,
        }
        if args.status:
            event["status"] = args.status
        if args.note:
            event["note"] = args.note

        with log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

        manifest["append_logs"] = [
            {"path": rel(log_path), "sha256": sha256_file(log_path), "first_seq": 1, "last_seq": event["seq"]}
        ]
        atomic_json(manifest_path, manifest)
        atomic_json(current_path, manifest)
    print(json.dumps(event, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
