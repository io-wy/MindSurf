import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_record(record: dict, errors: list[str]) -> None:
    if "path" not in record or "sha256" not in record:
        return
    path = resolve(record["path"])
    if not path.exists():
        errors.append(f"missing: {record['path']}")
        return
    actual = sha256_file(path)
    if actual != record["sha256"]:
        errors.append(f"sha256 mismatch: {record['path']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify files referenced by a serving snapshot manifest and append log.")
    parser.add_argument("snapshot_dir")
    parser.add_argument("--strict-artifacts", action="store_true", help="Also verify every artifact recorded in append-log events.")
    args = parser.parse_args()

    snapshot_dir = resolve(args.snapshot_dir)
    manifest = json.loads((snapshot_dir / "current.json").read_text(encoding="utf-8"))
    errors: list[str] = []
    check_record(manifest["checkpoint"], errors)
    for log in manifest.get("append_logs", []):
        check_record(log, errors)
        log_path = resolve(log["path"])
        if not log_path.exists():
            continue
        last_seq = 0
        seen: set[int] = set()
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            seq = int(event["seq"])
            if seq in seen:
                errors.append(f"duplicate seq: {seq}")
            if seq != last_seq + 1:
                errors.append(f"non-contiguous seq: expected {last_seq + 1}, got {seq}")
            seen.add(seq)
            last_seq = seq
            if args.strict_artifacts:
                for artifact in event.get("artifacts", []):
                    check_record(artifact, errors)
        if last_seq != log.get("last_seq"):
            errors.append(f"last_seq mismatch: {log['path']}")
    print(json.dumps({"status": "failed" if errors else "ok", "errors": errors}, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
