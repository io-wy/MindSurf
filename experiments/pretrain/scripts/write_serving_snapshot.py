import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
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


def file_record(path: Path) -> dict:
    return {
        "path": rel(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def git_rev() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def write_executable(path: Path, body: str) -> None:
    path.write_text(body, encoding="utf-8", newline="\n")
    path.chmod(path.stat().st_mode | 0o111)


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temp_path = Path(handle.name)
    temp_path.replace(path)


def write_append_log(path: Path, events: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a reproducible serving snapshot for the current MiniMind checkpoint.")
    parser.add_argument("--name", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--export_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8998)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--num_key_value_heads", type=int, default=8)
    parser.add_argument("--intermediate_size", type=int, default=3072)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--batch_max_size", type=int, default=4)
    parser.add_argument("--batch_wait_ms", type=float, default=8.0)
    parser.add_argument("--flash_attn_wheel", default="", help="Optional wheel candidate to record in the snapshot evidence.")
    args = parser.parse_args()

    checkpoint = resolve(args.checkpoint)
    export_dir = resolve(args.export_dir)
    output_dir = resolve(args.output_dir)
    flash_attn_wheel = resolve(args.flash_attn_wheel) if args.flash_attn_wheel else None
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if not export_dir.exists():
        raise FileNotFoundError(export_dir)
    if flash_attn_wheel and not flash_attn_wheel.exists():
        raise FileNotFoundError(flash_attn_wheel)

    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_json(export_dir / "config.json")

    serve_args = [
        ".venv/bin/python",
        "scripts/serve_openai_api.py",
        "--weight_path",
        rel(checkpoint),
        "--hidden_size",
        str(args.hidden_size),
        "--num_hidden_layers",
        str(args.num_hidden_layers),
        "--num_attention_heads",
        str(args.num_attention_heads),
        "--num_key_value_heads",
        str(args.num_key_value_heads),
        "--intermediate_size",
        str(args.intermediate_size),
        "--max_seq_len",
        str(args.max_seq_len),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--dynamic_batching",
        "--batch_max_size",
        str(args.batch_max_size),
        "--batch_wait_ms",
        str(args.batch_wait_ms),
    ]
    serve_command = " ".join(shlex.quote(part) for part in serve_args)
    benchmark_command = " ".join(
        shlex.quote(part)
        for part in [
            ".venv/bin/python",
            "experiments/pretrain/scripts/benchmark_openai_api_latency.py",
            "--endpoint",
            f"http://127.0.0.1:{args.port}/v1/chat/completions",
            "--model",
            "minimind",
            "--output_dir",
            f"{rel(output_dir)}/api_latency",
            "--prompt_chars",
            "128",
            "512",
            "1024",
            "--concurrency",
            "1",
            "2",
            "4",
            "8",
            "--requests_per_case",
            "4",
            "--max_tokens",
            "64",
            "--no-stream",
        ]
    )

    created_at = datetime.now(timezone.utc).isoformat()
    checkpoint_record = file_record(checkpoint)
    manifest = {
        "name": args.name,
        "snapshot_id": args.name,
        "created_at": created_at,
        "git_rev": git_rev(),
        "checkpoint": checkpoint_record,
        "qwen_export": {
            "path": rel(export_dir),
            "config": config,
        },
        "append_logs": [],
        "serve": {
            "host": args.host,
            "port": args.port,
            "dynamic_batching": True,
            "batch_max_size": args.batch_max_size,
            "batch_wait_ms": args.batch_wait_ms,
            "command": serve_command,
        },
        "benchmark": {
            "command": benchmark_command,
            "output_dir": f"{rel(output_dir)}/api_latency",
        },
        "code": {
            "server": "scripts/serve_openai_api.py",
            "api_benchmark": "experiments/pretrain/scripts/benchmark_openai_api_latency.py",
            "nano_vllm_benchmark": "experiments/pretrain/scripts/benchmark_nanovllm.py",
            "nano_vllm_attention_patch": "experiments/pretrain/vendor/nano-vllm/nanovllm/layers/attention.py",
            "flash_attn_fallback": "experiments/pretrain/vendor/flash_attn_fallback/flash_attn/__init__.py",
        },
        "runtime_dependencies": {
            "flash_attn_wheel_candidate": file_record(flash_attn_wheel) if flash_attn_wheel else None,
        },
        "notes": [
            "This snapshot is for serving reproducibility, not a new training result.",
            "Native API dynamic batching is the current stable serving path.",
            "nano-vLLM fallback results are smoke evidence only until real flash_attn is installed and profiled.",
        ],
    }

    events = [
        {
            "seq": 1,
            "ts": created_at,
            "event": "create_snapshot",
            "snapshot_id": args.name,
        },
        {
            "seq": 2,
            "ts": created_at,
            "event": "register_checkpoint",
            **checkpoint_record,
        },
        {
            "seq": 3,
            "ts": created_at,
            "event": "configure_service",
            "serve": manifest["serve"],
        },
        {
            "seq": 4,
            "ts": created_at,
            "event": "register_benchmark",
            "benchmark": manifest["benchmark"],
        },
    ]
    if flash_attn_wheel:
        events.append(
            {
                "seq": len(events) + 1,
                "ts": created_at,
                "event": "register_flash_attn_wheel_candidate",
                **file_record(flash_attn_wheel),
                "trusted": False,
                "status": "downloaded_not_installed",
            }
        )
    append_log = output_dir / "append-000001.jsonl"
    write_append_log(append_log, events)
    manifest["append_logs"] = [
        {
            "path": rel(append_log),
            "sha256": sha256_file(append_log),
            "first_seq": 1,
            "last_seq": events[-1]["seq"],
        }
    ]
    atomic_write_json(output_dir / "manifest.json", manifest)
    atomic_write_json(output_dir / "current.json", manifest)
    write_executable(
        output_dir / "serve_dynamic_batch.sh",
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"\n'
        'cd "$ROOT"\n'
        f"exec {serve_command}\n",
    )
    write_executable(
        output_dir / "benchmark_api_latency.sh",
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"\n'
        'cd "$ROOT"\n'
        f"{benchmark_command}\n",
    )
    print(output_dir / "manifest.json")


if __name__ == "__main__":
    sys.exit(main())
