import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPLITS_DIR = ROOT / "experiments" / "pretrain" / "diagnostic_splits"
DEFAULT_OUTPUT_DIR = ROOT / "experiments" / "pretrain" / "diagnostics" / datetime.now().strftime("%Y%m%d_%H%M%S")
DEFAULT_CHECKPOINTS = {
    "ffn3072_seq1024": ROOT
    / "experiments"
    / "pretrain"
    / "platform_runs"
    / "ffn3072_cooldown_seq1024"
    / "02_seq1024_lr5e6"
    / "ffn3072_cooldown_seq1024_seq1024_lr5e6_768.pth",
    "avg50_control": ROOT
    / "experiments"
    / "pretrain"
    / "platform_runs"
    / "checkpoint_averages"
    / "base_control"
    / "base50_control50_768.pth",
    "quality100_s384": ROOT
    / "experiments"
    / "pretrain"
    / "platform_runs"
    / "pretrain_avg50_quality100_s384_b24_lr1e5"
    / "01_continue"
    / "pretrain_avg50_quality100_s384_b24_lr1e5_continue_768.pth",
    "quality100_s512": ROOT
    / "experiments"
    / "pretrain"
    / "platform_runs"
    / "pretrain_q100_stage2_quality_s512_b16_lr5e6"
    / "01_continue"
    / "pretrain_q100_stage2_quality_s512_b16_lr5e6_continue_768.pth",
}


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must use name=path")
    name, raw_path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("checkpoint name is empty")
    path = Path(raw_path.strip())
    if not path.is_absolute():
        path = ROOT / path
    return name, path


def command_text(command: list[str]) -> str:
    return " ".join(str(item) for item in command)


def load_splits(splits_dir: Path, include: list[str] | None, exclude: set[str]) -> list[tuple[str, Path]]:
    if include:
        split_paths = [(name, splits_dir / f"{name}.jsonl") for name in include]
    else:
        split_paths = [(path.stem, path) for path in sorted(splits_dir.glob("*.jsonl"))]
    return [(name, path) for name, path in split_paths if name not in exclude and path.exists()]


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate several MiniMind checkpoints on diagnostic holdout buckets.")
    parser.add_argument("--splits_dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", action="append", type=parse_checkpoint, default=[])
    parser.add_argument("--include_split", action="append", default=[])
    parser.add_argument("--exclude_split", action="append", default=[])
    parser.add_argument("--tokenizer_path", type=Path, default=ROOT / "model")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--num_key_value_heads", type=int, default=8)
    parser.add_argument("--intermediate_size", type=int, default=3072)
    parser.add_argument("--max_seq_len", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_batches", type=int, default=80)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    splits_dir = args.splits_dir if args.splits_dir.is_absolute() else ROOT / args.splits_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    tokenizer_path = args.tokenizer_path if args.tokenizer_path.is_absolute() else ROOT / args.tokenizer_path
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = dict(args.checkpoint)
    if not checkpoints:
        checkpoints = {name: path for name, path in DEFAULT_CHECKPOINTS.items() if path.exists()}
    missing_checkpoints = {name: str(path) for name, path in checkpoints.items() if not path.exists()}
    checkpoints = {name: path for name, path in checkpoints.items() if path.exists()}
    if not checkpoints:
        raise SystemExit("no checkpoint paths exist")

    splits = load_splits(splits_dir, args.include_split or None, set(args.exclude_split or []))
    if not splits:
        raise SystemExit(f"no diagnostic splits found in {splits_dir}")

    eval_script = ROOT / "experiments" / "pretrain" / "scripts" / "eval_pretrain_loss.py"
    rows = []
    manifest = {
        "created_at": now_text(),
        "splits_dir": rel(splits_dir),
        "output_dir": rel(output_dir),
        "settings": {
            "hidden_size": args.hidden_size,
            "num_hidden_layers": args.num_hidden_layers,
            "num_attention_heads": args.num_attention_heads,
            "num_key_value_heads": args.num_key_value_heads,
            "intermediate_size": args.intermediate_size,
            "max_seq_len": args.max_seq_len,
            "batch_size": args.batch_size,
            "max_batches": args.max_batches,
            "dtype": args.dtype,
        },
        "missing_checkpoints": missing_checkpoints,
        "results": [],
    }

    for ckpt_name, ckpt_path in checkpoints.items():
        for split_name, split_path in splits:
            eval_dir = output_dir / ckpt_name
            output_path = eval_dir / f"{split_name}.json"
            log_path = eval_dir / f"{split_name}.log"
            command = [
                sys.executable,
                str(eval_script),
                "--data_path",
                str(split_path),
                "--weight_path",
                str(ckpt_path),
                "--output",
                str(output_path),
                "--tokenizer_path",
                str(tokenizer_path),
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
                "--batch_size",
                str(args.batch_size),
                "--max_batches",
                str(args.max_batches),
                "--dtype",
                args.dtype,
            ]
            result = {
                "checkpoint": ckpt_name,
                "checkpoint_path": rel(ckpt_path),
                "split": split_name,
                "split_path": rel(split_path),
                "output": rel(output_path),
                "log": rel(log_path),
                "command": command,
            }
            if output_path.exists() and not args.overwrite:
                metrics = load_metrics(output_path)
                status = "existing"
            else:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                started = time.time()
                with log_path.open("w", encoding="utf-8") as log:
                    log.write("COMMAND: " + command_text(command) + "\n\n")
                    log.flush()
                    code = subprocess.run(command, cwd=ROOT, text=True, stdout=log, stderr=subprocess.STDOUT).returncode
                status = "ok" if code == 0 and output_path.exists() else f"failed_{code}"
                metrics = load_metrics(output_path) if output_path.exists() else {}
                result["seconds_total"] = round(time.time() - started, 3)
            result["status"] = status
            result["metrics"] = metrics
            rows.append(
                {
                    "checkpoint": ckpt_name,
                    "split": split_name,
                    "status": status,
                    "mean_loss": metrics.get("mean_loss", ""),
                    "ppl": metrics.get("ppl", ""),
                    "tokens": metrics.get("tokens", ""),
                    "batches": metrics.get("batches", ""),
                    "seconds": metrics.get("seconds", ""),
                    "checkpoint_path": rel(ckpt_path),
                    "split_path": rel(split_path),
                }
            )
            manifest["results"].append(result)
            print(f"{ckpt_name}\t{split_name}\t{status}\t{metrics.get('mean_loss', '')}", flush=True)

    tsv_path = output_dir / "diagnostic_eval_summary.tsv"
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    manifest["summary_tsv"] = rel(tsv_path)
    manifest["finished_at"] = now_text()
    (output_dir / "diagnostic_eval_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(tsv_path)


if __name__ == "__main__":
    main()
