import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PLATFORM_ROOT = ROOT / "experiments" / "pretrain" / "platform_runs"
DEFAULT_BUNDLE_ROOT = ROOT / "experiments" / "pretrain" / "runs"
DEFAULT_OUTPUT = ROOT / "experiments" / "pretrain" / "status" / "summary.tsv"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def first_summary(run_dir: Path) -> dict:
    summaries = sorted(run_dir.glob("**/*_summary.json"))
    return read_json(summaries[0]) if summaries else {}


def eval_metrics(run_dir: Path, prefix: str) -> tuple[str, str, str]:
    candidates = [
        run_dir / f"eval_{prefix}.json",
        run_dir / "evals" / f"{prefix}.json",
    ]
    for path in candidates:
        if path.exists():
            data = read_json(path)
            return (
                f"{data.get('mean_loss', ''):.4f}" if isinstance(data.get("mean_loss"), (int, float)) else "",
                f"{data.get('ppl', ''):.2f}" if isinstance(data.get("ppl"), (int, float)) else "",
                str(data.get("tokens", "")),
            )
    return "", "", ""


def manifest_status(run_dir: Path) -> str:
    manifest = run_dir / "manifest.json"
    if manifest.exists():
        data = read_json(manifest)
        evals = data.get("evals", [])
        samples = data.get("samples", {})
        bad = [item.get("status") for item in evals if item.get("status") not in {"ok", "existing"}]
        if bad:
            return "eval_issue"
        if samples and samples.get("status") not in {"ok", "existing"}:
            return "sample_issue"
        return "bundled"
    if (run_dir / "eval_strict_test.json").exists():
        return "done"
    if (run_dir / "runner_console.log").exists():
        return "has_log"
    return "unknown"


def collect_rows(platform_root: Path, bundle_root: Path) -> list[dict]:
    rows = []
    seen = set()
    for root, source in [(platform_root, "platform"), (bundle_root, "bundle")]:
        if not root.exists():
            continue
        for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            key = (source, run_dir.name)
            if key in seen:
                continue
            seen.add(key)
            summary = first_summary(run_dir)
            val_loss, val_ppl, val_tokens = eval_metrics(run_dir, "strict_val")
            test_loss, test_ppl, test_tokens = eval_metrics(run_dir, "strict_test")
            rows.append({
                "source": source,
                "run_name": run_dir.name,
                "status": manifest_status(run_dir),
                "hidden": str(summary.get("hidden_size", "")),
                "layers": str(summary.get("num_hidden_layers", "")),
                "heads": str(summary.get("num_attention_heads", "")),
                "kv_heads": str(summary.get("num_key_value_heads", "")),
                "ffn": str(summary.get("intermediate_size", "")),
                "seq": str(summary.get("max_seq_len", "")),
                "steps": str(summary.get("steps", "")),
                "lr": str(summary.get("learning_rate", "")),
                "schedule": str(summary.get("lr_schedule", "")),
                "val_loss": val_loss,
                "val_ppl": val_ppl,
                "val_tokens": val_tokens,
                "test_loss": test_loss,
                "test_ppl": test_ppl,
                "test_tokens": test_tokens,
                "tok_s": f"{summary.get('slot_tokens_per_second', ''):.0f}" if isinstance(summary.get("slot_tokens_per_second"), (int, float)) else "",
                "mem_gb": f"{summary.get('max_memory_reserved_gb', ''):.2f}" if isinstance(summary.get("max_memory_reserved_gb"), (int, float)) else "",
            })
    return rows


def write_tsv(path: Path, rows: list[dict]) -> None:
    fields = [
        "source", "run_name", "status", "hidden", "layers", "heads", "kv_heads", "ffn", "seq", "steps",
        "lr", "schedule", "val_loss", "val_ppl", "val_tokens", "test_loss", "test_ppl", "test_tokens", "tok_s", "mem_gb",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["\t".join(fields)]
    for row in rows:
        lines.append("\t".join(row.get(field, "") for field in fields))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize MiniMind platform and bundled evaluation runs into a TSV status table.")
    parser.add_argument("--platform_root", type=Path, default=DEFAULT_PLATFORM_ROOT)
    parser.add_argument("--bundle_root", type=Path, default=DEFAULT_BUNDLE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    platform_root = args.platform_root if args.platform_root.is_absolute() else ROOT / args.platform_root
    bundle_root = args.bundle_root if args.bundle_root.is_absolute() else ROOT / args.bundle_root
    output = args.output if args.output.is_absolute() else ROOT / args.output
    rows = collect_rows(platform_root, bundle_root)
    write_tsv(output, rows)
    print(f"wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
