import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
STATUS_DIR = ROOT / "experiments" / "pretrain" / "status"
DEFAULT_OUTPUT_JSON = STATUS_DIR / "dashboard.json"
DEFAULT_OUTPUT_MD = STATUS_DIR / "dashboard.md"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_tsv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def maybe_float(value) -> float | None:
    try:
        if value in {None, ""}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def run_subprocess(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def collect_strict_summary(summary_tsv: Path) -> dict:
    rows = read_tsv(summary_tsv)
    candidates = []
    for row in rows:
        test_loss = maybe_float(row.get("test_loss"))
        val_loss = maybe_float(row.get("val_loss"))
        if test_loss is None:
            continue
        candidates.append({**row, "test_loss_float": test_loss, "val_loss_float": val_loss})
    candidates.sort(key=lambda row: (row["test_loss_float"], row["val_loss_float"] or 999))
    return {
        "summary_tsv": str(summary_tsv),
        "row_count": len(rows),
        "best_by_test_loss": candidates[:10],
    }


def collect_fixed_prompt_scores(runs_root: Path) -> dict:
    rows = []
    for path in sorted(runs_root.glob("*/samples/fixed_prompt_scores.json")):
        data = read_json(path)
        run_name = path.parents[1].name
        rows.append(
            {
                "run_name": run_name,
                "mean_score": data.get("mean_score"),
                "empty_count": data.get("empty_count"),
                "path": str(path),
            }
        )
    rows.sort(key=lambda row: (-(row["mean_score"] or 0), row["empty_count"] or 999))
    return {"row_count": len(rows), "best": rows[:10]}


def summarize_eval_tsv(path: Path) -> dict:
    rows = read_tsv(path)
    by_checkpoint: dict[str, list[float]] = {}
    for row in rows:
        loss = maybe_float(row.get("mean_loss"))
        if loss is None:
            continue
        by_checkpoint.setdefault(row.get("checkpoint", ""), []).append(loss)
    checkpoint_means = [
        {"checkpoint": checkpoint, "mean_loss": sum(values) / len(values), "buckets": len(values)}
        for checkpoint, values in by_checkpoint.items()
    ]
    checkpoint_means.sort(key=lambda row: row["mean_loss"])
    return {"path": str(path), "rows": len(rows), "checkpoint_means": checkpoint_means}


def collect_tsv_summaries(diagnostics_root: Path) -> list[dict]:
    return [summarize_eval_tsv(path) for path in sorted(diagnostics_root.glob("*/diagnostic_eval_summary.tsv"))]


def collect_mcq(diagnostics_root: Path) -> list[dict]:
    rows = []
    for mcq_root in sorted(diagnostics_root.glob("local_mcq*")):
        if not mcq_root.is_dir():
            continue
        for path in sorted(mcq_root.glob("*.json")):
            data = read_json(path)
            rows.append(
                {
                    "suite": mcq_root.name,
                    "checkpoint": path.stem,
                    "accuracy": data.get("accuracy"),
                    "correct": data.get("correct"),
                    "total": data.get("total"),
                    "path": str(path),
                }
            )
    rows.sort(key=lambda row: (row["suite"], -(row["accuracy"] or 0)))
    return rows


def collect_profiles(profile_root: Path) -> list[dict]:
    rows = []
    for path in sorted(profile_root.glob("*.json")):
        data = read_json(path)
        rows.append(
            {
                "name": path.stem,
                "mean_prefill_tps": data.get("mean_prefill_tps"),
                "mean_decode_tps": data.get("mean_decode_tps"),
                "max_memory_reserved_gb": data.get("max_memory_reserved_gb"),
                "path": str(path),
            }
        )
    rows.sort(key=lambda row: -(row["mean_decode_tps"] or 0))
    return rows


def collect_engineering_profiles(profile_root: Path) -> list[dict]:
    rows = []
    for path in sorted(profile_root.glob("*_stable.json")):
        data = read_json(path)
        for item in data.get("rows", []):
            rows.append(
                {
                    "name": path.stem,
                    "batch_size": item.get("batch_size"),
                    "prompt_tokens": item.get("prompt_tokens"),
                    "max_new_tokens": item.get("max_new_tokens"),
                    "use_cache": item.get("use_cache"),
                    "prefill_tokens_per_second": item.get("prefill_tokens_per_second"),
                    "decode_tokens_per_second": item.get("decode_tokens_per_second"),
                    "max_reserved_gb": item.get("max_reserved_gb"),
                    "path": str(path),
                }
            )
    rows.sort(key=lambda row: (row["name"], row["batch_size"] or 0, row["prompt_tokens"] or 0, not bool(row["use_cache"])))
    return rows


def collect_api_latency(latency_root: Path) -> list[dict]:
    rows = []
    for path in sorted(latency_root.glob("*/api_latency_summary.json")):
        run_name = path.parent.name
        for item in read_json(path):
            rows.append(
                {
                    "run_name": run_name,
                    "prompt_chars": item.get("prompt_chars"),
                    "concurrency": item.get("concurrency"),
                    "max_tokens": item.get("max_tokens"),
                    "stream": item.get("stream"),
                    "requests": item.get("requests"),
                    "errors": item.get("errors"),
                    "median_ttft_ms": item.get("median_ttft_ms"),
                    "p90_ttft_ms": item.get("p90_ttft_ms"),
                    "median_total_ms": item.get("median_total_ms"),
                    "p90_total_ms": item.get("p90_total_ms"),
                    "path": str(path),
                }
            )
    rows.sort(key=lambda row: (row["run_name"], row["prompt_chars"] or 0, row["concurrency"] or 0))
    return rows


def format_float(value, digits: int = 1) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def write_markdown(path: Path, dashboard: dict) -> None:
    lines = [
        "# MiniMind Pretrain Dashboard",
        "",
        "This file is generated by `experiments/pretrain/scripts/refresh_pretrain_dashboard.py`.",
        "",
        "## Strict Loss",
        "",
        "| rank | run | test loss | val loss | status |",
        "| ---: | --- | ---: | ---: | --- |",
    ]
    for index, row in enumerate(dashboard["strict"]["best_by_test_loss"][:8], start=1):
        lines.append(
            f"| {index} | `{row.get('run_name', '')}` | `{row.get('test_loss', '')}` | `{row.get('val_loss', '')}` | `{row.get('status', '')}` |"
        )

    lines.extend(["", "## Fixed Prompt Smoke", "", "| rank | run | score | empty |", "| ---: | --- | ---: | ---: |"])
    for index, row in enumerate(dashboard["fixed_prompt"]["best"][:8], start=1):
        lines.append(f"| {index} | `{row['run_name']}` | `{row['mean_score']}` | `{row['empty_count']}` |")

    lines.extend(["", "## Diagnostic / Probe PPL", ""])
    for summary in dashboard["diagnostics"]:
        lines.append(f"### `{Path(summary['path']).parent.name}`")
        lines.extend(["", "| rank | checkpoint | mean loss | buckets |", "| ---: | --- | ---: | ---: |"])
        for index, row in enumerate(summary["checkpoint_means"][:6], start=1):
            lines.append(f"| {index} | `{row['checkpoint']}` | `{row['mean_loss']:.4f}` | `{row['buckets']}` |")
        lines.append("")

    if dashboard["mcq"]:
        lines.extend(["## MCQ", "", "| suite | checkpoint | accuracy | correct/total |", "| --- | --- | ---: | ---: |"])
        for row in dashboard["mcq"]:
            lines.append(f"| `{row['suite']}` | `{row['checkpoint']}` | `{row['accuracy']:.4f}` | `{row['correct']}/{row['total']}` |")
        lines.append("")

    if dashboard["profiles"]:
        lines.extend(["## Inference Profiles", "", "| name | prefill tok/s | decode tok/s | max reserved GB |", "| --- | ---: | ---: | ---: |"])
        for row in dashboard["profiles"]:
            lines.append(
                f"| `{row['name']}` | `{row['mean_prefill_tps']:.1f}` | `{row['mean_decode_tps']:.1f}` | `{row['max_memory_reserved_gb']:.2f}` |"
            )
        lines.append("")

    if dashboard["engineering_profiles"]:
        lines.extend(
            [
                "## Engineering Profiles",
                "",
                "| name | batch | prompt | new | cache | prefill tok/s | decode tok/s | max reserved GB |",
                "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
            ]
        )
        for row in dashboard["engineering_profiles"]:
            lines.append(
                f"| `{row['name']}` | `{row['batch_size']}` | `{row['prompt_tokens']}` | `{row['max_new_tokens']}` | `{row['use_cache']}` | `{format_float(row['prefill_tokens_per_second'])}` | `{format_float(row['decode_tokens_per_second'])}` | `{format_float(row['max_reserved_gb'], 2)}` |"
            )
        lines.append("")

    if dashboard["api_latency"]:
        lines.extend(
            [
                "## API Latency",
                "",
                "| run | prompt chars | concurrency | max tokens | errors | median TTFT ms | p90 TTFT ms | median total ms | p90 total ms |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in dashboard["api_latency"]:
            lines.append(
                f"| `{row['run_name']}` | `{row['prompt_chars']}` | `{row['concurrency']}` | `{row['max_tokens']}` | `{row['errors']}` | `{format_float(row['median_ttft_ms'])}` | `{format_float(row['p90_ttft_ms'])}` | `{format_float(row['median_total_ms'])}` | `{format_float(row['p90_total_ms'])}` |"
            )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh one dashboard for pretrain experiments.")
    parser.add_argument("--output_json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output_md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--skip_resummarize", action="store_true")
    args = parser.parse_args()

    if not args.skip_resummarize:
        run_subprocess([sys.executable, str(ROOT / "experiments" / "pretrain" / "scripts" / "summarize_platform_runs.py")])
        run_subprocess([sys.executable, str(ROOT / "experiments" / "pretrain" / "scripts" / "score_fixed_prompt_samples.py"), str(ROOT / "experiments" / "pretrain" / "runs")])

    dashboard = {
        "strict": collect_strict_summary(STATUS_DIR / "summary.tsv"),
        "fixed_prompt": collect_fixed_prompt_scores(ROOT / "experiments" / "pretrain" / "runs"),
        "diagnostics": collect_tsv_summaries(ROOT / "experiments" / "pretrain" / "diagnostics"),
        "mcq": collect_mcq(ROOT / "experiments" / "pretrain" / "diagnostics"),
        "profiles": collect_profiles(ROOT / "experiments" / "pretrain" / "diagnostics" / "inference_profiles"),
        "engineering_profiles": collect_engineering_profiles(ROOT / "experiments" / "pretrain" / "diagnostics" / "engineering_profiles"),
        "api_latency": collect_api_latency(ROOT / "experiments" / "pretrain" / "diagnostics" / "api_latency"),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(dashboard, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.output_md, dashboard)
    print(args.output_md)


if __name__ == "__main__":
    main()
