import argparse
import json
import math
import random
import subprocess
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


DEFAULT_PROMPTS = ROOT / "experiments" / "pretrain" / "eval_suites" / "fixed_prompts.jsonl"
DEFAULT_MCQ = ROOT / "experiments" / "pretrain" / "eval_suites" / "local_mcq_benchmark_v1.jsonl"
DEFAULT_OUTPUT_ROOT = ROOT / "experiments" / "pretrain" / "runs"
DEFAULT_EVAL_SETS = [
    {
        "name": "strict_val",
        "data_path": ROOT / "experiments" / "pretrain" / "strict_splits" / "pretrain_strict_val_2k.jsonl",
        "max_batches": 250,
    },
    {
        "name": "strict_test",
        "data_path": ROOT / "experiments" / "pretrain" / "strict_splits" / "pretrain_strict_test_2k.jsonl",
        "max_batches": 250,
    },
]


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_prompts(path: Path) -> list[dict]:
    prompts = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "id" not in item or "prompt" not in item:
                raise ValueError(f"Prompt line {line_number} must include id and prompt.")
            prompts.append(item)
    return prompts


def command_text(command: list[str]) -> str:
    return " ".join(str(part) for part in command)


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def run_loss_eval(args: argparse.Namespace, run_dir: Path, manifest: dict) -> None:
    eval_script = ROOT / "experiments" / "pretrain" / "scripts" / "eval_pretrain_loss.py"
    eval_dir = run_dir / "evals"
    eval_results = []
    for item in DEFAULT_EVAL_SETS:
        output_path = eval_dir / f"{item['name']}.json"
        log_path = eval_dir / f"{item['name']}.log"
        command = [
            sys.executable,
            str(eval_script),
            "--data_path",
            str(item["data_path"]),
            "--weight_path",
            str(args.weight_path),
            "--output",
            str(output_path),
            "--tokenizer_path",
            str(args.tokenizer_path),
            "--hidden_size",
            str(args.hidden_size),
            "--num_hidden_layers",
            str(args.num_hidden_layers),
            "--num_attention_heads",
            str(args.num_attention_heads),
            "--num_key_value_heads",
            str(args.num_key_value_heads),
            "--max_seq_len",
            str(args.eval_seq_len),
            "--batch_size",
            str(args.eval_batch_size),
            "--max_batches",
            str(item["max_batches"]),
            "--dtype",
            args.dtype,
        ]
        if args.intermediate_size is not None:
            command.extend(["--intermediate_size", str(args.intermediate_size)])

        result = {
            "name": item["name"],
            "data_path": rel(item["data_path"]),
            "output": rel(output_path),
            "log": rel(log_path),
            "command": command,
        }
        if output_path.exists() and not args.overwrite:
            result["status"] = "existing"
            result["metrics"] = read_json(output_path)
        elif not item["data_path"].exists():
            result["status"] = "missing_data"
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("w", encoding="utf-8") as log:
                log.write("COMMAND: " + command_text(command) + "\n\n")
                log.flush()
                code = subprocess.run(command, cwd=ROOT, text=True, stdout=log, stderr=subprocess.STDOUT).returncode
            result["status"] = "ok" if code == 0 and output_path.exists() else f"failed_{code}"
            if output_path.exists():
                result["metrics"] = read_json(output_path)
        eval_results.append(result)
    manifest["evals"] = eval_results


def run_mcq_eval(args: argparse.Namespace, run_dir: Path, manifest: dict) -> None:
    output_path = run_dir / "mcq" / "local_mcq_benchmark_v1.json"
    log_path = run_dir / "mcq" / "local_mcq_benchmark_v1.log"
    command = [
        sys.executable,
        str(ROOT / "experiments" / "pretrain" / "scripts" / "eval_mcq_loglikelihood.py"),
        "--data_path",
        str(args.mcq_data_path),
        "--weight_path",
        str(args.weight_path),
        "--output",
        str(output_path),
        "--tokenizer_path",
        str(args.tokenizer_path),
        "--hidden_size",
        str(args.hidden_size),
        "--num_hidden_layers",
        str(args.num_hidden_layers),
        "--num_attention_heads",
        str(args.num_attention_heads),
        "--num_key_value_heads",
        str(args.num_key_value_heads),
        "--max_seq_len",
        str(args.mcq_seq_len),
        "--dtype",
        args.dtype,
    ]
    if args.intermediate_size is not None:
        command.extend(["--intermediate_size", str(args.intermediate_size)])

    result = {
        "name": "local_mcq_benchmark_v1",
        "data_path": rel(args.mcq_data_path),
        "output": rel(output_path),
        "log": rel(log_path),
        "command": command,
    }
    if output_path.exists() and not args.overwrite:
        result["status"] = "existing"
        result["metrics"] = read_json(output_path)
    elif not args.mcq_data_path.exists():
        result["status"] = "missing_data"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            log.write("COMMAND: " + command_text(command) + "\n\n")
            log.flush()
            code = subprocess.run(command, cwd=ROOT, text=True, stdout=log, stderr=subprocess.STDOUT).returncode
        result["status"] = "ok" if code == 0 and output_path.exists() else f"failed_{code}"
        if output_path.exists():
            result["metrics"] = read_json(output_path)
    manifest["mcq"] = result


def run_fixed_prompt_scores(run_dir: Path, manifest: dict, overwrite: bool) -> None:
    sample_path = run_dir / "samples" / "fixed_prompts.jsonl"
    score_path = run_dir / "samples" / "fixed_prompt_scores.json"
    log_path = run_dir / "samples" / "fixed_prompt_scores.log"
    if not sample_path.exists():
        manifest["fixed_prompt_score"] = {"status": "missing_samples"}
        return
    if score_path.exists() and not overwrite:
        manifest["fixed_prompt_score"] = {"status": "existing", "metrics": read_json(score_path)}
        return

    command = [
        sys.executable,
        str(ROOT / "experiments" / "pretrain" / "scripts" / "score_fixed_prompt_samples.py"),
        str(sample_path),
    ]
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND: " + command_text(command) + "\n\n")
        log.flush()
        code = subprocess.run(command, cwd=ROOT, text=True, stdout=log, stderr=subprocess.STDOUT).returncode
    manifest["fixed_prompt_score"] = {
        "status": "ok" if code == 0 and score_path.exists() else f"failed_{code}",
        "output": rel(score_path),
        "log": rel(log_path),
    }
    if score_path.exists():
        manifest["fixed_prompt_score"]["metrics"] = read_json(score_path)


def metric_value(manifest: dict, eval_name: str, key: str) -> str:
    for item in manifest.get("evals", []):
        if item.get("name") == eval_name:
            value = item.get("metrics", {}).get(key)
            return f"{value:.6g}" if isinstance(value, (int, float)) else "-"
    return "-"


def write_report(run_dir: Path, manifest: dict) -> None:
    mcq = manifest.get("mcq", {}).get("metrics", {})
    prompt_score = manifest.get("fixed_prompt_score", {}).get("metrics", {})
    lines = [
        f"# Checkpoint Evaluation: {manifest['run_name']}",
        "",
        f"- weight: `{manifest['weight_path']}`",
        f"- strict val loss: `{metric_value(manifest, 'strict_val', 'mean_loss')}`",
        f"- strict test loss: `{metric_value(manifest, 'strict_test', 'mean_loss')}`",
        f"- MCQ: `{mcq.get('correct', '-')}/{mcq.get('total', '-')}`",
        f"- fixed prompt score: `{prompt_score.get('mean_score', '-')}`",
        "",
    ]
    if mcq.get("category_summary"):
        lines.extend(["## MCQ by Category", "", "| category | correct | total | acc |", "| --- | ---: | ---: | ---: |"])
        for category, item in mcq["category_summary"].items():
            lines.append(f"| {category} | {item['correct']} | {item['total']} | {item['accuracy']:.3f} |")
        lines.append("")
    report_path = run_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    manifest["report"] = rel(report_path)


@torch.no_grad()
def run_fixed_prompts(args: argparse.Namespace, run_dir: Path, manifest: dict) -> None:
    samples_dir = run_dir / "samples"
    jsonl_path = samples_dir / "fixed_prompts.jsonl"
    md_path = samples_dir / "fixed_prompts.md"
    if jsonl_path.exists() and md_path.exists() and not args.overwrite:
        manifest["samples"] = {
            "status": "existing",
            "jsonl": rel(jsonl_path),
            "markdown": rel(md_path),
        }
        return

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if "cuda" in device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = torch.amp.autocast("cuda", dtype=dtype) if device_type == "cuda" else nullcontext()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    config_kwargs = {
        "hidden_size": args.hidden_size,
        "num_hidden_layers": args.num_hidden_layers,
        "num_attention_heads": args.num_attention_heads,
        "num_key_value_heads": args.num_key_value_heads,
        "max_position_embeddings": max(args.max_input_tokens + args.max_new_tokens + 8, 2048),
    }
    if args.intermediate_size is not None:
        config_kwargs["intermediate_size"] = args.intermediate_size
    config = MiniMindConfig(**config_kwargs)
    model = MiniMindForCausalLM(config)
    model.load_state_dict(torch.load(args.weight_path, map_location="cpu"), strict=True)
    model = model.to(device).eval()

    prompts = read_prompts(args.prompts_path)
    rows = []
    md_lines = [
        f"# Fixed Prompt Samples: {args.run_name}",
        "",
        f"- generated_at: `{now_text()}`",
        f"- weight_path: `{rel(args.weight_path)}`",
        f"- decoding: `do_sample={args.do_sample}, temperature={args.temperature}, top_p={args.top_p}, top_k={args.top_k}, max_new_tokens={args.max_new_tokens}`",
        "",
    ]

    started = time.time()
    for item in prompts:
        prompt_text = item["prompt"]
        if args.prompt_format == "pretrain":
            text = (tokenizer.bos_token or "") + prompt_text
        elif args.prompt_format == "chat":
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = prompt_text
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_input_tokens).to(device)
        before = time.time()
        with autocast_ctx:
            generated_ids = model.generate(
                inputs=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        input_len = inputs["input_ids"].shape[1]
        completion_ids = generated_ids[0][input_len:]
        completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
        row = {
            "id": item["id"],
            "category": item.get("category", ""),
            "prompt": prompt_text,
            "completion": completion,
            "input_tokens": input_len,
            "generated_tokens": int(completion_ids.numel()),
            "seconds": round(time.time() - before, 3),
        }
        rows.append(row)
        md_lines.extend([
            f"## {item['id']} ({item.get('category', '')})",
            "",
            "**Prompt**",
            "",
            prompt_text,
            "",
            "**Completion**",
            "",
            completion if completion else "<empty>",
            "",
        ])

    samples_dir.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    manifest["samples"] = {
        "status": "ok",
        "jsonl": rel(jsonl_path),
        "markdown": rel(md_path),
        "prompt_count": len(rows),
        "seconds": round(time.time() - started, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one MiniMind checkpoint with loss evals, fixed prompts, and a run manifest.")
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--weight_path", type=Path, required=True)
    parser.add_argument("--source_run_dir", type=Path, default=None)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--tokenizer_path", type=Path, default=ROOT / "model")
    parser.add_argument("--prompts_path", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--mcq_data_path", type=Path, default=DEFAULT_MCQ)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--num_key_value_heads", type=int, default=4)
    parser.add_argument("--intermediate_size", type=int, default=None)
    parser.add_argument("--eval_seq_len", type=int, default=384)
    parser.add_argument("--mcq_seq_len", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--max_input_tokens", type=int, default=384)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--prompt_format", choices=["pretrain", "raw", "chat"], default="pretrain")
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_loss", action="store_true")
    parser.add_argument("--skip_mcq", action="store_true")
    parser.add_argument("--skip_samples", action="store_true")
    parser.add_argument("--skip_prompt_score", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.weight_path = args.weight_path if args.weight_path.is_absolute() else ROOT / args.weight_path
    args.output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    args.tokenizer_path = args.tokenizer_path if args.tokenizer_path.is_absolute() else ROOT / args.tokenizer_path
    args.prompts_path = args.prompts_path if args.prompts_path.is_absolute() else ROOT / args.prompts_path
    args.mcq_data_path = args.mcq_data_path if args.mcq_data_path.is_absolute() else ROOT / args.mcq_data_path
    if args.source_run_dir is not None and not args.source_run_dir.is_absolute():
        args.source_run_dir = ROOT / args.source_run_dir

    if not args.weight_path.exists():
        raise SystemExit(f"missing weight: {args.weight_path}")
    run_dir = args.output_root / args.run_name
    manifest = {
        "schema_version": 1,
        "run_name": args.run_name,
        "created_at": now_text(),
        "git_commit": git_commit(),
        "weight_path": rel(args.weight_path),
        "source_run_dir": rel(args.source_run_dir) if args.source_run_dir else None,
        "model": {
            "hidden_size": args.hidden_size,
            "num_hidden_layers": args.num_hidden_layers,
            "num_attention_heads": args.num_attention_heads,
            "num_key_value_heads": args.num_key_value_heads,
            "intermediate_size": args.intermediate_size,
        },
        "decoding": {
            "prompt_format": args.prompt_format,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
        },
    }
    if not args.skip_loss:
        run_loss_eval(args, run_dir, manifest)
    if not args.skip_mcq:
        run_mcq_eval(args, run_dir, manifest)
    if not args.skip_samples:
        run_fixed_prompts(args, run_dir, manifest)
    if not args.skip_prompt_score:
        run_fixed_prompt_scores(run_dir, manifest, args.overwrite)
    manifest["finished_at"] = now_text()
    write_report(run_dir, manifest)
    write_json(run_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
