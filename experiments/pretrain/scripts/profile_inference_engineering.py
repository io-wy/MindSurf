import argparse
import csv
import json
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: E402


SEED_TEXT = (
    "MiniMind engineering benchmark prompt. "
    "This text is repeated to create a stable synthetic context for profiling prefill, decode, and KV cache behavior. "
    "请用简洁的中文解释模型推理服务中的预填充、解码和 KV cache。 "
)


def synchronize(device: str) -> None:
    if "cuda" in device and torch.cuda.is_available():
        torch.cuda.synchronize()


def make_input_ids(tokenizer, target_tokens: int, batch_size: int, device: str) -> torch.Tensor:
    token_ids: list[int] = []
    while len(token_ids) < target_tokens:
        token_ids.extend(tokenizer(SEED_TEXT, add_special_tokens=False).input_ids)
    token_ids = token_ids[:target_tokens]
    input_ids = torch.tensor([token_ids] * batch_size, dtype=torch.long, device=device)
    return input_ids


def kv_cache_bytes(past_key_values) -> int:
    if not past_key_values:
        return 0
    total = 0
    for layer in past_key_values:
        if layer is None:
            continue
        for tensor in layer:
            if tensor is not None:
                total += tensor.numel() * tensor.element_size()
    return total


def memory_snapshot(device: str) -> dict:
    if "cuda" not in device or not torch.cuda.is_available():
        return {"max_allocated_gb": 0.0, "max_reserved_gb": 0.0}
    return {
        "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "max_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
    }


@torch.inference_mode()
def run_cached_decode(model, input_ids, new_tokens: int, dtype_ctx, device: str) -> dict:
    attention_mask = torch.ones_like(input_ids)
    synchronize(device)
    started = time.perf_counter()
    with dtype_ctx:
        outputs = model(input_ids, attention_mask=attention_mask, use_cache=True, logits_to_keep=1)
    synchronize(device)
    prefill_seconds = time.perf_counter() - started
    past_key_values = outputs.past_key_values
    generated = input_ids
    decode_seconds = 0.0

    for _ in range(new_tokens):
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], dim=1)
        synchronize(device)
        before = time.perf_counter()
        with dtype_ctx:
            outputs = model(
                next_token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
        synchronize(device)
        decode_seconds += time.perf_counter() - before
        past_key_values = outputs.past_key_values

    return {
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "generated_tokens": new_tokens,
        "kv_cache_mb_final": kv_cache_bytes(past_key_values) / 1024**2,
    }


@torch.inference_mode()
def run_uncached_decode(model, input_ids, new_tokens: int, dtype_ctx, device: str) -> dict:
    attention_mask = torch.ones_like(input_ids)
    generated = input_ids
    synchronize(device)
    started = time.perf_counter()
    with dtype_ctx:
        outputs = model(generated, attention_mask=attention_mask, use_cache=False, logits_to_keep=1)
    synchronize(device)
    first_seconds = time.perf_counter() - started
    decode_seconds = 0.0

    for _ in range(new_tokens):
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], dim=1)
        synchronize(device)
        before = time.perf_counter()
        with dtype_ctx:
            outputs = model(generated, attention_mask=attention_mask, use_cache=False, logits_to_keep=1)
        synchronize(device)
        decode_seconds += time.perf_counter() - before

    return {
        "prefill_seconds": first_seconds,
        "decode_seconds": decode_seconds,
        "generated_tokens": new_tokens,
        "kv_cache_mb_final": 0.0,
    }


def load_model(args, device: str):
    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=max(max(args.prompt_tokens) + args.max_new_tokens + 8, 2048),
    )
    model = MiniMindForCausalLM(config)
    state = torch.load(args.weight_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    if args.dtype != "float32":
        model = model.to({"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype])
    return model


def profile_case_once(model, tokenizer, args, device: str, batch_size: int, prompt_tokens: int, use_cache: bool) -> dict:
    if "cuda" in device and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    dtype_ctx = torch.amp.autocast("cuda", dtype=dtype) if "cuda" in device and args.dtype != "float32" else nullcontext()
    input_ids = make_input_ids(tokenizer, prompt_tokens, batch_size, device)
    if use_cache:
        result = run_cached_decode(model, input_ids, args.max_new_tokens, dtype_ctx, device)
    else:
        result = run_uncached_decode(model, input_ids, args.max_new_tokens, dtype_ctx, device)
    mem = memory_snapshot(device)
    total_prompt_tokens = batch_size * prompt_tokens
    total_decode_tokens = batch_size * args.max_new_tokens
    return {
        "run_name": args.run_name,
        "weight_path": args.weight_path,
        "device": device,
        "dtype": args.dtype,
        "batch_size": batch_size,
        "prompt_tokens": prompt_tokens,
        "max_new_tokens": args.max_new_tokens,
        "use_cache": use_cache,
        "prefill_seconds": result["prefill_seconds"],
        "decode_seconds": result["decode_seconds"],
        "prefill_tokens_per_second": total_prompt_tokens / max(result["prefill_seconds"], 1e-9),
        "decode_tokens_per_second": total_decode_tokens / max(result["decode_seconds"], 1e-9),
        "kv_cache_mb_final": result["kv_cache_mb_final"],
        **mem,
    }


def profile_case(model, tokenizer, args, device: str, batch_size: int, prompt_tokens: int, use_cache: bool) -> dict:
    repeats = []
    for _ in range(args.repeats):
        repeats.append(profile_case_once(model, tokenizer, args, device, batch_size, prompt_tokens, use_cache))
    prefill_seconds = statistics.median(row["prefill_seconds"] for row in repeats)
    decode_seconds = statistics.median(row["decode_seconds"] for row in repeats)
    total_prompt_tokens = batch_size * prompt_tokens
    total_decode_tokens = batch_size * args.max_new_tokens
    first = {k: v for k, v in repeats[0].items() if k not in {
        "prefill_seconds",
        "decode_seconds",
        "prefill_tokens_per_second",
        "decode_tokens_per_second",
        "kv_cache_mb_final",
        "max_allocated_gb",
        "max_reserved_gb",
    }}
    return {
        **first,
        "repeats": args.repeats,
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "prefill_tokens_per_second": total_prompt_tokens / max(prefill_seconds, 1e-9),
        "decode_tokens_per_second": total_decode_tokens / max(decode_seconds, 1e-9),
        "kv_cache_mb_final": max(row["kv_cache_mb_final"] for row in repeats),
        "max_allocated_gb": max(row["max_allocated_gb"] for row in repeats),
        "max_reserved_gb": max(row["max_reserved_gb"] for row in repeats),
        "raw_repeats": repeats,
    }


def warmup(model, tokenizer, args, device: str) -> None:
    if args.warmup <= 0:
        return
    original_new_tokens = args.max_new_tokens
    args.max_new_tokens = min(args.max_new_tokens, 8)
    for _ in range(args.warmup):
        profile_case_once(model, tokenizer, args, device, batch_size=1, prompt_tokens=64, use_cache=True)
    args.max_new_tokens = original_new_tokens


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [key for key in rows[0].keys() if key != "raw_repeats"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile MiniMind inference engineering knobs.")
    parser.add_argument("--weight_path", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--tokenizer_path", default=str(ROOT / "model"))
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--num_key_value_heads", type=int, default=8)
    parser.add_argument("--intermediate_size", type=int, default=3072)
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--prompt_tokens", type=int, nargs="+", default=[128, 512, 1024])
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--include_no_cache", action="store_true")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = load_model(args, device)
    warmup(model, tokenizer, args, device)

    rows = []
    for batch_size in args.batch_sizes:
        for prompt_tokens in args.prompt_tokens:
            rows.append(profile_case(model, tokenizer, args, device, batch_size, prompt_tokens, True))
            if args.include_no_cache:
                rows.append(profile_case(model, tokenizer, args, device, batch_size, prompt_tokens, False))

    summary = {
        "run_name": args.run_name,
        "weight_path": args.weight_path,
        "settings": {
            "batch_sizes": args.batch_sizes,
            "prompt_tokens": args.prompt_tokens,
            "max_new_tokens": args.max_new_tokens,
            "include_no_cache": args.include_no_cache,
            "dtype": args.dtype,
        },
        "rows": rows,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(Path(args.output_csv), rows)
    for row in rows:
        print(
            f"{row['run_name']}\tbs={row['batch_size']}\tprompt={row['prompt_tokens']}"
            f"\tcache={row['use_cache']}\tprefill={row['prefill_tokens_per_second']:.1f}"
            f"\tdecode={row['decode_tokens_per_second']:.1f}\tmem={row['max_reserved_gb']:.2f}GB",
            flush=True,
        )


if __name__ == "__main__":
    main()
