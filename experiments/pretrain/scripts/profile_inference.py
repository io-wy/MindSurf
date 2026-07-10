import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: E402


DEFAULT_PROMPTS = [
    "请用三句话介绍你自己，并说明你能帮我做什么。",
    "请用高中生能听懂的话解释什么是机器学习。",
    "Explain in simple English why the sky looks blue.",
    "请写一个 Python 函数，输入 n，返回斐波那契数列的第 n 项。",
]


def load_prompts(path: Path | None) -> list[str]:
    if path is None:
        return DEFAULT_PROMPTS
    prompts = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            prompts.append(str(item.get("prompt", item.get("text", ""))))
    return [prompt for prompt in prompts if prompt.strip()]


def synchronize(device: str) -> None:
    if "cuda" in device and torch.cuda.is_available():
        torch.cuda.synchronize()


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


def decode_step(model, input_ids, attention_mask, past_key_values, dtype_ctx, use_cache: bool):
    past_len = past_key_values[0][0].shape[1] if past_key_values else 0
    with dtype_ctx:
        outputs = model(
            input_ids[:, past_len:],
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        logits = outputs.logits[:, -1, :]
    next_token = torch.argmax(logits, dim=-1, keepdim=True)
    next_input_ids = torch.cat([input_ids, next_token], dim=-1)
    next_attention = (
        torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], dim=-1)
        if attention_mask is not None
        else None
    )
    return next_input_ids, next_attention, outputs.past_key_values


@torch.inference_mode()
def profile_prompt(model, tokenizer, prompt: str, args, dtype_ctx, device: str) -> dict:
    if args.prompt_format == "pretrain":
        text = (tokenizer.bos_token or "") + prompt
    elif args.prompt_format == "chat":
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = prompt
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_input_tokens).to(device)
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")

    if "cuda" in device and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    synchronize(device)
    started = time.perf_counter()
    with dtype_ctx:
        prefill = model(input_ids, attention_mask=attention_mask, use_cache=True)
    synchronize(device)
    prefill_seconds = time.perf_counter() - started
    past_key_values = prefill.past_key_values
    kv_bytes_after_prefill = kv_cache_bytes(past_key_values)

    logits = prefill.logits[:, -1, :]
    next_token = torch.argmax(logits, dim=-1, keepdim=True)
    input_ids = torch.cat([input_ids, next_token], dim=-1)
    attention_mask = (
        torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], dim=-1)
        if attention_mask is not None
        else None
    )
    generated = 1 if args.max_new_tokens > 0 else 0
    decode_seconds = 0.0
    for _ in range(max(args.max_new_tokens - generated, 0)):
        synchronize(device)
        before = time.perf_counter()
        input_ids, attention_mask, past_key_values = decode_step(
            model,
            input_ids,
            attention_mask,
            past_key_values,
            dtype_ctx,
            use_cache=True,
        )
        synchronize(device)
        decode_seconds += time.perf_counter() - before
        generated += 1
        if int(input_ids[0, -1].item()) == tokenizer.eos_token_id:
            break

    max_reserved = torch.cuda.max_memory_reserved() if "cuda" in device and torch.cuda.is_available() else 0
    max_allocated = torch.cuda.max_memory_allocated() if "cuda" in device and torch.cuda.is_available() else 0
    return {
        "prompt": prompt,
        "input_tokens": int(encoded["input_ids"].shape[1]),
        "generated_tokens": generated,
        "prefill_seconds": prefill_seconds,
        "prefill_tokens_per_second": int(encoded["input_ids"].shape[1]) / max(prefill_seconds, 1e-9),
        "decode_seconds": decode_seconds,
        "decode_tokens_per_second": generated / max(decode_seconds, 1e-9),
        "kv_cache_mb_after_prefill": kv_bytes_after_prefill / 1024 / 1024,
        "kv_cache_mb_final": kv_cache_bytes(past_key_values) / 1024 / 1024,
        "max_memory_reserved_gb": max_reserved / 1024 / 1024 / 1024,
        "max_memory_allocated_gb": max_allocated / 1024 / 1024 / 1024,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile MiniMind native inference prefill/decode and KV cache.")
    parser.add_argument("--weight_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer_path", default=str(ROOT / "model"))
    parser.add_argument("--prompts_path", type=Path, default=None)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--num_key_value_heads", type=int, default=8)
    parser.add_argument("--intermediate_size", type=int, default=3072)
    parser.add_argument("--max_input_tokens", type=int, default=384)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--prompt_format", choices=["pretrain", "chat", "raw"], default="pretrain")
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    dtype_ctx = torch.amp.autocast("cuda", dtype=dtype) if "cuda" in device and dtype != torch.float32 else nullcontext()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=max(args.max_input_tokens + args.max_new_tokens + 8, 2048),
    )
    model = MiniMindForCausalLM(config)
    model.load_state_dict(torch.load(args.weight_path, map_location="cpu"), strict=True)
    model = model.to(device).eval()
    if dtype != torch.float32:
        model = model.to(dtype)

    prompts = load_prompts(args.prompts_path)
    rows = [profile_prompt(model, tokenizer, prompt, args, dtype_ctx, device) for prompt in prompts]
    summary = {
        "weight_path": args.weight_path,
        "prompt_count": len(rows),
        "settings": {
            "hidden_size": args.hidden_size,
            "num_hidden_layers": args.num_hidden_layers,
            "num_attention_heads": args.num_attention_heads,
            "num_key_value_heads": args.num_key_value_heads,
            "intermediate_size": args.intermediate_size,
            "max_input_tokens": args.max_input_tokens,
            "max_new_tokens": args.max_new_tokens,
            "prompt_format": args.prompt_format,
            "dtype": args.dtype,
            "device": device,
        },
        "mean_prefill_tps": sum(row["prefill_tokens_per_second"] for row in rows) / max(len(rows), 1),
        "mean_decode_tps": sum(row["decode_tokens_per_second"] for row in rows) / max(len(rows), 1),
        "max_memory_reserved_gb": max((row["max_memory_reserved_gb"] for row in rows), default=0),
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ["prompt_count", "mean_prefill_tps", "mean_decode_tps", "max_memory_reserved_gb"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
