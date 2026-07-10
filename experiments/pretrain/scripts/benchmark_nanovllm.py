import argparse
import json
import time
from pathlib import Path

from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams


ROOT = Path(__file__).resolve().parents[3]


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def make_prompt(target_chars: int) -> str:
    prefix = "MiniMind nano-vLLM baseline. Answer in one short sentence. Context: "
    filler = "benchmark context fragment. "
    if target_chars <= len(prefix):
        return prefix[:target_chars]
    repeats = ((target_chars - len(prefix)) // len(filler)) + 1
    return (prefix + filler * repeats)[:target_chars]


def patch_nanovllm_store_kvcache_python() -> None:
    import nanovllm.layers.attention as attention

    def store_kvcache_python(key, value, k_cache, v_cache, slot_mapping):
        flat_k_cache = k_cache.view(-1, key.shape[1], key.shape[2])
        flat_v_cache = v_cache.view(-1, value.shape[1], value.shape[2])
        for index in range(slot_mapping.numel()):
            slot = int(slot_mapping[index])
            if slot == -1:
                continue
            flat_k_cache[slot].copy_(key[index])
            flat_v_cache[slot].copy_(value[index])

    attention.store_kvcache = store_kvcache_python


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small nano-vLLM baseline on a Qwen-compatible MiniMind export.")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--prompt_chars", type=int, nargs="+", default=[128, 512, 1024])
    parser.add_argument("--requests_per_case", type=int, default=4)
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--ignore_eos", action="store_true")
    parser.add_argument("--max_model_len", type=int, default=2048)
    parser.add_argument("--max_num_seqs", type=int, default=16)
    parser.add_argument("--max_num_batched_tokens", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.5)
    parser.add_argument("--warmup", action="store_true", help="Run one short generation before timing.")
    parser.add_argument("--python_store_fallback", action="store_true", help="Use a slow Python KV-cache store fallback instead of nano-vLLM's Triton kernel.")
    args = parser.parse_args()

    model_dir = resolve(args.model_dir)
    if args.python_store_fallback:
        patch_nanovllm_store_kvcache_python()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    llm = LLM(
        str(model_dir),
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens, ignore_eos=args.ignore_eos)
    if args.warmup:
        warm_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": make_prompt(128)}],
            tokenize=False,
            add_generation_prompt=True,
        )
        llm.generate([warm_prompt], SamplingParams(temperature=args.temperature, max_tokens=min(args.max_tokens, 16), ignore_eos=True), use_tqdm=False)

    rows = []
    for prompt_chars in args.prompt_chars:
        user_prompt = make_prompt(prompt_chars)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts = [prompt] * args.requests_per_case
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        elapsed = time.perf_counter() - start
        output_tokens = sum(len(output["token_ids"]) for output in outputs)
        rows.append(
            {
                "prompt_chars": prompt_chars,
                "requests": len(prompts),
                "max_tokens": args.max_tokens,
                "elapsed_seconds": elapsed,
                "output_tokens": output_tokens,
                "output_tokens_per_second": output_tokens / elapsed if elapsed else None,
                "warmup": args.warmup,
                "python_store_fallback": args.python_store_fallback,
                "sample_text": outputs[0]["text"] if outputs else "",
            }
        )

    output_path = resolve(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
