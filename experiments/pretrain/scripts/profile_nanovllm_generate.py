import argparse
import json
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoTokenizer

from nanovllm import LLM, SamplingParams


ROOT = Path(__file__).resolve().parents[3]


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def make_prompt(target_chars: int) -> str:
    prefix = "MiniMind nano-vLLM profiler. Answer briefly. Context: "
    filler = "profile context fragment. "
    if target_chars <= len(prefix):
        return prefix[:target_chars]
    repeats = ((target_chars - len(prefix)) // len(filler)) + 1
    return (prefix + filler * repeats)[:target_chars]


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile nano-vLLM generate with torch.profiler.")
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prompt_chars", type=int, default=512)
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--max_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--ignore_eos", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=2048)
    parser.add_argument("--max_num_seqs", type=int, default=16)
    parser.add_argument("--max_num_batched_tokens", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.5)
    parser.add_argument("--with_stack", action="store_true")
    parser.add_argument("--record_shapes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--profile_memory", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for nano-vLLM profiling")

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = resolve(args.model_dir)

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
    sampling = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens, ignore_eos=args.ignore_eos)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": make_prompt(args.prompt_chars)}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prompts = [prompt] * args.requests

    for _ in range(args.warmup):
        with record_function("nanovllm_warmup_generate"):
            llm.generate(prompts, sampling, use_tqdm=False)
    torch.cuda.synchronize()

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    start = time.perf_counter()
    with profile(
        activities=activities,
        record_shapes=args.record_shapes,
        profile_memory=args.profile_memory,
        with_stack=args.with_stack,
    ) as prof:
        with record_function("nanovllm_profile_generate"):
            outputs = llm.generate(prompts, sampling, use_tqdm=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    trace_path = output_dir / "torch_profiler_trace.json"
    prof.export_chrome_trace(str(trace_path))
    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=40)
    (output_dir / "key_averages_cuda.txt").write_text(table + "\n", encoding="utf-8")
    cpu_table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=40)
    (output_dir / "key_averages_cpu.txt").write_text(cpu_table + "\n", encoding="utf-8")

    output_tokens = sum(len(output["token_ids"]) for output in outputs)
    summary = {
        "model_dir": str(model_dir),
        "prompt_chars": args.prompt_chars,
        "requests": args.requests,
        "max_tokens": args.max_tokens,
        "output_tokens": output_tokens,
        "elapsed_seconds": elapsed,
        "output_tokens_per_second": output_tokens / elapsed if elapsed else None,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "trace": str(trace_path),
        "key_averages_cuda": str(output_dir / "key_averages_cuda.txt"),
        "key_averages_cpu": str(output_dir / "key_averages_cpu.txt"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
