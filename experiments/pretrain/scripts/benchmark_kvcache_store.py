import argparse
import json
import time
from pathlib import Path

import torch
import triton
import triton.language as tl


ROOT = Path(__file__).resolve().parents[3]


@triton.jit
def store_kvcache_masked_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < D
    key = tl.load(key_ptr + idx * key_stride + offsets, mask=mask, other=0.0)
    value = tl.load(value_ptr + idx * value_stride + offsets, mask=mask, other=0.0)
    cache_offsets = slot * D + offsets
    tl.store(k_cache_ptr + cache_offsets, key, mask=mask & (slot != -1))
    tl.store(v_cache_ptr + cache_offsets, value, mask=mask & (slot != -1))


def store_kvcache_masked(key, value, k_cache, v_cache, slot_mapping):
    n, num_heads, head_dim = key.shape
    d = num_heads * head_dim
    block_d = triton.next_power_of_2(d)
    store_kvcache_masked_kernel[(n,)](
        key,
        key.stride(0),
        value,
        value.stride(0),
        k_cache,
        v_cache,
        slot_mapping,
        D=d,
        BLOCK_D=block_d,
    )


def store_kvcache_python(key, value, k_cache, v_cache, slot_mapping):
    flat_k = k_cache.view(-1, key.shape[1], key.shape[2])
    flat_v = v_cache.view(-1, value.shape[1], value.shape[2])
    for index in range(slot_mapping.numel()):
        slot = int(slot_mapping[index])
        if slot == -1:
            continue
        flat_k[slot].copy_(key[index])
        flat_v[slot].copy_(value[index])


def time_fn(fn, repeat: int):
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeat


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a masked Triton KV-cache store kernel for MiniMind/nano-vLLM.")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=96)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="bfloat16")
    args = parser.parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    device = torch.device("cuda")
    key = torch.randn(args.tokens, args.num_heads, args.head_dim, device=device, dtype=dtype)
    value = torch.randn_like(key)
    slot_mapping = torch.arange(args.tokens, device=device, dtype=torch.int32)
    k_cache_triton = torch.empty(args.tokens, args.num_heads, args.head_dim, device=device, dtype=dtype)
    v_cache_triton = torch.empty_like(k_cache_triton)
    k_cache_python = torch.empty_like(k_cache_triton)
    v_cache_python = torch.empty_like(v_cache_triton)

    store_kvcache_masked(key, value, k_cache_triton, v_cache_triton, slot_mapping)
    store_kvcache_python(key, value, k_cache_python, v_cache_python, slot_mapping)
    torch.cuda.synchronize()
    max_key_diff = (k_cache_triton - k_cache_python).abs().max().item()
    max_value_diff = (v_cache_triton - v_cache_python).abs().max().item()

    triton_seconds = time_fn(lambda: store_kvcache_masked(key, value, k_cache_triton, v_cache_triton, slot_mapping), args.repeat)
    python_seconds = time_fn(lambda: store_kvcache_python(key, value, k_cache_python, v_cache_python, slot_mapping), max(1, args.repeat // 10))

    result = {
        "tokens": args.tokens,
        "num_heads": args.num_heads,
        "head_dim": args.head_dim,
        "D": args.num_heads * args.head_dim,
        "BLOCK_D": triton.next_power_of_2(args.num_heads * args.head_dim),
        "dtype": args.dtype,
        "max_key_diff": max_key_diff,
        "max_value_diff": max_value_diff,
        "triton_seconds_per_call": triton_seconds,
        "python_seconds_per_call": python_seconds,
        "speedup_vs_python": python_seconds / triton_seconds if triton_seconds else None,
    }
    output_path = Path(args.output_json)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
