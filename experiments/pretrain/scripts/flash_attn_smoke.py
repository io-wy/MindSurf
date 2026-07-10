import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func


ROOT = Path(__file__).resolve().parents[3]


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small FlashAttention correctness smoke test for MiniMind dimensions.")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=96)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--max_abs_tolerance", type=float, default=5e-2)
    parser.add_argument("--output_json", default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for flash_attn smoke test")

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    device = torch.device("cuda")
    torch.manual_seed(0)
    q = torch.randn(args.batch, args.seq_len, args.num_heads, args.head_dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    out_flash = flash_attn_func(q, k, v, dropout_p=0.0, causal=True)
    out_ref = F.scaled_dot_product_attention(
        q.permute(0, 2, 1, 3),
        k.permute(0, 2, 1, 3),
        v.permute(0, 2, 1, 3),
        is_causal=True,
    ).permute(0, 2, 1, 3)
    torch.cuda.synchronize()
    max_abs = (out_flash - out_ref).abs().max().item()
    mean_abs = (out_flash - out_ref).abs().mean().item()
    passed = out_flash.shape == q.shape and max_abs <= args.max_abs_tolerance
    result = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "batch": args.batch,
        "seq_len": args.seq_len,
        "num_heads": args.num_heads,
        "head_dim": args.head_dim,
        "dtype": args.dtype,
        "shape": list(out_flash.shape),
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "max_abs_tolerance": args.max_abs_tolerance,
        "passed": passed,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        output = resolve(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
