import argparse
import json
import math
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dataset.lm_dataset import SFTDataset
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


def scheduled_lr(step, total_steps, base_lr, warmup_steps, min_lr_ratio):
    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    min_lr = base_lr * min_lr_ratio
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def make_loader(dataset, batch_size, seed, num_workers):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Platform SFT probe runner with arbitrary MiniMind checkpoint init.")
    parser.add_argument("--data_path", default=str(ROOT / "dataset" / "sft_t2t_mini.jsonl"))
    parser.add_argument("--tokenizer_path", default=str(ROOT / "model"))
    parser.add_argument("--init_weight", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--save_weight", required=True)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--num_key_value_heads", type=int, default=4)
    parser.add_argument("--intermediate_size", type=int, default=None)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=50)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
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
        "max_position_embeddings": max(args.max_seq_len + 8, 2048),
    }
    if args.intermediate_size is not None:
        config_kwargs["intermediate_size"] = args.intermediate_size
    config = MiniMindConfig(**config_kwargs)
    model = MiniMindForCausalLM(config)
    state = torch.load(args.init_weight, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"loaded init_weight={args.init_weight} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    model = model.to(device).train()
    dataset = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    loader = make_loader(dataset, args.batch_size, args.seed, args.num_workers)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == "float16"))

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    losses = []
    started = time.time()
    step = 0
    epoch = 0
    while step < args.max_steps:
        epoch += 1
        for input_ids, labels in loader:
            step += 1
            lr = scheduled_lr(step, args.max_steps, args.learning_rate, args.warmup_steps, args.min_lr_ratio)
            for group in optimizer.param_groups:
                group["lr"] = lr
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast_ctx:
                result = model(input_ids, labels=labels)
                loss = result.loss + result.aux_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            losses.append(float(loss.detach().cpu()))
            if step % args.log_interval == 0 or step == 1 or step == args.max_steps:
                elapsed = max(time.time() - started, 1e-6)
                tokens_per_second = step * args.batch_size * args.max_seq_len / elapsed
                print(
                    f"epoch={epoch} step={step}/{args.max_steps} loss={losses[-1]:.4f} "
                    f"lr={lr:.8f} tokens/s={tokens_per_second:.0f}",
                    flush=True,
                )
            if step >= args.max_steps:
                break

    model.eval()
    weight_path = save_dir / f"{args.save_weight}_{args.hidden_size}.pth"
    torch.save({k: v.half().cpu() for k, v in model.state_dict().items()}, weight_path)
    summary = {
        "data_path": args.data_path,
        "init_weight": args.init_weight,
        "weight_path": str(weight_path),
        "hidden_size": args.hidden_size,
        "num_hidden_layers": args.num_hidden_layers,
        "num_attention_heads": args.num_attention_heads,
        "num_key_value_heads": args.num_key_value_heads,
        "intermediate_size": args.intermediate_size,
        "max_seq_len": args.max_seq_len,
        "batch_size": args.batch_size,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "warmup_steps": args.warmup_steps,
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "mean_last_50_loss": sum(losses[-50:]) / min(len(losses), 50) if losses else None,
        "seconds": round(time.time() - started, 3),
        "max_memory_reserved_gb": round(torch.cuda.max_memory_reserved() / (1024 ** 3), 3) if torch.cuda.is_available() else None,
    }
    (save_dir / f"{args.save_weight}_{args.hidden_size}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
