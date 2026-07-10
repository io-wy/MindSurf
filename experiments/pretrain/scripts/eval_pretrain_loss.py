import argparse
import json
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dataset.lm_dataset import PretrainDataset
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Evaluate MiniMind pretrain CE/PPL on a jsonl validation set")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--weight_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer_path", default=str(ROOT / "model"))
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--num_key_value_heads", type=int, default=4)
    parser.add_argument("--intermediate_size", type=int, default=None)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_batches", type=int, default=100)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if "cuda" in device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = torch.amp.autocast("cuda", dtype=dtype) if device_type == "cuda" else nullcontext()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    dataset = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=(device_type == "cuda"))
    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=max(args.max_seq_len + 8, 2048),
    )
    model = MiniMindForCausalLM(config)
    model.load_state_dict(torch.load(args.weight_path, map_location="cpu"), strict=True)
    model = model.to(device).eval()

    losses = []
    tokens = 0
    started = time.time()
    for batch_index, (input_ids, labels) in enumerate(loader, start=1):
        if args.max_batches > 0 and batch_index > args.max_batches:
            break
        input_ids = input_ids.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast_ctx:
            result = model(input_ids, labels=labels)
            loss = result.loss + result.aux_loss
        losses.append(float(loss.detach().cpu()))
        tokens += int((labels != -100).sum().item())
        print(f"eval_batch={batch_index} loss={losses[-1]:.4f}", flush=True)

    mean_loss = sum(losses) / max(len(losses), 1)
    summary = {
        "data_path": args.data_path,
        "weight_path": args.weight_path,
        "batches": len(losses),
        "tokens": tokens,
        "mean_loss": mean_loss,
        "ppl": math.exp(mean_loss) if mean_loss < 20 else None,
        "hidden_size": args.hidden_size,
        "num_hidden_layers": args.num_hidden_layers,
        "num_attention_heads": args.num_attention_heads,
        "num_key_value_heads": args.num_key_value_heads,
        "intermediate_size": args.intermediate_size,
        "max_seq_len": args.max_seq_len,
        "seconds": round(time.time() - started, 3),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
