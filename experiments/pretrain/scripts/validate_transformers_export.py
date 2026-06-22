import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT))

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_native(args, device, dtype):
    config_kwargs = {
        "hidden_size": args.hidden_size,
        "num_hidden_layers": args.num_hidden_layers,
        "num_attention_heads": args.num_attention_heads,
        "num_key_value_heads": args.num_key_value_heads,
        "intermediate_size": args.intermediate_size,
        "max_position_embeddings": args.max_position_embeddings,
    }
    model = MiniMindForCausalLM(MiniMindConfig(**config_kwargs))
    state_dict = torch.load(resolve(args.native_checkpoint), map_location=device)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device=device, dtype=dtype).eval()


def compare_logits(native_model, exported_model, tokenizer, texts, device) -> list[dict]:
    rows = []
    for text in texts:
        encoded = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            native_logits = native_model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded.get("attention_mask"),
                use_cache=False,
            ).logits.float()
            exported_logits = exported_model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded.get("attention_mask"),
                use_cache=False,
            ).logits.float()
        diff = (native_logits - exported_logits).abs()
        rows.append(
            {
                "text": text,
                "tokens": int(encoded["input_ids"].numel()),
                "max_abs_diff": float(diff.max().item()),
                "mean_abs_diff": float(diff.mean().item()),
            }
        )
    return rows


def compare_generation(native_model, exported_model, tokenizer, texts, device, max_new_tokens: int) -> list[dict]:
    rows = []
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id
    for text in texts:
        encoded = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            native_ids = native_model.generate(
                encoded["input_ids"],
                attention_mask=encoded.get("attention_mask"),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            exported_ids = exported_model.generate(
                encoded["input_ids"],
                attention_mask=encoded.get("attention_mask"),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
        prompt_width = encoded["input_ids"].shape[1]
        native_new = native_ids[0, prompt_width:].detach().cpu().tolist()
        exported_new = exported_ids[0, prompt_width:].detach().cpu().tolist()
        rows.append(
            {
                "text": text,
                "exact_match": native_new == exported_new,
                "native_new_ids": native_new,
                "exported_new_ids": exported_new,
                "native_text": tokenizer.decode(native_new, skip_special_tokens=True),
                "exported_text": tokenizer.decode(exported_new, skip_special_tokens=True),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a MiniMind Transformers/Qwen export against native weights.")
    parser.add_argument("--native_checkpoint", required=True)
    parser.add_argument("--export_dir", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--num_key_value_heads", type=int, default=8)
    parser.add_argument("--intermediate_size", type=int, default=3072)
    parser.add_argument("--max_position_embeddings", type=int, default=2048)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    args = parser.parse_args()

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(resolve(args.export_dir))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    native_model = load_native(args, device, dtype)
    exported_model = AutoModelForCausalLM.from_pretrained(resolve(args.export_dir), torch_dtype=dtype).to(device).eval()

    texts = [
        "南京邮电大学位于",
        "Solve 2 + 3 =",
        "def fibonacci(n):",
        "请用一句话说明 MHA 和 GQA 的区别：",
    ]
    logits = compare_logits(native_model, exported_model, tokenizer, texts, device)
    generation = compare_generation(native_model, exported_model, tokenizer, texts, device, args.max_new_tokens)
    result = {
        "native_checkpoint": str(resolve(args.native_checkpoint)),
        "export_dir": str(resolve(args.export_dir)),
        "dtype": args.dtype,
        "logits": logits,
        "generation": generation,
        "max_abs_diff": max(row["max_abs_diff"] for row in logits),
        "mean_abs_diff_max": max(row["mean_abs_diff"] for row in logits),
        "generation_all_exact": all(row["exact_match"] for row in generation),
    }
    output_path = resolve(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["max_abs_diff", "mean_abs_diff_max", "generation_all_exact"]}, ensure_ascii=False))
    print(output_path)


if __name__ == "__main__":
    main()
