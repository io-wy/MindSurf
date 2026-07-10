import argparse
import csv
import json
import math
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if "prompt" not in item or "choices" not in item or "answer" not in item:
                raise ValueError(f"line {line_number} must include prompt, choices, answer")
            rows.append(item)
    return rows


def answer_index(item: dict) -> int:
    answer = item["answer"]
    if isinstance(answer, int):
        return answer
    if isinstance(answer, str):
        labels = item.get("labels") or []
        if labels and answer in labels:
            return labels.index(answer)
        if answer.isdigit():
            return int(answer)
    raise ValueError(f"cannot resolve answer index for item {item.get('id', '')}: {answer!r}")


def build_ids(tokenizer, prompt: str, choice: str, max_seq_len: int) -> tuple[torch.Tensor, torch.Tensor, int]:
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    prompt_ids = tokenizer(str(prompt), add_special_tokens=False).input_ids
    choice_ids = tokenizer(str(choice), add_special_tokens=False).input_ids
    if not choice_ids:
        choice_ids = [tokenizer.eos_token_id]
    ids = bos + prompt_ids + choice_ids
    labels = [-100] * (len(bos) + len(prompt_ids)) + choice_ids
    if len(ids) > max_seq_len:
        overflow = len(ids) - max_seq_len
        prompt_start = len(bos)
        removable = min(overflow, max(len(prompt_ids) - 1, 0))
        del ids[prompt_start : prompt_start + removable]
        del labels[prompt_start : prompt_start + removable]
    if len(ids) > max_seq_len:
        ids = ids[-max_seq_len:]
        labels = labels[-max_seq_len:]
    target_tokens = sum(1 for value in labels if value != -100)
    return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long), target_tokens


@torch.no_grad()
def score_choice(model, tokenizer, prompt: str, choice: str, args, autocast_ctx, device: str) -> dict:
    input_ids, labels, target_tokens = build_ids(tokenizer, prompt, choice, args.max_seq_len)
    input_ids = input_ids.unsqueeze(0).to(device)
    labels = labels.unsqueeze(0).to(device)
    with autocast_ctx:
        logits = model(input_ids).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        losses = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view_as(shift_labels)
    mask = shift_labels != -100
    nll_sum = float(losses[mask].sum().detach().cpu()) if mask.any() else float("inf")
    token_count = int(mask.sum().detach().cpu())
    nll_mean = nll_sum / max(token_count, 1)
    score = nll_mean if args.length_norm == "mean" else nll_sum
    return {
        "score": score,
        "nll_sum": nll_sum,
        "nll_mean": nll_mean,
        "tokens": token_count,
        "target_tokens_before_truncation": target_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MCQ accuracy by continuation log-likelihood.")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--weight_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tokenizer_path", default=str(ROOT / "model"))
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--num_key_value_heads", type=int, default=8)
    parser.add_argument("--intermediate_size", type=int, default=3072)
    parser.add_argument("--max_seq_len", type=int, default=384)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--length_norm", choices=["mean", "sum"], default="mean")
    args = parser.parse_args()

    data_path = Path(args.data_path)
    output_path = Path(args.output)
    device = args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if "cuda" in device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = torch.amp.autocast("cuda", dtype=dtype) if device_type == "cuda" else nullcontext()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
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

    started = time.time()
    items = read_jsonl(data_path)
    rows = []
    correct = 0
    by_category: dict[str, dict[str, int]] = {}
    for item in items:
        prompt = item["prompt"]
        choices = item["choices"]
        gold = answer_index(item)
        scored = [score_choice(model, tokenizer, prompt, choice, args, autocast_ctx, device) for choice in choices]
        ranked = sorted(range(len(scored)), key=lambda index: scored[index]["score"])
        pred = ranked[0]
        is_correct = int(pred == gold)
        correct += is_correct
        category = item.get("category", "uncategorized")
        by_category.setdefault(category, {"correct": 0, "total": 0})
        by_category[category]["correct"] += is_correct
        by_category[category]["total"] += 1
        second_score = scored[ranked[1]]["score"] if len(ranked) > 1 else math.nan
        rows.append(
            {
                "id": item.get("id", ""),
                "category": category,
                "gold": gold,
                "pred": pred,
                "correct": is_correct,
                "gold_score": scored[gold]["score"],
                "pred_score": scored[pred]["score"],
                "margin": second_score - scored[pred]["score"] if len(ranked) > 1 else math.nan,
                "choices": choices,
                "choice_scores": scored,
            }
        )
        print(f"{item.get('id', '')}\tpred={pred}\tgold={gold}\tcorrect={is_correct}", flush=True)

    total = len(rows)
    category_summary = {
        category: {
            "correct": values["correct"],
            "total": values["total"],
            "accuracy": values["correct"] / max(values["total"], 1),
        }
        for category, values in sorted(by_category.items())
    }
    summary = {
        "data_path": str(data_path),
        "weight_path": args.weight_path,
        "total": total,
        "correct": correct,
        "accuracy": correct / max(total, 1),
        "length_norm": args.length_norm,
        "max_seq_len": args.max_seq_len,
        "category_summary": category_summary,
        "seconds": round(time.time() - started, 3),
        "rows": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tsv_path = output_path.with_suffix(".tsv")
    with tsv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "category", "gold", "pred", "correct", "gold_score", "pred_score", "margin"],
            delimiter="\t",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})
    print(json.dumps({key: summary[key] for key in ["total", "correct", "accuracy", "seconds"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
