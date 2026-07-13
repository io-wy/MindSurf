import argparse
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from dataset.lm_dataset import PretrainDataset
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from experiments.pretrain.training_checkpoint import (
    load_training_checkpoint,
    save_training_checkpoint,
    validate_resume_config,
)
from experiments.pretrain.experiment_tracking import (
    ExperimentTracker,
    detect_source_revision,
    write_run_manifest,
)
from trainer.trainer_utils import get_model_params, setup_seed


class PackedPretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.blocks = []
        stream = []
        raw_tokens = 0
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                text = json.loads(line)["text"]
                ids = tokenizer(str(text), add_special_tokens=False).input_ids + [tokenizer.eos_token_id]
                raw_tokens += len(ids)
                stream.extend(ids)
                while len(stream) >= max_length:
                    self.blocks.append(torch.tensor(stream[:max_length], dtype=torch.long))
                    stream = stream[max_length:]
        self.raw_tokens = raw_tokens
        self.used_tokens = len(self.blocks) * max_length

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, index):
        input_ids = self.blocks[index]
        return input_ids, input_ids.clone()


class StreamingPackedPretrainDataset(IterableDataset):
    def __init__(self, data_path, tokenizer, max_length=512, shuffle_buffer=0, seed=42, skip_blocks=0):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.skip_blocks = skip_blocks

    def __iter__(self):
        token_buffer = []
        block_buffer = []
        emitted_blocks = 0
        rng = random.Random(self.seed)

        def emit(block):
            if self.shuffle_buffer <= 1:
                return [(block, block.clone())]
            block_buffer.append(block)
            if len(block_buffer) < self.shuffle_buffer:
                return []
            index = rng.randrange(len(block_buffer))
            selected = block_buffer.pop(index)
            return [(selected, selected.clone())]

        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                text = json.loads(line)["text"]
                token_buffer.extend(self.tokenizer(str(text), add_special_tokens=False).input_ids)
                token_buffer.append(self.tokenizer.eos_token_id)
                while len(token_buffer) >= self.max_length:
                    block = torch.tensor(token_buffer[:self.max_length], dtype=torch.long)
                    token_buffer = token_buffer[self.max_length:]
                    for item in emit(block):
                        emitted_blocks += 1
                        if emitted_blocks <= self.skip_blocks:
                            continue
                        yield item
        if self.shuffle_buffer > 1:
            rng.shuffle(block_buffer)
            for block in block_buffer:
                emitted_blocks += 1
                if emitted_blocks <= self.skip_blocks:
                    continue
                yield block, block.clone()


class StreamingMixedPackedPretrainDataset(IterableDataset):
    def __init__(self, mix_path, tokenizer, max_length=512, shuffle_buffer=0, seed=42, skip_blocks=0):
        self.mix_path = Path(mix_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.skip_blocks = skip_blocks
        self.sources = self._load_sources()

    def _resolve_path(self, raw_path):
        path = Path(raw_path)
        if path.is_absolute():
            return path
        local_path = (self.mix_path.parent / path).resolve()
        if local_path.exists():
            return local_path
        return (ROOT / path).resolve()

    def _load_sources(self):
        payload = json.loads(self.mix_path.read_text(encoding="utf-8"))
        raw_sources = payload["sources"] if isinstance(payload, dict) else payload
        sources = []
        for item in raw_sources:
            path = self._resolve_path(item["path"])
            weight = float(item.get("weight", 1.0))
            if weight <= 0:
                continue
            if not path.exists():
                raise FileNotFoundError(f"data mix source not found: {path}")
            sources.append({"path": path, "weight": weight, "name": item.get("name", path.stem)})
        if not sources:
            raise ValueError(f"data mix has no positive-weight sources: {self.mix_path}")
        return sources

    @staticmethod
    def _line_iterator(path):
        while True:
            yielded = False
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    yielded = True
                    yield line
            if not yielded:
                raise RuntimeError(f"data mix source has no non-empty lines: {path}")

    @staticmethod
    def _choose_source(rng, sources, total_weight):
        target = rng.random() * total_weight
        cumulative = 0.0
        for index, source in enumerate(sources):
            cumulative += source["weight"]
            if target <= cumulative:
                return index
        return len(sources) - 1

    def __iter__(self):
        rng = random.Random(self.seed)
        sources = [
            {**source, "iterator": self._line_iterator(source["path"])}
            for source in self.sources
        ]
        total_weight = sum(source["weight"] for source in sources)
        token_buffer = []
        block_buffer = []
        emitted_blocks = 0

        def emit(block):
            if self.shuffle_buffer <= 1:
                return [(block, block.clone())]
            block_buffer.append(block)
            if len(block_buffer) < self.shuffle_buffer:
                return []
            index = rng.randrange(len(block_buffer))
            selected = block_buffer.pop(index)
            return [(selected, selected.clone())]

        while True:
            source_index = self._choose_source(rng, sources, total_weight)
            line = next(sources[source_index]["iterator"])
            text = json.loads(line)["text"]
            token_buffer.extend(self.tokenizer(str(text), add_special_tokens=False).input_ids)
            token_buffer.append(self.tokenizer.eos_token_id)
            while len(token_buffer) >= self.max_length:
                block = torch.tensor(token_buffer[: self.max_length], dtype=torch.long)
                token_buffer = token_buffer[self.max_length :]
                for item in emit(block):
                    emitted_blocks += 1
                    if emitted_blocks <= self.skip_blocks:
                        continue
                    yield item


def make_optimizer(params, lr, use_fused, weight_decay, betas, eps):
    if use_fused and torch.cuda.is_available():
        try:
            return optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, fused=True), True
        except TypeError:
            pass
    return optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay), False


def scheduled_lr(
    step,
    total_steps,
    base_lr,
    warmup_steps,
    schedule="cosine",
    min_lr_ratio=0.1,
    stable_ratio=0.8,
):
    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * step / warmup_steps

    adjusted_step = step - warmup_steps if warmup_steps > 0 else step
    adjusted_total = max(total_steps - warmup_steps, 1)
    progress = min(max(adjusted_step / adjusted_total, 0.0), 1.0)
    min_lr_ratio = min(max(min_lr_ratio, 0.0), 1.0)

    if schedule == "constant":
        return base_lr
    if schedule == "linear":
        return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - progress))
    if schedule == "wsd":
        stable_ratio = min(max(stable_ratio, 0.0), 0.99)
        if progress <= stable_ratio:
            return base_lr
        decay_progress = (progress - stable_ratio) / max(1.0 - stable_ratio, 1e-8)
        return base_lr * (
            min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * decay_progress))
        )

    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))


def train(args):
    if args.init_weight and args.resume_checkpoint:
        raise ValueError("--init_weight and --resume_checkpoint are mutually exclusive")
    if args.resume_checkpoint and not args.stream_packed:
        raise ValueError("--resume_checkpoint currently requires --stream_packed for a reproducible data cursor")
    setup_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    device = args.device if args.device else ("cuda:0" if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if "cuda" in device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = torch.amp.autocast("cuda", dtype=dtype) if device_type == "cuda" else nullcontext()

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    config = MiniMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        intermediate_size=args.intermediate_size,
        use_moe=bool(args.use_moe),
        max_position_embeddings=max(args.max_seq_len + 8, 2048),
    )
    model = MiniMindForCausalLM(config).to(device)
    if args.init_weight:
        init_path = Path(args.init_weight)
        if not init_path.exists():
            raise FileNotFoundError(f"--init_weight not found: {init_path}")
        state = torch.load(init_path, map_location=device)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(
            f"loaded init_weight={init_path} missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
    get_model_params(model, config)

    optimizer, fused_used = make_optimizer(
        model.parameters(),
        args.learning_rate,
        args.fused_adamw,
        args.weight_decay,
        (args.adam_beta1, args.adam_beta2),
        args.adam_eps,
    )
    scaler = torch.amp.GradScaler(device_type, enabled=(args.dtype == "float16" and device_type == "cuda"))
    run_config = {
        key: value
        for key, value in vars(args).items()
        if key not in {
            "resume_checkpoint",
            "checkpoint_interval",
            "stop_after_steps",
            "log_interval",
            "save_dir",
            "trackio_project",
            "trackio_space_id",
        }
    }
    checkpoint_path = out_dir / f"{args.save_weight}_training_state.pt"
    weight_path = out_dir / f"{args.save_weight}_{args.hidden_size}.pth"
    summary_path = out_dir / f"{args.save_weight}_{args.hidden_size}_summary.json"
    metrics_path = out_dir / f"{args.save_weight}_metrics.jsonl"
    run_manifest_path = out_dir / f"{args.save_weight}_run_manifest.json"
    source_revision = detect_source_revision(ROOT)
    write_run_manifest(
        run_manifest_path,
        run_name=args.save_weight,
        config=run_config,
        source_revision=source_revision,
        artifacts={
            "metrics": metrics_path.name,
            "training_checkpoint": checkpoint_path.name,
            "weights": weight_path.name,
            "summary": summary_path.name,
        },
    )
    tracker = ExperimentTracker(
        metrics_path,
        run_name=args.save_weight,
        config=run_config,
        trackio_project=args.trackio_project,
        trackio_space_id=args.trackio_space_id,
    )
    global_step = 0
    optimizer_step = 0
    consumed_blocks = 0
    start_epoch = 0
    if args.resume_checkpoint:
        resume_path = Path(args.resume_checkpoint)
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume_checkpoint not found: {resume_path}")
        restored = load_training_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            map_location=device,
        )
        validate_resume_config(restored["run_config"], run_config)
        progress = restored["progress"]
        global_step = int(progress["global_step"])
        optimizer_step = int(progress["optimizer_step"])
        consumed_blocks = int(progress["consumed_blocks"])
        start_epoch = int(progress["epoch"])
        args.skip_blocks = consumed_blocks
        print(
            f"resumed checkpoint={resume_path} global_step={global_step} "
            f"optimizer_step={optimizer_step} consumed_blocks={consumed_blocks}",
            flush=True,
        )
        tracker.alert(
            "training resumed",
            f"restored global_step={global_step} optimizer_step={optimizer_step} consumed_blocks={consumed_blocks}",
            level="info",
            step=global_step,
        )

    if args.data_mix_json:
        if not args.stream_packed:
            raise ValueError("--data_mix_json requires --stream_packed and --max_steps > 0.")
        if args.max_steps <= 0:
            raise ValueError("--data_mix_json requires --max_steps > 0 because the mixed stream can cycle indefinitely.")
        dataset = StreamingMixedPackedPretrainDataset(
            args.data_mix_json,
            tokenizer,
            max_length=args.max_seq_len,
            shuffle_buffer=args.shuffle_buffer,
            seed=args.seed,
            skip_blocks=args.skip_blocks,
        )
        token_utilization = 1.0
        dataset_len = None
    elif args.stream_packed:
        if args.max_steps <= 0:
            raise ValueError("--stream_packed requires --max_steps > 0 because IterableDataset length is unknown.")
        if not args.data_path:
            raise ValueError("--data_path is required unless --data_mix_json is used.")
        dataset = StreamingPackedPretrainDataset(
            args.data_path,
            tokenizer,
            max_length=args.max_seq_len,
            shuffle_buffer=args.shuffle_buffer,
            seed=args.seed,
            skip_blocks=args.skip_blocks,
        )
        token_utilization = 1.0
        dataset_len = None
    elif args.packed:
        if not args.data_path:
            raise ValueError("--data_path is required unless --data_mix_json is used.")
        dataset = PackedPretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
        if len(dataset) == 0:
            raise ValueError("Packed dataset produced zero blocks; lower max_seq_len or use more data.")
        token_utilization = 1.0
        dataset_len = len(dataset)
    else:
        if not args.data_path:
            raise ValueError("--data_path is required unless --data_mix_json is used.")
        dataset = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
        token_utilization = None
        dataset_len = len(dataset)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(not args.stream_packed),
        num_workers=args.num_workers,
        pin_memory=(device_type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    if args.stream_packed:
        total_steps = args.max_steps
    else:
        total_steps = args.epochs * len(loader)
    if args.max_steps > 0 and not args.stream_packed:
        total_steps = min(total_steps, args.max_steps)
    losses = []
    if device_type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    invocation_start_blocks = consumed_blocks
    model.train()
    optimizer.zero_grad(set_to_none=True)

    execution_limit = min(total_steps, args.stop_after_steps) if args.stop_after_steps > 0 else total_steps
    for epoch in range(start_epoch, args.epochs):
        for input_ids, labels in loader:
            if global_step >= execution_limit:
                break
            global_step += 1
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            consumed_blocks += int(input_ids.shape[0])
            lr = scheduled_lr(
                global_step,
                total_steps,
                args.learning_rate,
                args.warmup_steps,
                schedule=args.lr_schedule,
                min_lr_ratio=args.min_lr_ratio,
                stable_ratio=args.lr_stable_ratio,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            with autocast_ctx:
                result = model(input_ids, labels=labels)
                total_loss = result.loss + result.aux_loss
                if not torch.isfinite(total_loss).all():
                    tracker.alert(
                        "non-finite loss",
                        f"loss became non-finite at global_step={global_step}",
                        level="error",
                        step=global_step,
                    )
                    save_training_checkpoint(
                        checkpoint_path,
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        progress={
                            "global_step": global_step,
                            "optimizer_step": optimizer_step,
                            "consumed_blocks": consumed_blocks,
                            "epoch": epoch,
                        },
                        run_config=run_config,
                    )
                    tracker.finish(status="failed", summary={"reason": "non_finite_loss", "step": global_step})
                    raise FloatingPointError(f"non-finite loss at global_step={global_step}")
                loss = total_loss / args.accumulation_steps
            scaler.scale(loss).backward()

            if global_step % args.accumulation_steps == 0 or global_step == total_steps:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                if args.checkpoint_interval > 0 and optimizer_step % args.checkpoint_interval == 0:
                    save_training_checkpoint(
                        checkpoint_path,
                        model=model,
                        optimizer=optimizer,
                        scaler=scaler,
                        progress={
                            "global_step": global_step,
                            "optimizer_step": optimizer_step,
                            "consumed_blocks": consumed_blocks,
                            "epoch": epoch,
                        },
                        run_config=run_config,
                    )

            current_loss = float((loss.detach().cpu() * args.accumulation_steps).item())
            losses.append(current_loss)
            if global_step % args.log_interval == 0 or global_step == total_steps:
                elapsed = time.time() - started
                tokens = (consumed_blocks - invocation_start_blocks) * args.max_seq_len
                tokens_per_second = tokens / max(elapsed, 1e-6)
                tracker.log(
                    {
                        "epoch": epoch + 1,
                        "loss": current_loss,
                        "learning_rate": lr,
                        "tokens_per_second": tokens_per_second,
                        "consumed_blocks": consumed_blocks,
                        "optimizer_step": optimizer_step,
                        "max_memory_reserved_gb": (
                            torch.cuda.max_memory_reserved() / 1024 ** 3 if device_type == "cuda" else 0.0
                        ),
                    },
                    step=global_step,
                )
                print(
                    f"epoch={epoch + 1}/{args.epochs} step={global_step}/{total_steps} "
                    f"loss={current_loss:.4f} lr={lr:.8f} tokens/s={tokens_per_second:.0f}",
                    flush=True,
                )
        if global_step >= execution_limit:
            break

    if args.checkpoint_interval > 0 or args.resume_checkpoint:
        save_training_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            progress={
                "global_step": global_step,
                "optimizer_step": optimizer_step,
                "consumed_blocks": consumed_blocks,
                "epoch": min(args.epochs - 1, max(start_epoch, 0)),
            },
            run_config=run_config,
        )

    torch.save({k: v.detach().half().cpu() for k, v in model.state_dict().items()}, weight_path)
    run_seconds = time.time() - started
    invocation_tokens = (consumed_blocks - invocation_start_blocks) * args.max_seq_len
    run_status = "completed" if global_step >= total_steps else "stopped"
    summary = {
        "data_path": args.data_path,
        "data_mix_json": args.data_mix_json,
        "init_weight": args.init_weight,
        "packed": args.packed,
        "stream_packed": args.stream_packed,
        "shuffle_buffer": args.shuffle_buffer,
        "skip_blocks": args.skip_blocks,
        "packed_effective": args.packed or args.stream_packed,
        "fused_adamw_requested": args.fused_adamw,
        "fused_adamw_used": fused_used,
        "hidden_size": args.hidden_size,
        "num_hidden_layers": args.num_hidden_layers,
        "num_attention_heads": args.num_attention_heads,
        "num_key_value_heads": args.num_key_value_heads,
        "intermediate_size": args.intermediate_size,
        "max_seq_len": args.max_seq_len,
        "batch_size": args.batch_size,
        "accumulation_steps": args.accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "adam_beta1": args.adam_beta1,
        "adam_beta2": args.adam_beta2,
        "adam_eps": args.adam_eps,
        "lr_schedule": args.lr_schedule,
        "min_lr_ratio": args.min_lr_ratio,
        "lr_stable_ratio": args.lr_stable_ratio,
        "warmup_steps": args.warmup_steps,
        "epochs": args.epochs,
        "steps": global_step,
        "optimizer_steps": optimizer_step,
        "consumed_blocks": consumed_blocks,
        "max_steps": args.max_steps,
        "dataset_len": dataset_len,
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "seconds": round(run_seconds, 3),
        "slot_tokens_per_second": invocation_tokens / max(run_seconds, 1e-6),
        "token_utilization": token_utilization,
        "max_memory_reserved_gb": (torch.cuda.max_memory_reserved() / 1024 ** 3) if device_type == "cuda" else None,
        "weight_path": str(weight_path),
        "training_checkpoint_path": str(checkpoint_path) if checkpoint_path.exists() else None,
        "resumed_from": args.resume_checkpoint,
        "source_revision": source_revision,
        "metrics_path": str(metrics_path),
        "run_manifest_path": str(run_manifest_path),
        "trackio_enabled": bool(args.trackio_project),
        "status": run_status,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    tracker.finish(status=run_status, summary=summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description="MiniMind optimized pretrain runner for local experiments")
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--data_mix_json", default=None, help="Optional weighted data mix JSON for streaming packed training")
    parser.add_argument("--tokenizer_path", default=str(ROOT / "model"))
    parser.add_argument("--save_dir", default=str(Path(__file__).resolve().parent / "out"))
    parser.add_argument("--save_weight", default="pretrain_optimized")
    parser.add_argument("--init_weight", default=None, help="Optional model state dict to load before training")
    parser.add_argument("--resume_checkpoint", default=None, help="Resume model, optimizer, scaler, RNG, and data progress from a training checkpoint")
    parser.add_argument("--checkpoint_interval", type=int, default=0, help="Save a resumable checkpoint every N optimizer steps; 0 disables periodic saves")
    parser.add_argument("--stop_after_steps", type=int, default=0, help="Operational test hook: stop this invocation at an absolute dataloader step while preserving the target schedule")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=0, help="Stop early after this many dataloader steps; 0 means full epochs")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_eps", type=float, default=1e-8)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--lr_schedule", choices=["constant", "linear", "cosine", "wsd"], default="cosine")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--lr_stable_ratio", type=float, default=0.8)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--num_attention_heads", type=int, default=8)
    parser.add_argument("--num_key_value_heads", type=int, default=4)
    parser.add_argument("--intermediate_size", type=int, default=None)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--trackio_project", default=None, help="Optional Trackio project; JSONL metrics are always written")
    parser.add_argument("--trackio_space_id", default=None, help="Optional Hugging Face Space used by Trackio")
    parser.add_argument("--use_moe", type=int, default=0)
    parser.add_argument("--fused_adamw", action="store_true")
    parser.add_argument("--packed", action="store_true")
    parser.add_argument("--stream_packed", action="store_true", help="Stream jsonl and yield packed blocks without pre-tokenizing the whole file; requires --max_steps")
    parser.add_argument("--shuffle_buffer", type=int, default=0, help="Shuffle this many packed blocks when using --stream_packed; 0 disables streaming shuffle")
    parser.add_argument("--skip_blocks", type=int, default=0, help="Skip this many packed blocks before yielding data when using --stream_packed")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
