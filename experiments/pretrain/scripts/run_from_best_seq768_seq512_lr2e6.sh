#!/usr/bin/env bash
set -euo pipefail

cd /home/oscar/minimind

PY=".venv/bin/python"
OUT="experiments/pretrain/platform_runs/from_best_seq768_seq512_lr2e6"
BASE="experiments/pretrain/platform_runs/best_seq768_from_wsd_cooldown/01_seq768_lr5e6/best_seq768_from_wsd_cooldown_seq768_lr5e6_768.pth"

mkdir -p "$OUT"

{
  "$PY" experiments/pretrain/scripts/train_pretrain_optimized.py \
    --save_dir "$OUT/01_seq512_lr2e6" \
    --save_weight from_best_seq768_seq512_lr2e6 \
    --init_weight "$BASE" \
    --data_path experiments/pretrain/strict_splits/pretrain_strict_train.jsonl \
    --tokenizer_path model \
    --epochs 1 \
    --max_steps 1200 \
    --batch_size 24 \
    --accumulation_steps 1 \
    --learning_rate 0.000002 \
    --warmup_steps 20 \
    --lr_schedule cosine \
    --min_lr_ratio 0.1 \
    --hidden_size 768 \
    --num_hidden_layers 8 \
    --max_seq_len 512 \
    --num_workers 0 \
    --dtype bfloat16 \
    --seed 42 \
    --log_interval 50 \
    --shuffle_buffer 2048 \
    --fused_adamw \
    --stream_packed

  WEIGHT="$OUT/01_seq512_lr2e6/from_best_seq768_seq512_lr2e6_768.pth"
  "$PY" experiments/pretrain/scripts/eval_pretrain_loss.py \
    --data_path experiments/pretrain/strict_splits/pretrain_strict_val_2k.jsonl \
    --weight_path "$WEIGHT" \
    --output "$OUT/eval_strict_val.json" \
    --hidden_size 768 \
    --num_hidden_layers 8 \
    --max_seq_len 512 \
    --batch_size 8 \
    --max_batches 250 \
    --dtype bfloat16
  "$PY" experiments/pretrain/scripts/eval_pretrain_loss.py \
    --data_path experiments/pretrain/strict_splits/pretrain_strict_test_2k.jsonl \
    --weight_path "$WEIGHT" \
    --output "$OUT/eval_strict_test.json" \
    --hidden_size 768 \
    --num_hidden_layers 8 \
    --max_seq_len 512 \
    --batch_size 8 \
    --max_batches 250 \
    --dtype bfloat16
} 2>&1 | tee "$OUT/runner_console.log"
