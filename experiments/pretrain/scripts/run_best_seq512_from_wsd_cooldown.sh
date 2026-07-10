#!/usr/bin/env bash
set -euo pipefail

cd /home/oscar/minimind

PY=".venv/bin/python"
OUT="experiments/pretrain/platform_runs/best_seq512_from_wsd_cooldown"
BASE="experiments/pretrain/platform_runs/wsd_cooldown_s384/01_s384_lr5e5/wsd_cooldown_s384_lr5e5_768.pth"

mkdir -p "$OUT"

{
  "$PY" experiments/pretrain/scripts/train_pretrain_optimized.py \
    --save_dir "$OUT/01_seq512_lr1e5" \
    --save_weight best_seq512_from_wsd_cooldown_seq512_lr1e5 \
    --init_weight "$BASE" \
    --data_path experiments/pretrain/strict_splits/pretrain_strict_train.jsonl \
    --tokenizer_path model \
    --epochs 1 \
    --max_steps 1200 \
    --batch_size 24 \
    --accumulation_steps 1 \
    --learning_rate 0.00001 \
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

  STAGE1="$OUT/01_seq512_lr1e5/best_seq512_from_wsd_cooldown_seq512_lr1e5_768.pth"
  "$PY" experiments/pretrain/scripts/train_pretrain_optimized.py \
    --save_dir "$OUT/02_seq512_lr3e6" \
    --save_weight best_seq512_from_wsd_cooldown_seq512_lr3e6 \
    --init_weight "$STAGE1" \
    --data_path experiments/pretrain/strict_splits/pretrain_strict_train.jsonl \
    --tokenizer_path model \
    --epochs 1 \
    --max_steps 600 \
    --batch_size 24 \
    --accumulation_steps 1 \
    --learning_rate 0.000003 \
    --warmup_steps 10 \
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

  WEIGHT="$OUT/02_seq512_lr3e6/best_seq512_from_wsd_cooldown_seq512_lr3e6_768.pth"
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
