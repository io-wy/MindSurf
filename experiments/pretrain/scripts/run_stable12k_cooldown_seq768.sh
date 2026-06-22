#!/usr/bin/env bash
set -euo pipefail

cd /home/oscar/minimind

PY=".venv/bin/python"
OUT="experiments/pretrain/platform_runs/stable12k_cooldown_seq768"
BASE="experiments/pretrain/platform_runs/wsd_s384_12k_stable085/01_wsd_s384/wsd_s384_12k_stable085_wsd_s384_768.pth"

if [[ -e "$OUT/eval_strict_test.json" ]]; then
  echo "refusing to overwrite completed run: $OUT" >&2
  exit 3
fi

mkdir -p "$OUT"

{
  "$PY" experiments/pretrain/scripts/train_pretrain_optimized.py \
    --save_dir "$OUT/01_s384_lr5e5" \
    --save_weight stable12k_cooldown_seq768_s384_lr5e5 \
    --init_weight "$BASE" \
    --data_path experiments/pretrain/strict_splits/pretrain_strict_train.jsonl \
    --tokenizer_path model \
    --epochs 1 \
    --max_steps 1800 \
    --batch_size 32 \
    --accumulation_steps 1 \
    --learning_rate 0.00005 \
    --warmup_steps 20 \
    --lr_schedule cosine \
    --min_lr_ratio 0.1 \
    --hidden_size 768 \
    --num_hidden_layers 8 \
    --max_seq_len 384 \
    --num_workers 0 \
    --dtype bfloat16 \
    --seed 42 \
    --log_interval 50 \
    --shuffle_buffer 2048 \
    --fused_adamw \
    --stream_packed

  STAGE1="$OUT/01_s384_lr5e5/stable12k_cooldown_seq768_s384_lr5e5_768.pth"
  "$PY" experiments/pretrain/scripts/train_pretrain_optimized.py \
    --save_dir "$OUT/02_seq768_lr5e6" \
    --save_weight stable12k_cooldown_seq768_seq768_lr5e6 \
    --init_weight "$STAGE1" \
    --data_path experiments/pretrain/strict_splits/pretrain_strict_train.jsonl \
    --tokenizer_path model \
    --epochs 1 \
    --max_steps 800 \
    --batch_size 16 \
    --accumulation_steps 1 \
    --learning_rate 0.000005 \
    --warmup_steps 20 \
    --lr_schedule cosine \
    --min_lr_ratio 0.1 \
    --hidden_size 768 \
    --num_hidden_layers 8 \
    --max_seq_len 768 \
    --num_workers 0 \
    --dtype bfloat16 \
    --seed 42 \
    --log_interval 50 \
    --shuffle_buffer 2048 \
    --fused_adamw \
    --stream_packed

  WEIGHT="$OUT/02_seq768_lr5e6/stable12k_cooldown_seq768_seq768_lr5e6_768.pth"
  "$PY" experiments/pretrain/scripts/eval_pretrain_loss.py \
    --data_path experiments/pretrain/strict_splits/pretrain_strict_val_2k.jsonl \
    --weight_path "$WEIGHT" \
    --output "$OUT/eval_strict_val.json" \
    --hidden_size 768 \
    --num_hidden_layers 8 \
    --max_seq_len 768 \
    --batch_size 8 \
    --max_batches 250 \
    --dtype bfloat16
  "$PY" experiments/pretrain/scripts/eval_pretrain_loss.py \
    --data_path experiments/pretrain/strict_splits/pretrain_strict_test_2k.jsonl \
    --weight_path "$WEIGHT" \
    --output "$OUT/eval_strict_test.json" \
    --hidden_size 768 \
    --num_hidden_layers 8 \
    --max_seq_len 768 \
    --batch_size 8 \
    --max_batches 250 \
    --dtype bfloat16
} 2>&1 | tee "$OUT/runner_console.log"
