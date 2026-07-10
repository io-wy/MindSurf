#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 <run_name> <intermediate_size> <base_weight> <mode>" >&2
  echo "modes: seq768 | seq512 | seq1024" >&2
  exit 2
fi

RUN_NAME="$1"
INTERMEDIATE_SIZE="$2"
BASE="$3"
MODE="$4"

cd /home/oscar/minimind

PY=".venv/bin/python"
OUT="experiments/pretrain/platform_runs/${RUN_NAME}"

if [[ -e "$OUT/eval_strict_test.json" ]]; then
  echo "refusing to overwrite completed run: $OUT" >&2
  exit 3
fi

mkdir -p "$OUT"

eval_final() {
  local weight="$1"
  local seq_len="$2"
  "$PY" experiments/pretrain/scripts/eval_pretrain_loss.py \
    --data_path experiments/pretrain/strict_splits/pretrain_strict_val_2k.jsonl \
    --weight_path "$weight" \
    --output "$OUT/eval_strict_val.json" \
    --hidden_size 768 \
    --num_hidden_layers 8 \
    --num_attention_heads 8 \
    --num_key_value_heads 8 \
    --intermediate_size "$INTERMEDIATE_SIZE" \
    --max_seq_len "$seq_len" \
    --batch_size 8 \
    --max_batches 250 \
    --dtype bfloat16
  "$PY" experiments/pretrain/scripts/eval_pretrain_loss.py \
    --data_path experiments/pretrain/strict_splits/pretrain_strict_test_2k.jsonl \
    --weight_path "$weight" \
    --output "$OUT/eval_strict_test.json" \
    --hidden_size 768 \
    --num_hidden_layers 8 \
    --num_attention_heads 8 \
    --num_key_value_heads 8 \
    --intermediate_size "$INTERMEDIATE_SIZE" \
    --max_seq_len "$seq_len" \
    --batch_size 8 \
    --max_batches 250 \
    --dtype bfloat16
}

{
  "$PY" experiments/pretrain/scripts/train_pretrain_optimized.py \
    --save_dir "$OUT/01_s384_lr5e5" \
    --save_weight "${RUN_NAME}_s384_lr5e5" \
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
    --num_attention_heads 8 \
    --num_key_value_heads 8 \
    --intermediate_size "$INTERMEDIATE_SIZE" \
    --max_seq_len 384 \
    --num_workers 0 \
    --dtype bfloat16 \
    --seed 42 \
    --log_interval 50 \
    --shuffle_buffer 2048 \
    --fused_adamw \
    --stream_packed

  STAGE1="$OUT/01_s384_lr5e5/${RUN_NAME}_s384_lr5e5_768.pth"

  if [[ "$MODE" == "seq768" ]]; then
    "$PY" experiments/pretrain/scripts/train_pretrain_optimized.py \
      --save_dir "$OUT/02_seq768_lr5e6" \
      --save_weight "${RUN_NAME}_seq768_lr5e6" \
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
      --num_attention_heads 8 \
      --num_key_value_heads 8 \
      --intermediate_size "$INTERMEDIATE_SIZE" \
      --max_seq_len 768 \
      --num_workers 0 \
      --dtype bfloat16 \
      --seed 42 \
      --log_interval 50 \
      --shuffle_buffer 2048 \
      --fused_adamw \
      --stream_packed
    eval_final "$OUT/02_seq768_lr5e6/${RUN_NAME}_seq768_lr5e6_768.pth" 768

  elif [[ "$MODE" == "seq512" ]]; then
    "$PY" experiments/pretrain/scripts/train_pretrain_optimized.py \
      --save_dir "$OUT/02_seq512_lr1e5" \
      --save_weight "${RUN_NAME}_seq512_lr1e5" \
      --init_weight "$STAGE1" \
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
      --num_attention_heads 8 \
      --num_key_value_heads 8 \
      --intermediate_size "$INTERMEDIATE_SIZE" \
      --max_seq_len 512 \
      --num_workers 0 \
      --dtype bfloat16 \
      --seed 42 \
      --log_interval 50 \
      --shuffle_buffer 2048 \
      --fused_adamw \
      --stream_packed
    STAGE2="$OUT/02_seq512_lr1e5/${RUN_NAME}_seq512_lr1e5_768.pth"
    "$PY" experiments/pretrain/scripts/train_pretrain_optimized.py \
      --save_dir "$OUT/03_seq512_lr3e6" \
      --save_weight "${RUN_NAME}_seq512_lr3e6" \
      --init_weight "$STAGE2" \
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
      --num_attention_heads 8 \
      --num_key_value_heads 8 \
      --intermediate_size "$INTERMEDIATE_SIZE" \
      --max_seq_len 512 \
      --num_workers 0 \
      --dtype bfloat16 \
      --seed 42 \
      --log_interval 50 \
      --shuffle_buffer 2048 \
      --fused_adamw \
      --stream_packed
    eval_final "$OUT/03_seq512_lr3e6/${RUN_NAME}_seq512_lr3e6_768.pth" 512

  elif [[ "$MODE" == "seq1024" ]]; then
    "$PY" experiments/pretrain/scripts/train_pretrain_optimized.py \
      --save_dir "$OUT/02_seq1024_lr5e6" \
      --save_weight "${RUN_NAME}_seq1024_lr5e6" \
      --init_weight "$STAGE1" \
      --data_path experiments/pretrain/strict_splits/pretrain_strict_train.jsonl \
      --tokenizer_path model \
      --epochs 1 \
      --max_steps 600 \
      --batch_size 8 \
      --accumulation_steps 1 \
      --learning_rate 0.000005 \
      --warmup_steps 20 \
      --lr_schedule cosine \
      --min_lr_ratio 0.1 \
      --hidden_size 768 \
      --num_hidden_layers 8 \
      --num_attention_heads 8 \
      --num_key_value_heads 8 \
      --intermediate_size "$INTERMEDIATE_SIZE" \
      --max_seq_len 1024 \
      --num_workers 0 \
      --dtype bfloat16 \
      --seed 42 \
      --log_interval 50 \
      --shuffle_buffer 2048 \
      --fused_adamw \
      --stream_packed
    eval_final "$OUT/02_seq1024_lr5e6/${RUN_NAME}_seq1024_lr5e6_768.pth" 1024

  else
    echo "unknown mode: $MODE" >&2
    exit 2
  fi
} 2>&1 | tee "$OUT/runner_console.log"
