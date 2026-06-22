#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 <run_name> <num_key_value_heads> <base_weight> <mode>" >&2
  echo "modes: seq768_lr2e6 | seq512_lr2e6 | seq1024_lr2e6" >&2
  exit 2
fi

RUN_NAME="$1"
NUM_KV_HEADS="$2"
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

case "$MODE" in
  seq768_lr2e6)
    SEQ_LEN=768
    BATCH_SIZE=16
    STEPS=800
    LR=0.000002
    WARMUP=10
    ;;
  seq512_lr2e6)
    SEQ_LEN=512
    BATCH_SIZE=24
    STEPS=1200
    LR=0.000002
    WARMUP=10
    ;;
  seq1024_lr2e6)
    SEQ_LEN=1024
    BATCH_SIZE=8
    STEPS=600
    LR=0.000002
    WARMUP=10
    ;;
  *)
    echo "unknown mode: $MODE" >&2
    exit 2
    ;;
esac

{
  "$PY" experiments/pretrain/scripts/train_pretrain_optimized.py \
    --save_dir "$OUT/01_${MODE}" \
    --save_weight "${RUN_NAME}_${MODE}" \
    --init_weight "$BASE" \
    --data_path experiments/pretrain/strict_splits/pretrain_strict_train.jsonl \
    --tokenizer_path model \
    --epochs 1 \
    --max_steps "$STEPS" \
    --batch_size "$BATCH_SIZE" \
    --accumulation_steps 1 \
    --learning_rate "$LR" \
    --warmup_steps "$WARMUP" \
    --lr_schedule cosine \
    --min_lr_ratio 0.1 \
    --hidden_size 768 \
    --num_hidden_layers 8 \
    --num_attention_heads 8 \
    --num_key_value_heads "$NUM_KV_HEADS" \
    --max_seq_len "$SEQ_LEN" \
    --num_workers 0 \
    --dtype bfloat16 \
    --seed 42 \
    --log_interval 50 \
    --shuffle_buffer 2048 \
    --fused_adamw \
    --stream_packed

  WEIGHT="$OUT/01_${MODE}/${RUN_NAME}_${MODE}_768.pth"
  "$PY" experiments/pretrain/scripts/eval_pretrain_loss.py \
    --data_path experiments/pretrain/strict_splits/pretrain_strict_val_2k.jsonl \
    --weight_path "$WEIGHT" \
    --output "$OUT/eval_strict_val.json" \
    --hidden_size 768 \
    --num_hidden_layers 8 \
    --num_attention_heads 8 \
    --num_key_value_heads "$NUM_KV_HEADS" \
    --max_seq_len "$SEQ_LEN" \
    --batch_size 8 \
    --max_batches 250 \
    --dtype bfloat16
  "$PY" experiments/pretrain/scripts/eval_pretrain_loss.py \
    --data_path experiments/pretrain/strict_splits/pretrain_strict_test_2k.jsonl \
    --weight_path "$WEIGHT" \
    --output "$OUT/eval_strict_test.json" \
    --hidden_size 768 \
    --num_hidden_layers 8 \
    --num_attention_heads 8 \
    --num_key_value_heads "$NUM_KV_HEADS" \
    --max_seq_len "$SEQ_LEN" \
    --batch_size 8 \
    --max_batches 250 \
    --dtype bfloat16
} 2>&1 | tee "$OUT/runner_console.log"
