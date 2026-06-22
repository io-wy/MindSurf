#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 <run_name> <num_key_value_heads> <intermediate_size> <max_steps>" >&2
  exit 2
fi

RUN_NAME="$1"
NUM_KEY_VALUE_HEADS="$2"
INTERMEDIATE_SIZE="$3"
MAX_STEPS="$4"

cd /home/oscar/minimind

PY=".venv/bin/python"
OUT="experiments/pretrain/platform_runs/${RUN_NAME}"

if [[ -e "$OUT/eval_strict_test.json" ]]; then
  echo "refusing to overwrite completed run: $OUT" >&2
  exit 3
fi

mkdir -p "$OUT"
cat > "$OUT/run_spec.json" <<EOF
{
  "run_name": "$RUN_NAME",
  "question": "architecture control: compare GQA/MHA and FFN width under fixed WSD recipe",
  "fixed": {
    "hidden_size": 768,
    "num_hidden_layers": 8,
    "num_attention_heads": 8,
    "max_seq_len": 384,
    "batch_size": 32,
    "learning_rate": 0.0005,
    "lr_schedule": "wsd",
    "lr_stable_ratio": 0.8,
    "warmup_steps": 200,
    "max_steps": $MAX_STEPS,
    "data_path": "experiments/pretrain/strict_splits/pretrain_strict_train.jsonl",
    "dtype": "bfloat16",
    "fused_adamw": true,
    "stream_packed": true,
    "seed": 42
  },
  "variables": {
    "num_key_value_heads": $NUM_KEY_VALUE_HEADS,
    "intermediate_size": $INTERMEDIATE_SIZE
  }
}
EOF

{
  "$PY" experiments/pretrain/scripts/train_pretrain_optimized.py \
    --save_dir "$OUT/01_wsd_s384" \
    --save_weight "${RUN_NAME}_wsd_s384" \
    --data_path experiments/pretrain/strict_splits/pretrain_strict_train.jsonl \
    --tokenizer_path model \
    --epochs 1 \
    --max_steps "$MAX_STEPS" \
    --batch_size 32 \
    --accumulation_steps 1 \
    --learning_rate 0.0005 \
    --weight_decay 0.01 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --adam_eps 0.00000001 \
    --warmup_steps 200 \
    --lr_schedule wsd \
    --lr_stable_ratio 0.8 \
    --min_lr_ratio 0.1 \
    --hidden_size 768 \
    --num_hidden_layers 8 \
    --num_attention_heads 8 \
    --num_key_value_heads "$NUM_KEY_VALUE_HEADS" \
    --intermediate_size "$INTERMEDIATE_SIZE" \
    --max_seq_len 384 \
    --num_workers 0 \
    --dtype bfloat16 \
    --seed 42 \
    --log_interval 50 \
    --shuffle_buffer 2048 \
    --fused_adamw \
    --stream_packed

  WEIGHT="$OUT/01_wsd_s384/${RUN_NAME}_wsd_s384_768.pth"
  "$PY" experiments/pretrain/scripts/eval_pretrain_loss.py \
    --data_path experiments/pretrain/strict_splits/pretrain_strict_val_2k.jsonl \
    --weight_path "$WEIGHT" \
    --output "$OUT/eval_strict_val.json" \
    --hidden_size 768 \
    --num_hidden_layers 8 \
    --num_attention_heads 8 \
    --num_key_value_heads "$NUM_KEY_VALUE_HEADS" \
    --intermediate_size "$INTERMEDIATE_SIZE" \
    --max_seq_len 384 \
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
    --num_key_value_heads "$NUM_KEY_VALUE_HEADS" \
    --intermediate_size "$INTERMEDIATE_SIZE" \
    --max_seq_len 384 \
    --batch_size 8 \
    --max_batches 250 \
    --dtype bfloat16
} 2>&1 | tee "$OUT/runner_console.log"
