#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 7 || $# -gt 8 ]]; then
  echo "usage: $0 <run_name> <data_mode:path|mix> <data_source> <base_weight> <max_steps> <learning_rate> <max_seq_len> [batch_size]" >&2
  exit 2
fi

RUN_NAME="$1"
DATA_MODE="$2"
DATA_SOURCE="$3"
BASE_WEIGHT="$4"
MAX_STEPS="$5"
LEARNING_RATE="$6"
MAX_SEQ_LEN="$7"
BATCH_OVERRIDE="${8:-}"

cd /home/oscar/minimind

PY=".venv/bin/python"
OUT="experiments/pretrain/platform_runs/${RUN_NAME}"
BATCH_SIZE=24
if [[ "$MAX_SEQ_LEN" -ge 768 ]]; then
  BATCH_SIZE=16
fi
if [[ -n "$BATCH_OVERRIDE" ]]; then
  BATCH_SIZE="$BATCH_OVERRIDE"
fi

if [[ "$DATA_MODE" != "path" && "$DATA_MODE" != "mix" ]]; then
  echo "data_mode must be 'path' or 'mix'" >&2
  exit 2
fi
if [[ -e "$OUT/eval_strict_test.json" ]]; then
  echo "refusing to overwrite completed run: $OUT" >&2
  exit 3
fi

mkdir -p "$OUT"
cat > "$OUT/run_spec.json" <<EOF
{
  "run_name": "$RUN_NAME",
  "question": "pretraining continuation: compare targeted repair mixes against strict-only control",
  "data_mode": "$DATA_MODE",
  "data_source": "$DATA_SOURCE",
  "base_weight": "$BASE_WEIGHT",
  "fixed": {
    "hidden_size": 768,
    "num_hidden_layers": 8,
    "num_attention_heads": 8,
    "num_key_value_heads": 8,
    "intermediate_size": 3072,
    "max_seq_len": $MAX_SEQ_LEN,
    "batch_size": $BATCH_SIZE,
    "learning_rate": $LEARNING_RATE,
    "lr_schedule": "cosine",
    "warmup_steps": 20,
    "max_steps": $MAX_STEPS,
    "dtype": "bfloat16",
    "fused_adamw": true,
    "stream_packed": true,
    "seed": 42
  }
}
EOF

{
  if [[ "$DATA_MODE" == "path" ]]; then
    "$PY" experiments/pretrain/scripts/train_pretrain_optimized.py \
      --save_dir "$OUT/01_continue" \
      --save_weight "${RUN_NAME}_continue" \
      --init_weight "$BASE_WEIGHT" \
      --data_path "$DATA_SOURCE" \
      --tokenizer_path model \
      --epochs 1 \
      --max_steps "$MAX_STEPS" \
      --batch_size "$BATCH_SIZE" \
      --accumulation_steps 1 \
      --learning_rate "$LEARNING_RATE" \
      --weight_decay 0.01 \
      --adam_beta1 0.9 \
      --adam_beta2 0.999 \
      --adam_eps 0.00000001 \
      --warmup_steps 20 \
      --lr_schedule cosine \
      --min_lr_ratio 0.1 \
      --hidden_size 768 \
      --num_hidden_layers 8 \
      --num_attention_heads 8 \
      --num_key_value_heads 8 \
      --intermediate_size 3072 \
      --max_seq_len "$MAX_SEQ_LEN" \
      --num_workers 0 \
      --dtype bfloat16 \
      --seed 42 \
      --log_interval 50 \
      --shuffle_buffer 2048 \
      --fused_adamw \
      --stream_packed
  else
    "$PY" experiments/pretrain/scripts/train_pretrain_optimized.py \
      --save_dir "$OUT/01_continue" \
      --save_weight "${RUN_NAME}_continue" \
      --init_weight "$BASE_WEIGHT" \
      --data_mix_json "$DATA_SOURCE" \
      --tokenizer_path model \
      --epochs 1 \
      --max_steps "$MAX_STEPS" \
      --batch_size "$BATCH_SIZE" \
      --accumulation_steps 1 \
      --learning_rate "$LEARNING_RATE" \
      --weight_decay 0.01 \
      --adam_beta1 0.9 \
      --adam_beta2 0.999 \
      --adam_eps 0.00000001 \
      --warmup_steps 20 \
      --lr_schedule cosine \
      --min_lr_ratio 0.1 \
      --hidden_size 768 \
      --num_hidden_layers 8 \
      --num_attention_heads 8 \
      --num_key_value_heads 8 \
      --intermediate_size 3072 \
      --max_seq_len "$MAX_SEQ_LEN" \
      --num_workers 0 \
      --dtype bfloat16 \
      --seed 42 \
      --log_interval 50 \
      --shuffle_buffer 2048 \
      --fused_adamw \
      --stream_packed
  fi

  WEIGHT="$OUT/01_continue/${RUN_NAME}_continue_768.pth"
  "$PY" experiments/pretrain/scripts/eval_pretrain_loss.py \
    --data_path experiments/pretrain/strict_splits/pretrain_strict_val_2k.jsonl \
    --weight_path "$WEIGHT" \
    --output "$OUT/eval_strict_val.json" \
    --hidden_size 768 \
    --num_hidden_layers 8 \
    --num_attention_heads 8 \
    --num_key_value_heads 8 \
    --intermediate_size 3072 \
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
    --num_key_value_heads 8 \
    --intermediate_size 3072 \
    --max_seq_len 384 \
    --batch_size 8 \
    --max_batches 250 \
    --dtype bfloat16
  "$PY" experiments/pretrain/scripts/evaluate_checkpoint_bundle.py \
    --run_name "$RUN_NAME" \
    --weight_path "$WEIGHT" \
    --source_run_dir "$OUT" \
    --hidden_size 768 \
    --num_hidden_layers 8 \
    --num_attention_heads 8 \
    --num_key_value_heads 8 \
    --intermediate_size 3072 \
    --eval_seq_len 384 \
    --max_input_tokens 384 \
    --max_new_tokens 96 \
    --dtype bfloat16 \
    --skip_loss
  date > "$OUT/eval_marker.done"
} 2>&1 | tee "$OUT/runner_console.log"
