#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 <run_name> <base_weight> <num_key_value_heads> <intermediate_size_or_none> <max_steps> <max_seq_len>" >&2
  exit 2
fi

RUN_NAME="$1"
BASE_WEIGHT="$2"
NUM_KEY_VALUE_HEADS="$3"
INTERMEDIATE_SIZE="$4"
MAX_STEPS="$5"
MAX_SEQ_LEN="$6"

cd /home/oscar/minimind

PY=".venv/bin/python"
OUT="experiments/pretrain/platform_runs/${RUN_NAME}"

if [[ -e "$OUT/eval_marker.done" ]]; then
  echo "refusing to overwrite completed run: $OUT" >&2
  exit 3
fi

mkdir -p "$OUT"

{
  if [[ "$INTERMEDIATE_SIZE" == "none" ]]; then
    "$PY" experiments/pretrain/scripts/train_sft_optimized.py \
      --save_dir "$OUT/01_sft_probe" \
      --save_weight "${RUN_NAME}_sft_probe" \
      --init_weight "$BASE_WEIGHT" \
      --data_path dataset/sft_t2t_mini.jsonl \
      --tokenizer_path model \
      --hidden_size 768 \
      --num_hidden_layers 8 \
      --num_attention_heads 8 \
      --num_key_value_heads "$NUM_KEY_VALUE_HEADS" \
      --max_seq_len "$MAX_SEQ_LEN" \
      --batch_size 16 \
      --max_steps "$MAX_STEPS" \
      --learning_rate 0.00001 \
      --warmup_steps 50 \
      --dtype bfloat16 \
      --num_workers 0 \
      --seed 42 \
      --log_interval 50
  else
    "$PY" experiments/pretrain/scripts/train_sft_optimized.py \
      --save_dir "$OUT/01_sft_probe" \
      --save_weight "${RUN_NAME}_sft_probe" \
      --init_weight "$BASE_WEIGHT" \
      --data_path dataset/sft_t2t_mini.jsonl \
      --tokenizer_path model \
      --hidden_size 768 \
      --num_hidden_layers 8 \
      --num_attention_heads 8 \
      --num_key_value_heads "$NUM_KEY_VALUE_HEADS" \
      --intermediate_size "$INTERMEDIATE_SIZE" \
      --max_seq_len "$MAX_SEQ_LEN" \
      --batch_size 16 \
      --max_steps "$MAX_STEPS" \
      --learning_rate 0.00001 \
      --warmup_steps 50 \
      --dtype bfloat16 \
      --num_workers 0 \
      --seed 42 \
      --log_interval 50
  fi

  WEIGHT="$OUT/01_sft_probe/${RUN_NAME}_sft_probe_768.pth"
  if [[ "$INTERMEDIATE_SIZE" == "none" ]]; then
    "$PY" experiments/pretrain/scripts/evaluate_checkpoint_bundle.py \
      --run_name "$RUN_NAME" \
      --weight_path "$WEIGHT" \
      --source_run_dir "$OUT" \
      --hidden_size 768 \
      --num_hidden_layers 8 \
      --num_attention_heads 8 \
      --num_key_value_heads "$NUM_KEY_VALUE_HEADS" \
      --eval_seq_len 384 \
      --max_input_tokens 512 \
      --max_new_tokens 128 \
      --prompt_format chat \
      --dtype bfloat16 \
      --skip_loss
  else
    "$PY" experiments/pretrain/scripts/evaluate_checkpoint_bundle.py \
      --run_name "$RUN_NAME" \
      --weight_path "$WEIGHT" \
      --source_run_dir "$OUT" \
      --hidden_size 768 \
      --num_hidden_layers 8 \
      --num_attention_heads 8 \
      --num_key_value_heads "$NUM_KEY_VALUE_HEADS" \
      --intermediate_size "$INTERMEDIATE_SIZE" \
      --eval_seq_len 384 \
      --max_input_tokens 512 \
      --max_new_tokens 128 \
      --prompt_format chat \
      --dtype bfloat16 \
      --skip_loss
  fi

  date > "$OUT/eval_marker.done"
} 2>&1 | tee "$OUT/runner_console.log"
