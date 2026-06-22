#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 <run_name> <mix_json> <base_weight> <max_steps>" >&2
  exit 2
fi

RUN_NAME="$1"
MIX_JSON="$2"
BASE_WEIGHT="$3"
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
  "question": "data flywheel continuation: compare a 5% supplement mix against strict-only continuation",
  "base_weight": "$BASE_WEIGHT",
  "mix_json": "$MIX_JSON",
  "fixed": {
    "hidden_size": 768,
    "num_hidden_layers": 8,
    "num_attention_heads": 8,
    "num_key_value_heads": 8,
    "intermediate_size": 3072,
    "max_seq_len": 384,
    "batch_size": 32,
    "learning_rate": 0.00005,
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
  "$PY" experiments/pretrain/scripts/train_pretrain_optimized.py \
    --save_dir "$OUT/01_s384_mix_lr5e5" \
    --save_weight "${RUN_NAME}_s384_mix_lr5e5" \
    --init_weight "$BASE_WEIGHT" \
    --data_mix_json "$MIX_JSON" \
    --tokenizer_path model \
    --epochs 1 \
    --max_steps "$MAX_STEPS" \
    --batch_size 32 \
    --accumulation_steps 1 \
    --learning_rate 0.00005 \
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
    --max_seq_len 384 \
    --num_workers 0 \
    --dtype bfloat16 \
    --seed 42 \
    --log_interval 50 \
    --shuffle_buffer 2048 \
    --fused_adamw \
    --stream_packed

  WEIGHT="$OUT/01_s384_mix_lr5e5/${RUN_NAME}_s384_mix_lr5e5_768.pth"
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
} 2>&1 | tee "$OUT/runner_console.log"
