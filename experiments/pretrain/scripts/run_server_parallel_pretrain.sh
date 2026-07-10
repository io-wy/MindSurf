#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/oscar/minimind}"
cd "$ROOT"

PY=".venv/bin/python"
TRAIN="experiments/pretrain/scripts/train_pretrain_optimized.py"
EVAL="experiments/pretrain/scripts/eval_pretrain_loss.py"
DATA="experiments/pretrain/strict_splits/pretrain_strict_train.jsonl"
VAL="experiments/pretrain/strict_splits/pretrain_strict_val_2k.jsonl"
TEST="experiments/pretrain/strict_splits/pretrain_strict_test_2k.jsonl"
BASE="experiments/pretrain/platform_runs/reproduce_local_baseline_s384_5000/01_baseline_s384/reproduce_local_baseline_s384_5000_baseline_s384_768.pth"
LAUNCH_ROOT="experiments/pretrain/platform_runs/_launch_scripts"

mkdir -p "$LAUNCH_ROOT"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "missing required file: $path" >&2
    exit 1
  fi
}

require_file "$PY"
require_file "$TRAIN"
require_file "$EVAL"
require_file "$DATA"
require_file "$VAL"
require_file "$TEST"
require_file "$BASE"

write_job() {
  local name="$1"
  local script="$LAUNCH_ROOT/${name}.sh"
  cat >"$script"
  chmod +x "$script"
  echo "$script"
}

start_job() {
  local session="$1"
  local script="$2"
  if tmux has-session -t "$session" 2>/dev/null; then
    echo "session already exists: $session"
    return
  fi
  tmux new-session -d -s "$session" "cd '$ROOT' && bash '$script'"
  echo "started $session -> $script"
}

job_cooldown="$(write_job cooldown_from_baseline <<'JOB'
#!/usr/bin/env bash
set -euo pipefail
cd /home/oscar/minimind
PY=".venv/bin/python"
OUT="experiments/pretrain/platform_runs/cooldown_from_baseline"
BASE="experiments/pretrain/platform_runs/reproduce_local_baseline_s384_5000/01_baseline_s384/reproduce_local_baseline_s384_5000_baseline_s384_768.pth"
mkdir -p "$OUT"
{
  "$PY" experiments/pretrain/scripts/train_pretrain_optimized.py \
    --save_dir "$OUT/01_cooldown_s384" \
    --save_weight cooldown_from_baseline_cooldown_s384 \
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
  WEIGHT="$OUT/01_cooldown_s384/cooldown_from_baseline_cooldown_s384_768.pth"
  "$PY" experiments/pretrain/scripts/eval_pretrain_loss.py --data_path experiments/pretrain/strict_splits/pretrain_strict_val_2k.jsonl --weight_path "$WEIGHT" --output "$OUT/eval_strict_val.json" --hidden_size 768 --num_hidden_layers 8 --max_seq_len 384 --batch_size 8 --max_batches 250 --dtype bfloat16
  "$PY" experiments/pretrain/scripts/eval_pretrain_loss.py --data_path experiments/pretrain/strict_splits/pretrain_strict_test_2k.jsonl --weight_path "$WEIGHT" --output "$OUT/eval_strict_test.json" --hidden_size 768 --num_hidden_layers 8 --max_seq_len 384 --batch_size 8 --max_batches 250 --dtype bfloat16
} 2>&1 | tee "$OUT/runner_console.log"
JOB
)"

job_balanced="$(write_job balanced_seq512_from_baseline <<'JOB'
#!/usr/bin/env bash
set -euo pipefail
cd /home/oscar/minimind
PY=".venv/bin/python"
OUT="experiments/pretrain/platform_runs/balanced_seq512_from_baseline"
BASE="experiments/pretrain/platform_runs/reproduce_local_baseline_s384_5000/01_baseline_s384/reproduce_local_baseline_s384_5000_baseline_s384_768.pth"
mkdir -p "$OUT"
{
  "$PY" experiments/pretrain/scripts/train_pretrain_optimized.py \
    --save_dir "$OUT/01_seq512_lr2e5" \
    --save_weight balanced_seq512_from_baseline_seq512_lr2e5 \
    --init_weight "$BASE" \
    --data_path experiments/pretrain/strict_splits/pretrain_strict_train.jsonl \
    --tokenizer_path model \
    --epochs 1 \
    --max_steps 1200 \
    --batch_size 24 \
    --accumulation_steps 1 \
    --learning_rate 0.00002 \
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
  STAGE1="$OUT/01_seq512_lr2e5/balanced_seq512_from_baseline_seq512_lr2e5_768.pth"
  "$PY" experiments/pretrain/scripts/train_pretrain_optimized.py \
    --save_dir "$OUT/02_post_lr5e6" \
    --save_weight balanced_seq512_from_baseline_post_lr5e6 \
    --init_weight "$STAGE1" \
    --data_path experiments/pretrain/strict_splits/pretrain_strict_train.jsonl \
    --tokenizer_path model \
    --epochs 1 \
    --max_steps 600 \
    --batch_size 24 \
    --accumulation_steps 1 \
    --learning_rate 0.000005 \
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
  WEIGHT="$OUT/02_post_lr5e6/balanced_seq512_from_baseline_post_lr5e6_768.pth"
  "$PY" experiments/pretrain/scripts/eval_pretrain_loss.py --data_path experiments/pretrain/strict_splits/pretrain_strict_val_2k.jsonl --weight_path "$WEIGHT" --output "$OUT/eval_strict_val.json" --hidden_size 768 --num_hidden_layers 8 --max_seq_len 512 --batch_size 8 --max_batches 250 --dtype bfloat16
  "$PY" experiments/pretrain/scripts/eval_pretrain_loss.py --data_path experiments/pretrain/strict_splits/pretrain_strict_test_2k.jsonl --weight_path "$WEIGHT" --output "$OUT/eval_strict_test.json" --hidden_size 768 --num_hidden_layers 8 --max_seq_len 512 --batch_size 8 --max_batches 250 --dtype bfloat16
} 2>&1 | tee "$OUT/runner_console.log"
JOB
)"

job_scaling="$(write_job scaling_probe_h512_l8 <<'JOB'
#!/usr/bin/env bash
set -euo pipefail
cd /home/oscar/minimind
.venv/bin/python experiments/pretrain/scripts/run_platform_experiment.py --experiment scaling_probe_h512_l8 --run-eval 2>&1 | tee experiments/pretrain/platform_runs/scaling_probe_h512_l8/runner_console.log
JOB
)"

job_wsd="$(write_job wsd_s384_long_probe <<'JOB'
#!/usr/bin/env bash
set -euo pipefail
cd /home/oscar/minimind
.venv/bin/python experiments/pretrain/scripts/run_platform_experiment.py --experiment wsd_s384_long_probe --run-eval 2>&1 | tee experiments/pretrain/platform_runs/wsd_s384_long_probe/runner_console.log
JOB
)"

job_tail="$(write_job tail_seq512_from_baseline <<'JOB'
#!/usr/bin/env bash
set -euo pipefail
cd /home/oscar/minimind
PY=".venv/bin/python"
OUT="experiments/pretrain/platform_runs/tail_seq512_from_baseline"
BASE="experiments/pretrain/platform_runs/reproduce_local_baseline_s384_5000/01_baseline_s384/reproduce_local_baseline_s384_5000_baseline_s384_768.pth"
mkdir -p "$OUT"
{
  "$PY" experiments/pretrain/scripts/train_pretrain_optimized.py \
    --save_dir "$OUT/01_seq512_lr1e5" \
    --save_weight tail_seq512_from_baseline_seq512_lr1e5 \
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
    --seed 7 \
    --log_interval 50 \
    --shuffle_buffer 2048 \
    --fused_adamw \
    --stream_packed
  WEIGHT="$OUT/01_seq512_lr1e5/tail_seq512_from_baseline_seq512_lr1e5_768.pth"
  "$PY" experiments/pretrain/scripts/eval_pretrain_loss.py --data_path experiments/pretrain/strict_splits/pretrain_strict_val_2k.jsonl --weight_path "$WEIGHT" --output "$OUT/eval_strict_val.json" --hidden_size 768 --num_hidden_layers 8 --max_seq_len 512 --batch_size 8 --max_batches 250 --dtype bfloat16
  "$PY" experiments/pretrain/scripts/eval_pretrain_loss.py --data_path experiments/pretrain/strict_splits/pretrain_strict_test_2k.jsonl --weight_path "$WEIGHT" --output "$OUT/eval_strict_test.json" --hidden_size 768 --num_hidden_layers 8 --max_seq_len 512 --batch_size 8 --max_batches 250 --dtype bfloat16
} 2>&1 | tee "$OUT/runner_console.log"
JOB
)"

start_job minimind_cooldown "$job_cooldown"
start_job minimind_balanced_seq512 "$job_balanced"
start_job minimind_scaling_h512 "$job_scaling"

if [[ "${1:-}" == "--include-wsd" ]]; then
  start_job minimind_wsd_s384 "$job_wsd"
fi

if [[ "${1:-}" == "--include-tail" || "${2:-}" == "--include-tail" ]]; then
  start_job minimind_tail_seq512 "$job_tail"
fi

tmux list-sessions
