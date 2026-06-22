#!/usr/bin/env bash
set -euo pipefail

cd /home/oscar/minimind

BASE_WEIGHT="experiments/pretrain/platform_runs/pretrain_q100_stage2_quality_s512_b16_lr5e6/01_continue/pretrain_q100_stage2_quality_s512_b16_lr5e6_continue_768.pth"
MCQ_SUITE="experiments/pretrain/eval_suites/local_mcq_benchmark_v1.jsonl"
MCQ_OUT="experiments/pretrain/diagnostics/local_mcq_benchmark_v1"

launch_run() {
  local session="$1"
  local run_name="$2"
  local mix_path="$3"

  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "session exists, skip: ${session}"
    return 0
  fi
  if [[ -e "experiments/pretrain/platform_runs/${run_name}/eval_strict_test.json" ]]; then
    echo "completed run exists, skip: ${run_name}"
    return 0
  fi

  local cmd
  cmd=$(cat <<EOF
set -euo pipefail
cd /home/oscar/minimind
RUN_NAME="${run_name}"
MIX_PATH="${mix_path}"
BASE_WEIGHT="${BASE_WEIGHT}"
bash experiments/pretrain/scripts/run_pretrain_continuation_eval.sh "\${RUN_NAME}" mix "\${MIX_PATH}" "\${BASE_WEIGHT}" 700 8e-7 512 16
WEIGHT="experiments/pretrain/platform_runs/\${RUN_NAME}/01_continue/\${RUN_NAME}_continue_768.pth"
mkdir -p "${MCQ_OUT}"
.venv/bin/python experiments/pretrain/scripts/eval_mcq_loglikelihood.py \
  --data_path "${MCQ_SUITE}" \
  --weight_path "\${WEIGHT}" \
  --output "${MCQ_OUT}/\${RUN_NAME}.json" \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --num_attention_heads 8 \
  --num_key_value_heads 8 \
  --intermediate_size 3072 \
  --max_seq_len 512 \
  --dtype bfloat16
.venv/bin/python experiments/pretrain/scripts/score_fixed_prompt_samples.py "experiments/pretrain/runs/\${RUN_NAME}" || true
EOF
)
  tmux new-session -d -s "${session}" bash -lc "${cmd}"
  echo "started ${session}: ${run_name}"
}

launch_run \
  "minimind_ext80" \
  "pretrain_stage10_ext80_from_quality_s512_lr8e7" \
  "experiments/pretrain/flywheel_sources/mix_quality80_external_english10_math10.json"

launch_run \
  "minimind_ext70" \
  "pretrain_stage10_ext70_from_quality_s512_lr8e7" \
  "experiments/pretrain/flywheel_sources/mix_quality70_external_english15_math15.json"

launch_run \
  "minimind_ext65_code" \
  "pretrain_stage10_ext65_code_from_quality_s512_lr8e7" \
  "experiments/pretrain/flywheel_sources/mix_quality65_external_english15_math15_code5.json"

tmux list-sessions
