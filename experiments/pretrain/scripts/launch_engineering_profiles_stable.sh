#!/usr/bin/env bash
set -euo pipefail

cd /home/oscar/minimind

OUT_DIR="experiments/pretrain/diagnostics/engineering_profiles"
mkdir -p "${OUT_DIR}"

profile_one() {
  local run_name="$1"
  local weight_path="$2"
  local include_no_cache="$3"
  local no_cache_flag=()
  if [[ "${include_no_cache}" == "yes" ]]; then
    no_cache_flag=(--include_no_cache)
  fi
  if [[ -f "${OUT_DIR}/${run_name}.json" ]]; then
    echo "profile exists, skip: ${run_name}"
    return 0
  fi
  .venv/bin/python experiments/pretrain/scripts/profile_inference_engineering.py \
    --run_name "${run_name}" \
    --weight_path "${weight_path}" \
    --output_json "${OUT_DIR}/${run_name}.json" \
    --output_csv "${OUT_DIR}/${run_name}.csv" \
    --hidden_size 768 \
    --num_hidden_layers 8 \
    --num_attention_heads 8 \
    --num_key_value_heads 8 \
    --intermediate_size 3072 \
    --batch_sizes 1 4 \
    --prompt_tokens 128 512 1024 \
    --max_new_tokens 64 \
    --warmup 2 \
    --repeats 3 \
    --dtype bfloat16 \
    "${no_cache_flag[@]}"
}

profile_one \
  "quality100_s512_native_profile_stable" \
  "experiments/pretrain/platform_runs/pretrain_q100_stage2_quality_s512_b16_lr5e6/01_continue/pretrain_q100_stage2_quality_s512_b16_lr5e6_continue_768.pth" \
  "yes"

profile_one \
  "stage11_ext70_native_profile_stable" \
  "experiments/pretrain/platform_runs/pretrain_stage11_ext70_same_from_ext70_s512_lr5e7/01_continue/pretrain_stage11_ext70_same_from_ext70_s512_lr5e7_continue_768.pth" \
  "no"

profile_one \
  "stage12_ext80_replay_native_profile_stable" \
  "experiments/pretrain/platform_runs/pretrain_stage12_ext80_replay_from_stage11_s512_lr3e7/01_continue/pretrain_stage12_ext80_replay_from_stage11_s512_lr3e7_continue_768.pth" \
  "no"

echo "wrote stable engineering profiles to ${OUT_DIR}"
