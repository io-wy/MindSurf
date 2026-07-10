#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
CODEX="$ROOT/experiments/pretrain"
LOGDIR="$CODEX/logs"
STATUS_DIR="$CODEX/status"
RUNTIME_STATUS="$STATUS_DIR/repro_status.txt"
STATE="$STATUS_DIR/overnight_repro_state.tsv"
GUARD="$CODEX/scripts/gpu_guard_any_training.sh"

mkdir -p "$LOGDIR" "$STATUS_DIR" "$CODEX/smoke_data" "$CODEX/smoke_out"

log() {
  printf '%s\t%s\n' "$(date -Iseconds)" "$*" | tee -a "$STATE"
}

gpu_snapshot() {
  nvidia-smi --query-gpu=memory.used,memory.total,temperature.gpu,utilization.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true
}

write_report() {
  local stage="$1"
  local latest_sft
  local gpu
  latest_sft="$(grep 'Epoch:' "$LOGDIR/full_sft_current.log" "$LOGDIR/full_sft_resume_auto.log" 2>/dev/null | tail -n 1 || true)"
  gpu="$(gpu_snapshot)"
  {
    echo "# MiniMind Base Reproduction Status"
    echo
    echo "- Updated: $(date -Iseconds)"
    echo "- Stage: $stage"
    echo "- Codex automation: disabled; this is a local WSL runner only."
    echo "- GPU snapshot: ${gpu:-unavailable}"
    echo "- Pretrain: completed, out/pretrain_768.pth exists."
    echo "- Full SFT latest: ${latest_sft:-not available}"
    echo
    echo "## Current Artifacts"
    find "$ROOT/out" "$ROOT/checkpoints" "$CODEX/smoke_out" -maxdepth 1 -type f \
      \( -name 'pretrain_768.pth' -o -name 'full_sft_768.pth' -o -name 'full_sft_768_resume.pth' -o -name 'lora_medical_repro1_768.pth' -o -name 'dpo_smoke_768.pth' \) \
      -printf '- %p (%s bytes, %TY-%Tm-%Td %TH:%TM)\n' 2>/dev/null | sort
    echo
    echo "## Logs"
    find "$LOGDIR" -maxdepth 1 -type f \
      \( -name 'full_sft*.log' -o -name 'eval_*_auto.log' -o -name 'train_lora_*_auto.log' -o -name 'train_dpo_*_auto.log' -o -name 'gpu_guard_any_training.log' \) \
      -printf '- %p (%s bytes, %TY-%Tm-%Td %TH:%TM)\n' 2>/dev/null | sort
    echo
    echo "## Runner Notes"
    tail -n 40 "$STATE" 2>/dev/null || true
  } > "$RUNTIME_STATUS"
}

start_guard() {
  if pgrep -af 'gpu_guard_any_training.sh' >/dev/null; then
    return 0
  fi
  nohup bash "$GUARD" >> "$LOGDIR/gpu_guard_launcher.log" 2>&1 &
  log "guard_started pid=$!"
}

full_sft_done() {
  [ -s "$ROOT/out/full_sft_768.pth" ] && \
    grep -q 'Epoch:\[2/2\](56608/56608)' "$LOGDIR/full_sft_current.log" "$LOGDIR/full_sft_resume_auto.log" 2>/dev/null
}

wait_for_existing_full_sft() {
  log "wait_full_sft_begin"
  while pgrep -af '[p]ython train_full_sft.py' >/dev/null; do
    local latest gpu
    latest="$(grep 'Epoch:' "$LOGDIR/full_sft_current.log" 2>/dev/null | tail -n 1 || true)"
    gpu="$(gpu_snapshot)"
    log "full_sft_running gpu=${gpu:-na} latest=${latest:-na}"
    write_report "Full SFT running"
    sleep 300
  done
  log "wait_full_sft_process_gone"
}

resume_full_sft_once_if_needed() {
  if full_sft_done; then
    log "full_sft_completed_no_resume_needed"
    return 0
  fi

  log "full_sft_not_complete_attempt_resume_after_cooldown"
  sleep 300
  start_guard
  (
    cd "$ROOT/trainer" || exit 2
    PYTHONUNBUFFERED=1 "$PY" train_full_sft.py --from_resume 1 --num_workers 0 --log_interval 50 --save_interval 1000
  ) > "$LOGDIR/full_sft_resume_auto.log" 2>&1
  local code=$?
  log "full_sft_resume_exit code=$code"
  write_report "Full SFT resume finished"
  return "$code"
}

run_eval_full_sft() {
  if [ -f "$LOGDIR/eval_full_sft_auto.done" ]; then
    log "skip_eval_full_sft_already_done"
    return 0
  fi
  log "eval_full_sft_begin"
  (
    cd "$ROOT" || exit 2
    printf '0\n' | "$PY" eval_llm.py --weight full_sft --max_new_tokens 128 --temperature 0.7 --top_p 0.9
  ) > "$LOGDIR/eval_full_sft_auto.log" 2>&1
  local code=$?
  log "eval_full_sft_exit code=$code"
  [ "$code" -eq 0 ] && touch "$LOGDIR/eval_full_sft_auto.done"
  write_report "Full SFT eval finished"
  return "$code"
}

run_lora_medical_one_epoch() {
  if [ -s "$ROOT/out/lora_medical_repro1_768.pth" ]; then
    log "skip_lora_medical_existing"
    return 0
  fi
  log "lora_medical_begin"
  start_guard
  (
    cd "$ROOT/trainer" || exit 2
    PYTHONUNBUFFERED=1 "$PY" train_lora.py \
      --lora_name lora_medical_repro1 \
      --epochs 1 \
      --batch_size 16 \
      --accumulation_steps 1 \
      --max_seq_len 340 \
      --num_workers 0 \
      --log_interval 20 \
      --save_interval 200 \
      --data_path ../dataset/lora_medical.jsonl \
      --from_weight full_sft
  ) > "$LOGDIR/train_lora_medical_repro1_auto.log" 2>&1
  local code=$?
  log "lora_medical_exit code=$code"
  write_report "LoRA medical finished"
  return "$code"
}

run_eval_lora() {
  if [ ! -s "$ROOT/out/lora_medical_repro1_768.pth" ]; then
    log "skip_eval_lora_missing_weight"
    return 0
  fi
  log "eval_lora_begin"
  (
    cd "$ROOT" || exit 2
    printf '0\n' | "$PY" eval_llm.py --weight full_sft --lora_weight lora_medical_repro1 --max_new_tokens 96 --temperature 0.7 --top_p 0.9
  ) > "$LOGDIR/eval_lora_medical_repro1_auto.log" 2>&1
  local code=$?
  log "eval_lora_exit code=$code"
  write_report "LoRA eval finished"
  return "$code"
}

run_dpo_smoke() {
  if [ -s "$CODEX/smoke_out/dpo_smoke_768.pth" ]; then
    log "skip_dpo_smoke_existing"
    return 0
  fi
  log "dpo_smoke_begin"
  head -n 64 "$ROOT/dataset/dpo.jsonl" > "$CODEX/smoke_data/dpo_64.jsonl"
  start_guard
  (
    cd "$ROOT/trainer" || exit 2
    PYTHONUNBUFFERED=1 "$PY" train_dpo.py \
      --data_path ../experiments/pretrain/smoke_data/dpo_64.jsonl \
      --save_dir ../experiments/pretrain/smoke_out \
      --save_weight dpo_smoke \
      --epochs 1 \
      --batch_size 1 \
      --accumulation_steps 1 \
      --max_seq_len 512 \
      --num_workers 0 \
      --log_interval 1 \
      --save_interval 100 \
      --from_weight full_sft
  ) > "$LOGDIR/train_dpo_smoke_auto.log" 2>&1
  local code=$?
  log "dpo_smoke_exit code=$code"
  write_report "DPO smoke finished"
  return "$code"
}

main() {
  cd "$ROOT" || exit 2
  log "runner_start"
  write_report "Runner started"
  wait_for_existing_full_sft
  resume_full_sft_once_if_needed || true

  if ! full_sft_done; then
    log "stop_full_sft_unfinished"
    write_report "Stopped: Full SFT unfinished"
    exit 20
  fi

  run_eval_full_sft || true
  run_lora_medical_one_epoch || true
  run_eval_lora || true
  run_dpo_smoke || true
  write_report "Runner finished"
  log "runner_finished"
}

main "$@"
