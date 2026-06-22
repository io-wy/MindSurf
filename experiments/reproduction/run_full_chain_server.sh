#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
LOGDIR="$ROOT/experiments/reproduction/logs"
STATUS_DIR="$ROOT/experiments/reproduction/status"
STATE="$STATUS_DIR/full_chain_state.tsv"
REPORT="$STATUS_DIR/full_chain_status.md"
NUM_WORKERS="${NUM_WORKERS:-8}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-128}"

mkdir -p "$LOGDIR" "$STATUS_DIR" "$ROOT/out" "$ROOT/checkpoints"

log() {
  printf '%s\t%s\n' "$(date -Iseconds)" "$*" | tee -a "$STATE"
}

gpu_snapshot() {
  nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu,utilization.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true
}

latest_line() {
  local file="$1"
  local pattern="${2:-Epoch:}"
  grep "$pattern" "$file" 2>/dev/null | tail -n 1 || true
}

write_report() {
  local stage="$1"
  {
    echo "# MiniMind Full Chain Reproduction"
    echo
    echo "- Updated: $(date -Iseconds)"
    echo "- Stage: $stage"
    echo "- User: $(id -un)"
    echo "- Root: $ROOT"
    echo "- Python: $PY"
    echo "- GPU: $(gpu_snapshot)"
    echo
    echo "## Stage Markers"
    find "$STATUS_DIR" -maxdepth 1 -type f -name '*.done' -printf '- %f\n' 2>/dev/null | sort || true
    echo
    echo "## Latest Training Lines"
    echo "- Pretrain: $(latest_line "$LOGDIR/01_pretrain.log")"
    echo "- Full SFT: $(latest_line "$LOGDIR/02_full_sft.log")"
    echo "- LoRA: $(latest_line "$LOGDIR/04_lora_medical.log")"
    echo "- DPO: $(latest_line "$LOGDIR/06_dpo.log")"
    echo
    echo "## Artifacts"
    find "$ROOT/out" "$ROOT/checkpoints" -maxdepth 1 -type f \
      \( -name '*.pth' -o -name '*.pt' -o -name '*.safetensors' \) \
      -printf '- %p (%s bytes, %TY-%Tm-%Td %TH:%TM)\n' 2>/dev/null | sort || true
    echo
    echo "## Recent State"
    tail -n 60 "$STATE" 2>/dev/null || true
  } > "$REPORT"
}

require_server_context() {
  if [ "$(id -un)" != "oscar" ]; then
    echo "This runner must be executed as the personal Linux user oscar." >&2
    exit 2
  fi
  case "$ROOT" in
    /home/oscar/*) ;;
    *)
      echo "Refusing to run outside /home/oscar: $ROOT" >&2
      exit 2
      ;;
  esac
  if [ ! -x "$PY" ]; then
    echo "Python venv not found or not executable: $PY" >&2
    exit 2
  fi
}

run_stage() {
  local id="$1"
  local title="$2"
  local cmd="$3"
  local marker="$STATUS_DIR/${id}.done"
  local logfile="$LOGDIR/${id}.log"

  if [ -f "$marker" ]; then
    log "skip $id $title"
    write_report "Skipped $title"
    return 0
  fi

  log "begin $id $title gpu=$(gpu_snapshot)"
  write_report "Running $title"
  set +e
  (cd "$ROOT" && bash -lc "$cmd") > "$logfile" 2>&1
  local code=$?
  set -e
  log "end $id $title code=$code gpu=$(gpu_snapshot)"
  if [ "$code" -ne 0 ]; then
    write_report "Failed $title"
    echo "Stage failed: $title. See $logfile" >&2
    exit "$code"
  fi
  touch "$marker"
  write_report "Completed $title"
}

main() {
  require_server_context
  log "runner_start root=$ROOT"
  write_report "Preflight"

  run_stage "00_env_check" "environment check" \
    "\"$PY\" - <<'PY'
import sys, torch, transformers, datasets
print('python', sys.version)
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
print('cuda_device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
print('transformers', transformers.__version__)
print('datasets', datasets.__version__)
PY"

  run_stage "01_pretrain" "official pretrain 2 epochs" \
    "cd trainer && PYTHONUNBUFFERED=1 \"$PY\" train_pretrain.py --num_workers \"$NUM_WORKERS\" --log_interval \"$LOG_INTERVAL\" --save_interval \"$SAVE_INTERVAL\""

  run_stage "02_full_sft" "official full SFT 2 epochs" \
    "cd trainer && PYTHONUNBUFFERED=1 \"$PY\" train_full_sft.py --num_workers \"$NUM_WORKERS\" --log_interval \"$LOG_INTERVAL\" --save_interval \"$SAVE_INTERVAL\""

  run_stage "03_eval_full_sft" "full SFT inference check" \
    "printf '0\n' | \"$PY\" eval_llm.py --weight full_sft --max_new_tokens \"$EVAL_MAX_NEW_TOKENS\" --temperature 0.7 --top_p 0.9"

  run_stage "04_lora_medical" "official LoRA medical training" \
    "cd trainer && PYTHONUNBUFFERED=1 \"$PY\" train_lora.py --num_workers \"$NUM_WORKERS\" --log_interval 50 --save_interval \"$SAVE_INTERVAL\""

  run_stage "05_eval_lora_medical" "LoRA inference check" \
    "printf '0\n' | \"$PY\" eval_llm.py --weight full_sft --lora_weight lora_medical --max_new_tokens \"$EVAL_MAX_NEW_TOKENS\" --temperature 0.7 --top_p 0.9"

  run_stage "06_dpo" "official DPO full-data training" \
    "cd trainer && PYTHONUNBUFFERED=1 \"$PY\" train_dpo.py --num_workers \"$NUM_WORKERS\" --log_interval 50 --save_interval \"$SAVE_INTERVAL\""

  log "runner_finished"
  write_report "Finished"
}

main "$@"
