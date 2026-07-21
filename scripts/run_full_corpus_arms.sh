#!/usr/bin/env bash
# Build the full-corpus training view and run two fixed-budget seeds on two GPUs.
#
# The budget is deliberately identical to the arm that already passed the gate
# on the mini corpus, so the only variable is data uniqueness: 2.27 repeated
# epochs of 1.24 GB against 0.34 of one epoch of 8.28 GB. The frozen strict
# holdout is inherited unchanged, so both arms are directly comparable and the
# gate thresholds still apply.
#
# Runs unattended: every stage logs, and a failed stage stops the chain rather
# than training on a view that was never audited.

set -euo pipefail

REPO="${REPO:-/root/mindsurf-workspace/repo}"
CORPUS="${CORPUS:-/root/mindsurf-workspace/data-staging/pretrain_t2t_full.jsonl}"
RAW_ROOT="${RAW_ROOT:-$REPO/data/raw/minimind_official_v1}"
STEPS="${STEPS:-60000}"
PY="$REPO/.venv/bin/python"
LOGS="$REPO/logs"

cd "$REPO"
mkdir -p "$LOGS" "$RAW_ROOT"

SPLITS="$RAW_ROOT/strict_splits"
TRAIN_FILE="$SPLITS/pretrain_strict_train_full.jsonl"
META="$SPLITS/strict_splits_meta_full.json"

echo "=== stage 1: holdout-disjoint train file ==="
# Must precede the audit: it asserts train_holdout_disjoint, and the raw full
# corpus contains the frozen holdout rows.
"$PY" scripts/prepare_full_corpus_splits.py \
  --corpus "$CORPUS" \
  --validation "$SPLITS/pretrain_strict_val_2k.jsonl" \
  --test "$SPLITS/pretrain_strict_test_2k.jsonl" \
  --output "$TRAIN_FILE" \
  --metadata "$META"

echo "=== stage 2: dataset spec ==="
"$PY" scripts/prepare_full_corpus_spec.py \
  --corpus "$TRAIN_FILE" \
  --root "$RAW_ROOT" \
  --output configs/datasets/minimind_official_full_v1.json

echo "=== stage 3: source audit ==="
"$PY" scripts/audit_pretrain_dataset.py \
  --spec configs/datasets/minimind_official_full_v1.json \
  --root "$RAW_ROOT" \
  --metadata "$META" \
  --output artifacts/data/minimind_official_full_v1/audit.json

echo "=== stage 4: training view ==="
"$PY" scripts/build_training_view.py \
  --spec configs/datasets/minimind_official_full_v1.json \
  --root "$RAW_ROOT" \
  --output data/processed/minimind_official_full_v1/pretrain_train_nfkc_dedup.jsonl \
  --manifest artifacts/data/minimind_official_full_v1/training_view.json \
  --audit artifacts/data/minimind_official_full_v1/audit.json

echo "=== stage 5: two seeds, one per GPU ==="
for pair in "20260511:0" "20260721:1"; do
  seed="${pair%%:*}"
  gpu="${pair##*:}"
  setsid nohup "$PY" scripts/run_formal_pipeline.py \
    --dataset minimind_official_full_v1 \
    --run-suffix "_full${STEPS}_seed${seed}" \
    --device "cuda:${gpu}" \
    --override "seed=${seed}" \
    --override "training.max_steps=${STEPS}" \
    --override "data.epochs=1" \
    --override "training.save_every=2000" \
    --override "training.eval_every=2000" \
    --override "training.logging_every=50" \
    > "$LOGS/full${STEPS}_seed${seed}.log" 2>&1 < /dev/null &
  echo "launched seed ${seed} on cuda:${gpu}"
  sleep 20
done

echo "=== launched; watch $LOGS ==="
