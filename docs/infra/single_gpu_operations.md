# MiniMind Single-GPU Infrastructure

This project uses a single-GPU production path sized for the promoted 64M-parameter model. The operational source of truth is the release specification under `experiments/pretrain/releases/`; model binaries, full training state, runtime logs, and virtual environments remain external assets.

## Design boundaries

- Use one GPU process at a time for training, native serving, or vLLM serving.
- Use synchronous atomic checkpoints. A full model checkpoint is small enough that distributed or asynchronous checkpoint coordination adds more cost than it removes.
- Use the native backend for low concurrency and vLLM for concurrency of eight or more. The threshold comes from the checked-in release benchmark gates.
- Keep JSONL metrics as the durable training record. Trackio is an optional local dashboard, not a training dependency.
- Do not add FSDP, ZeRO, Kubernetes, Ray, model quantization, or speculative decoding without a new measured bottleneck.

All commands below assume the current directory is `$PROJECT_ROOT`.

## Environment recovery

Restore the training environment from the main requirements, then install optional infrastructure extras:

```bash
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -r experiments/pretrain/requirements-infra.txt
.venv/bin/pip check
```

The infrastructure requirements pin one resolver-tested stack rather than
mixing old and new observability packages: Transformers 5.13.1, Hub 1.23.0,
Trackio 0.30.3, SwanLab 0.8.4, and W&B 0.28.0. Upgrade these pins together and
require `pip check` to pass; independently upgrading one of them can create
incompatible Hub, Rich, or Protobuf constraints.

Rebuild the isolated text-only vLLM environment with:

```bash
bash experiments/pretrain/scripts/bootstrap_vllm_env.sh
```

The bootstrap pins vLLM, requires its complete dependency set to pass
`pip check`, and records the resolved environment inside the ignored virtual
environment. Text-only serving does not exercise TorchCodec, but keeping the
declared dependency installed makes rebuilds reproducible. The launcher exposes
the compatible shared FFmpeg libraries already shipped by vLLM's
PyNvVideoCodec wheel, so no host-wide FFmpeg mutation is required.

## Training preflight and telemetry

Training requires an idle GPU and sufficient disk space:

```bash
.venv/bin/python experiments/pretrain/scripts/infra_preflight.py \
  --mode training \
  --root . \
  --min-free-disk-gb 20 \
  --max-gpu-memory-used-mb 512
```

Every optimized pretraining run writes `<run>_metrics.jsonl` and `<run>_run_manifest.json`. Enable the optional local Trackio mirror with:

```bash
export TRACKIO_DIR="$PROJECT_ROOT/experiments/pretrain/status/trackio"
.venv/bin/python experiments/pretrain/scripts/train_pretrain_optimized.py \
  ... \
  --trackio_project minimind-pretrain
```

Add `--trackio_space_id namespace/space` only when an approved remote dashboard is required. A missing Trackio installation fails before training starts when Trackio was explicitly requested; normal JSONL logging remains dependency-free.

## Release gate and promotion

Validate hashes, evaluation thresholds, export equivalence, benchmark volume, errors, and high-concurrency latency:

```bash
.venv/bin/python experiments/pretrain/scripts/release_gate.py \
  --spec experiments/pretrain/releases/infra_baseline_gqa64m_20260712.json \
  --root . \
  --output experiments/pretrain/diagnostics/release_gate/current.json
```

Promote only a passing release:

```bash
.venv/bin/python experiments/pretrain/scripts/promote_release.py \
  --spec experiments/pretrain/releases/infra_baseline_gqa64m_20260712.json \
  --root . \
  --output out/releases/current.json
```

After the service starts, rerun `release_gate.py` with `--check-service` to verify the live health, model-list, and metrics endpoints.

Each later promotion preserves the old pointer as `out/releases/previous.json`.
Rollback swaps the two pointers atomically per file; restart and smoke the
service after the swap:

```bash
.venv/bin/python experiments/pretrain/scripts/rollback_release.py
```

## vLLM high-concurrency service

```bash
bash experiments/pretrain/scripts/start_vllm_tmux.sh
bash experiments/pretrain/scripts/status_vllm_tmux.sh
bash experiments/pretrain/scripts/smoke_vllm_service.sh
bash experiments/pretrain/scripts/stop_vllm_tmux.sh
```

`serve_vllm.sh` resolves the promoted artifact from `out/releases/current.json`, uses 10% GPU memory, a 512-token context, and eager execution. These defaults keep the fixed cost proportional to the small model while retaining continuous batching and `/metrics`.

## Native low-concurrency service

```bash
bash experiments/pretrain/scripts/start_openai_service_tmux.sh
bash experiments/pretrain/scripts/status_openai_service_tmux.sh
bash experiments/pretrain/scripts/smoke_openai_service.sh
bash experiments/pretrain/scripts/stop_openai_service_tmux.sh
```

The native service exposes `/healthz`, `/metrics`, and `/v1/chat/completions`. Its Prometheus metrics cover request counts, errors, handler time, batch counts, batch jobs, maximum batch size, and queue depth.

## Recovery sequence

1. Stop the active service with its lifecycle script.
2. Rebuild the asset manifest and verify the restored file hashes.
3. Run `release_gate.py` without service checks.
4. Promote the verified release.
5. Start the selected backend and run its smoke script.
6. Run `release_gate.py --check-service`.

Never delete older checkpoints or runtime releases as part of promotion. Retention is a separate, report-first operation.
