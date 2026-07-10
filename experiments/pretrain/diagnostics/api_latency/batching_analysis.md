# API batching analysis

Model: `stage12_ext80_replay`, checkpoint `pretrain_stage12_ext80_replay_from_stage11_s512_lr3e7_continue_768.pth`.

Control:

- no batch: `scripts/serve_openai_api.py`, no dynamic batching
- dynamic batch: `--dynamic_batching --batch_max_size 4 --batch_wait_ms 8`
- request shape: `prompt_chars=128/512/1024`, `max_tokens=64`, no streaming, `temperature=0`, `top_p=1.0`
- concurrency: `1/2/4/8`, `requests_per_case=3`

| prompt chars | concurrency | no batch median total ms | dynamic batch median total ms | speedup |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1 | 228.9 | 237.2 | 0.97x |
| 128 | 2 | 459.2 | 246.0 | 1.87x |
| 128 | 4 | 911.4 | 235.5 | 3.87x |
| 128 | 8 | 1750.6 | 468.6 | 3.74x |
| 512 | 1 | 225.5 | 234.4 | 0.96x |
| 512 | 2 | 448.0 | 242.7 | 1.85x |
| 512 | 4 | 905.5 | 238.4 | 3.80x |
| 512 | 8 | 1764.7 | 474.3 | 3.72x |
| 1024 | 1 | 226.0 | 230.6 | 0.98x |
| 1024 | 2 | 449.0 | 243.0 | 1.85x |
| 1024 | 4 | 897.1 | 236.3 | 3.80x |
| 1024 | 8 | 1796.2 | 468.3 | 3.84x |

Decision: keep dynamic batching enabled for shared service and benchmark paths. The 8 ms wait slightly hurts single-request latency but pays back heavily once concurrency is above 1. This is a higher-value serving optimization than hand-writing KV store kernels for the current profile.

The earlier stochastic run (`temperature=0.7`, `top_p=0.92`) showed the same pattern, but the greedy run is the canonical number because output length is stable.

## Mixed prompt lengths

The first batching implementation grouped by exact input token count. That made repeated-prompt benchmarks look good, but real traffic with different prompt lengths would batch less often.

Fix: group dynamic batches by generation controls only: `temperature`, `top_p`, and `max_tokens`. Tokenizer padding already supports mixed prompt lengths.

Smoke:

```bash
.venv/bin/python experiments/pretrain/scripts/check_dynamic_batch_mixed.py \
  --base_url http://127.0.0.1:19022 \
  --min_batch_size 2
```

Observed stats:

```json
{
  "batches": 1,
  "jobs": 4,
  "max_batch_size": 4,
  "last_batch_size": 4,
  "last_input_token_counts": [32, 67, 107, 187]
}
```
