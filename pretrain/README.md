# MiniMind 预训练复现与优化

目前可复现的主线是：

```text
strict split
-> MHA + FFN3072 从头 WSD 训练
-> quality continuation
-> 少量外部干净英文/数学数据
-> quality80 replay
-> strict loss + MCQ + fixed prompts + 服务吞吐评估
```

## 当前最好结果

按已完成的固定评估，当前基座候选是 `stage12_ext80_replay`。但是，该候选的
checkpoint 目前在已审计的服务器与本地资产集中均缺失；以下指标是已有评估记录，
不能据此认为权重当前可恢复或可重新运行评估。

| 项 | 值 |
| --- | --- |
| 结构 | `hidden=768, layers=8, heads=8, kv_heads=8, intermediate_size=3072` |
| 注意力 | MHA |
| 参数量 | 约 `80.43M` |
| 权重 | 已评估；原记录路径为 `experiments/pretrain/platform_runs/pretrain_stage12_ext80_replay_from_stage11_s512_lr3e7/01_continue/pretrain_stage12_ext80_replay_from_stage11_s512_lr3e7_continue_768.pth`，但 checkpoint 当前缺失 |
| 评估包 | `experiments/pretrain/runs/stage12_ext80_replay_value_bundle/` |

固定评估结果：

| 指标 | 结果 |
| --- | ---: |
| strict val loss, `eval_seq_len=1024` | `2.31377` |
| strict test loss, `eval_seq_len=1024` | `2.33291` |
| MCQ v1 | `23/48` |
| fixed prompt score | `0.4575` |

这个结果只能说明它是当前已评估的最好基座候选，不能说明 checkpoint 当前可用。
MCQ 和固定 prompt 里还有数学差、校准推理差、重复输出的问题。

## 结构实验

固定 `hidden=768, layers=8, heads=8, seq384, WSD, 10000 steps` 后，对比 MHA、GQA 和 FFN 宽度。

| 对照 | 参数量 | strict val loss | strict test loss | 结论 |
| --- | ---: | ---: | ---: | --- |
| GQA, `kv_heads=4, FFN=2432` | 约 `63.91M` | `2.4356` | `2.4489` | 原默认线 |
| MHA, `kv_heads=8, FFN=2432` | 约 `68.63M` | `2.4235` | `2.4386` | MHA 更好 |
| GQA, `kv_heads=4, FFN=3072` | 约 `75.71M` | `2.4263` | `2.4405` | 加宽 FFN 有收益，但不够 |
| GQA, `kv_heads=4, FFN=3328` | 约 `80.43M` | `2.4121` | `2.4263` | 同参数量仍弱于 MHA |
| MHA, `kv_heads=8, FFN=3072` | 约 `80.43M` | `2.4022` | `2.4177` | 当前结构主线 |

结论：先用MHA。GQA/MQA 的价值主要在推理省显存和提速，

## 数据实验

做过三类有效尝试。

| 阶段 | 做法 | 结果 |
| --- | --- | --- |
| quality continuation | 从 strict train 里过滤低质量文本，再续训 | strict loss 继续下降 |
| 外部干净数据 | 加少量英文、数学、TinyStories 风格数据 | MCQ 从早期候选 `20/48` 到 `23/48` |
| quality80 replay | 外部数据后回到质量过滤主线 | fixed prompt 更稳 |

做过但不继续的方向：

- 低学习率 continuation：有时能降 loss，但收益小，容易变成反复磨同一个 checkpoint。
- checkpoint averaging：能小幅平滑，不解决能力短板。
- 手写坏例回流：能修个别答案，但 strict val/test 变差。

后面如果继续做预训练优化，优先做数据质量和评估集，不优先继续堆低学习率续训。

## 评估

> 注意：下面的命令记录了当时的统一评估入口。由于 stage12 checkpoint 当前缺失，
> 在恢复并校验对应权重前不能直接复现该次评估。

统一入口：

```bash
.venv/bin/python experiments/pretrain/scripts/evaluate_checkpoint_bundle.py \
  --run_name stage12_ext80_replay_value_bundle \
  --weight_path experiments/pretrain/platform_runs/pretrain_stage12_ext80_replay_from_stage11_s512_lr3e7/01_continue/pretrain_stage12_ext80_replay_from_stage11_s512_lr3e7_continue_768.pth \
  --source_run_dir experiments/pretrain/platform_runs/pretrain_stage12_ext80_replay_from_stage11_s512_lr3e7 \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --num_attention_heads 8 \
  --num_key_value_heads 8 \
  --intermediate_size 3072 \
  --eval_seq_len 1024 \
  --mcq_seq_len 512 \
  --max_input_tokens 384 \
  --max_new_tokens 96 \
  --prompt_format pretrain \
  --overwrite
```

输出位置：

| 文件 | 内容 |
| --- | --- |
| `report.md` | 单个 checkpoint 的摘要 |
| `manifest.json` | 权重、参数、命令、评估产物 |
| `evals/` | strict val/test loss |
| `mcq/` | 选择题明细 |
| `samples/` | 固定 prompt 输出和简单打分 |

MCQ v1 分项：

| 类别 | 正确数 | 总数 | acc |
| --- | ---: | ---: | ---: |
| zh_fact | 8 | 8 | 1.000 |
| code | 4 | 8 | 0.500 |
| long_context | 4 | 8 | 0.500 |
| english | 3 | 8 | 0.375 |
| calibration_reasoning | 3 | 8 | 0.375 |
| math | 1 | 8 | 0.125 |

这里能看出当前短板：数学和校准推理最弱。下一轮评估应该补分领域 PPL 和更稳定的固定题集。

## 服务和推理

服务入口是 `scripts/serve_openai_api.py`。

当前可恢复的 infrastructure 基线是 `infra-baseline-gqa64m-20260712`。它是约
`63.91M` 参数的 GQA 模型，用于验证训练恢复、评估、HF 导出和双后端服务闭环；它不替代
上面记录的 stage12 算法最佳结果。发布规格与门禁见
`experiments/pretrain/releases/infra_baseline_gqa64m_20260712.json`。

支持内容：

- OpenAI-compatible `/v1/chat/completions`
- 非流式 dynamic batching
- `temperature=0` greedy 路径
- `/healthz`

启动当前默认服务：

```bash
bash experiments/pretrain/scripts/start_openai_service_tmux.sh
```

恢复服务前先校验 snapshot：

```bash
bash experiments/pretrain/scripts/restore_openai_service_tmux.sh
```

检查服务：

```bash
bash experiments/pretrain/scripts/status_openai_service_tmux.sh
bash experiments/pretrain/scripts/smoke_openai_service.sh
```

高并发使用经过门禁的 vLLM 后端：

```bash
bash experiments/pretrain/scripts/start_vllm_tmux.sh
bash experiments/pretrain/scripts/status_vllm_tmux.sh
bash experiments/pretrain/scripts/smoke_vllm_service.sh
```

单 GPU 环境恢复、训练观测、release promotion、指标和故障恢复命令见
`docs/infra/single_gpu_operations.md`。

## Dynamic batching

固定 `temperature=0, top_p=1.0, max_tokens=64, no-stream` 后，`batch_max_size=4, batch_wait_ms=8` 的结果如下。

| prompt chars | concurrency | no batch ms | dynamic batch ms | speedup |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1 | 228.9 | 237.2 | 0.97x |
| 128 | 2 | 459.2 | 246.0 | 1.87x |
| 128 | 4 | 911.4 | 235.5 | 3.87x |
| 128 | 8 | 1750.6 | 468.6 | 3.74x |
| 512 | 4 | 905.5 | 238.4 | 3.80x |
| 1024 | 8 | 1796.2 | 468.3 | 3.84x |

单请求会慢一点。并发大于 1 后收益明显。

真实流量里 prompt 长度不同，所以 batching key 不再按输入长度分组，只按 `temperature/top_p/max_tokens` 分组。不同长度交给 tokenizer padding。

混合长度 smoke：

```json
{
  "batches": 1,
  "jobs": 4,
  "max_batch_size": 4,
  "last_batch_size": 4,
  "last_input_token_counts": [32, 67, 107, 187]
}
```

## flash-attn 和 profiler

项目内有私有 `nvcc`：

```text
<project-root>/.tools/cuda-nvcc-12.9.86
```

官方 `flash_attn-2.8.3` wheel 已构建并安装到 `.venv`：

```text
experiments/pretrain/wheelhouse/flash_attn_official/flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl
sha256=b56e6c91c12ebafcfcd2061f38167e1bbe3ac6b116c63bfe65623b798fd3137a
```

`flash_attn_smoke.py` 已通过，`max_abs=0.000244140625`。

nano-vLLM 回归：

| 条件 | 结果 |
| --- | ---: |
| prompt `128/512/1024`, requests `4`, max tokens `64`, greedy | `969.8 / 958.8 / 927.6 tok/s` |
| torch profiler, prompt `128`, requests `4`, max tokens `64` | `350.1 tok/s` |

Profiler 看到的主要时间在 GEMM、FlashAttention 和小 kernel 调度上。`store_kvcache_kernel` 约 `1.9%` self CUDA。现在不优先手搓kernel，除非后续 profiler 证明它变成主瓶颈。

## 复现实验

严格数据切分：

```bash
.venv/bin/python experiments/pretrain/scripts/prepare_strict_splits.py --write-train
```

MHA + FFN3072 从头 WSD：

```bash
bash experiments/pretrain/scripts/run_ffn_width_variant.sh \
  --run_name ffn3072_mha_wsd_s384_10000 \
  --intermediate_size 3072 \
  --num_key_value_heads 8
```

WSD 后处理：

```bash
bash experiments/pretrain/scripts/run_ffn_post_wsd_variant.sh \
  ffn3072_cooldown_seq768 \
  experiments/pretrain/platform_runs/ffn3072_mha_wsd_s384_10000/01_wsd_s384/ffn3072_mha_wsd_s384_10000_wsd_s384_768.pth \
  3072 \
  8
```

## 文件地图

| 路径 | 用途 |
| --- | --- |
| `model/model_minimind.py` | MiniMind 结构，含 MHA/GQA/MQA 参数 |
| `experiments/pretrain/scripts/train_pretrain_optimized.py` | 预训练入口 |
| `experiments/pretrain/scripts/evaluate_checkpoint_bundle.py` | 统一评估入口 |
| `experiments/pretrain/scripts/benchmark_openai_api_latency.py` | OpenAI API latency benchmark |
| `experiments/pretrain/scripts/check_dynamic_batch_mixed.py` | 混合长度 batching smoke |
| `experiments/pretrain/diagnostics/api_latency/batching_analysis.md` | batching 对比结果 |
| `experiments/pretrain/diagnostics/nanovllm/trace_analysis.md` | nano-vLLM profiler 结论 |
| `experiments/pretrain/serving_snapshots/` | 服务 snapshot 和 append log |

## 下一步

1. 固定 prompt，先覆盖数学、代码、英文、长上下文、校准推理。
2. 补分领域 PPL，不再把所有文本混成一个 loss。
3. 继续做数据质量实验，少做低学习率反复续训。
