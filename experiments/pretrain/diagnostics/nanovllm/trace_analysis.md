# nano-vLLM trace analysis

条件：`stage12_ext80_replay`，Qwen-compatible export，RTX 4090，PyTorch `2.11.0+cu128`，真实 `flash_attn 2.8.3`，vendored nano-vLLM。

## 结果

| 场景 | 控制参数 | 结果 |
| --- | --- | ---: |
| real flash-attn benchmark | prompt 128/512/1024 chars, requests=4, max_tokens=64, warmup=true | `727.6 / 975.3 / 970.6 tok/s` |
| prefill profile | prompt=1024 chars, requests=4, max_tokens=1 | elapsed `0.037s` |
| decode profile | prompt=128 chars, requests=4, max_tokens=128 | elapsed `1.390s`, `368.5 tok/s` under profiler |

## profiler 结论

| 项 | prefill | decode | 判断 |
| --- | ---: | ---: | --- |
| `aten::mm` / Cutlass GEMM | CUDA top，约 `66%` self CUDA | CUDA top，约 `62%` self CUDA | 主算力路径 |
| FlashAttention kernel | 约 `14%` self CUDA | 约 `11%` self CUDA | 已是真实 flash-attn，不是 fallback |
| `store_kvcache_kernel` | 约 `1.5%` self CUDA | 约 `1.9%` self CUDA | 不是当前第一瓶颈 |
| CPU compiled-region / launch / cache lookup | 明显存在 | 更明显 | decode 更像小 kernel 调度成本 + GEMM/attention 混合瓶颈 |

## 决策

不继续把 KV store 当主优化点。它已经从“不兼容 D=768”变成“能跑且占比很小”。

下一步最小优化应放在两处：

1. 减少 decode 每 token 的小 kernel / compiled region / launch 开销；
2. 复测更大 batch / concurrency，看瓶颈是否从 CPU 调度转向 GPU GEMM 或 FlashAttention。

## Nsight 复核

`nsys/ncu` 已用用户态方式安装到 `/home/oscar/minimind/.tools/nsight`。普通用户可以跑 `nsys` CUDA timeline；`ncu` 需要 sudo，否则会被 `ERR_NVGPUCTRPERM` 挡住。

`nsys` kernel 汇总里，decode-like greedy 路径的主要时间不是 KV store：

| kernel 类别 | GPU kernel time | 判断 |
| --- | ---: | --- |
| Cutlass/CUBLAS GEMM | 约 `31.2% + 7.8% + 7.3% + 6.4%` | 主算力路径 |
| FlashAttention split-kv / combine | 约 `8.0% + 3.6%` | 真实 flash-attn 路径 |
| Triton/PyTorch 小 kernel | 多个 `1%~5%` 项，实例数高 | launch/调度成本明显 |
| `store_kvcache_kernel` | `1.5%`，384 次 | 不是第一优化点 |

定向 `ncu --set basic` 结果：

| kernel | 平均 duration | DRAM throughput | SM throughput | achieved occupancy |
| --- | ---: | ---: | ---: | ---: |
| `store_kvcache_kernel` | `2.08us` | `9.99%` | `0.44%` | `7.58%` |
| `flash_fwd_splitkv_kernel` | `6.70us` | `3.72%` | `1.15%` | `8.33%` |

决策：不先手搓 KV store。下一步如果继续工程优化，先补 NVTX 分段，复测服务侧 batching/调度；只有 ncu 证明某个 kernel 成为主要瓶颈，再写 Triton/CUDA。

## NVTX 分段

vendored nano-vLLM 已加 NVTX 范围：`schedule`、`model_prefill/decode`、`prepare_prefill/decode`、`run_model_prefill/decode`、`sample`、`postprocess`。

`nsys` 的 `nvtx_gpu_proj_sum` 显示：

| NVTX range | instances | total GPU projected time | median range time | 判断 |
| --- | ---: | ---: | ---: | --- |
| `run_model_decode` | 46 | `589.3ms` | `4.79ms` | decode 主体 |
| `model_decode` | 46 | `628.9ms` | `4.97ms` | 包含 runner 调用 |
| `prepare_decode` | 46 | `4.51ms` | `0.094ms` | 不是主瓶颈 |
| `sample` | 49 | `1.52ms` GPU 投影，range 总时长 `72.6ms` | `0.023ms` GPU 投影 | greedy 后 GPU 侧很小 |
| `run_model_prefill` | 3 | `1862.0ms` | `613.0ms` | 包含初始化/预热影响 |

现在 profiler 语言已经能对应到代码阶段。继续优化应盯 `run_model_decode` 内部的 GEMM/FlashAttention/小 Triton kernel 组合，或回到服务侧 batching/调度复测。

## Greedy sampling fast path

After allowing `temperature=0` and routing all-greedy batches to `argmax`, decode-like profile changed from 368.5 tok/s to 493.3 tok/s under the same prompt=128, requests=4, max_tokens=128 condition. CUDA time is still dominated by GEMM/FlashAttention/small compiled kernels; the win mainly removes sampling softmax/random overhead and improves deterministic serving/eval.

This is now a real optimization, not only a diagnostic switch. Keep stochastic sampling for `temperature>0`; use `temperature=0` for reproducible eval and latency checks.

## Official flash-attn wheel

The project now has a private CUDA compile path under `/home/oscar/minimind/.tools/cuda-nvcc-12.9.86`. It provides `nvcc 12.9.86`, runtime dev/static libraries, driver stubs, and CCCL headers without changing system CUDA.

Validation:

| check | condition | result |
| --- | --- | --- |
| CUDA smoke | `g++ 13.3`, `sm_89`, one `add_one<<<1,1>>>` kernel | compile/link/run passed, `nvcc_smoke2_exit=0` |
| official source | PyPI sdist `flash_attn-2.8.3.tar.gz` | includes `csrc/cutlass`; GitHub tag zip does not |
| official wheel | `FLASH_ATTN_CUDA_ARCHS=89`, `MAX_JOBS=4`, `NVCC_THREADS=1` | built `flash_attn-2.8.3-cp312-cp312-linux_x86_64.whl` |
| wheel hash | built on MindSurf server | `b56e6c91c12ebafcfcd2061f38167e1bbe3ac6b116c63bfe65623b798fd3137a` |
| runtime smoke | `flash_attn_smoke.py`, batch=2, seq_len=128, heads=8, head_dim=96, fp16 | passed, `max_abs=0.000244140625` |

This replaces the community-wheel route for future reproducible runs. The earlier profiler conclusions still stand because both paths use real `flash_attn 2.8.3`; the difference is provenance and rebuildability, not model behavior.

## Official wheel regression

After installing the self-built official `flash_attn-2.8.3` wheel, nano-vLLM still runs on the Qwen-compatible `stage12_ext80_replay` export.

| scenario | control parameters | result |
| --- | --- | ---: |
| official wheel benchmark | prompt 128/512/1024 chars, requests=4, max_tokens=64, temperature=0, warmup=true | `969.8 / 958.8 / 927.6 tok/s` |
| previous real flash-attn warm benchmark | same shape | `727.6 / 975.3 / 970.6 tok/s` |
| official wheel profiler | prompt=128 chars, requests=4, max_tokens=64 | `350.1 tok/s` under profiler |

Profiler top line is unchanged in substance: `aten::mm`/Cutlass is the main CUDA path, FlashAttention is active, and `store_kvcache_kernel` is about `1.9%` self CUDA. The official wheel therefore closes provenance/rebuildability, not a new bottleneck. Keep KV store as solved-for-now; put the next engineering pass on batching/scheduling or decode small-kernel overhead.
