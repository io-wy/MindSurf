# MindSurf Pretrain

该分支提供 MiniMind 官方数据与 MindSurf 团队数据的可复现 80M 语言模型预训练
链路。数据、tokenizer、训练步数、评测阈值和 checkpoint 进度均有不可变身份；
训练入口采用单进程单卡，不引入该规模不需要的分布式系统。

## 数据与边界

仓库保留两套并列数据源，默认仍是此前实验使用的 MiniMind 官方
`pretrain_t2t_mini.jsonl`；团队数据是新增实验臂，不冒充官方数据：

| 数据臂 | 身份 | Strict train | Validation / test | 许可状态 |
| --- | --- | ---: | ---: | --- |
| Official | `gongjy/minimind_dataset@74aad49` | 1,265,983 | 2,000 / 2,000 | `CC-BY-NC-4.0` |
| Team | `wyywnab/mindsurf_pretrain_dataset@ab97cc8` | 2,177,856 | 2,000 / 2,000 | `other`，来源许可待澄清 |

两臂固定同一 MiniMind tokenizer、seed、模型、优化器、seen tokens 和评测套件。
每个候选都在 official 与 team 两套 strict holdout 上交叉评测，避免把“更贴合本域”
误判为通用提升。详细身份和发布措辞见
[数据集对照说明](docs/data/pretraining-dataset-comparison.md)与
[团队数据集说明](docs/data/mindsurf-team-dataset-v1.md)。

## 冻结候选

| 配置项 | 数值 |
| --- | ---: |
| Hidden size / layers | 768 / 8 |
| Attention heads / KV heads | 8 / 8 |
| FFN size | 3,584 |
| Vocabulary | 6,400 |
| 参数量 | 89,864,448 |
| Sequence length / batch size | 384 / 32 |
| Optimizer steps | 10,000 |
| Seen tokens | 122,880,000 |
| Learning rate | `5e-4` |
| Schedule | warmup 200 + stable 80% + cosine decay |

模型使用 QK RMSNorm、RoPE、MHA 和 SwiGLU。两套源训练 split 都经同一 NFKC
规范化去重。团队臂移除 11 条等价重复记录，形成 2,177,845 行、SHA-256 为
`37800ee01d5294e3d40765ec686fce7d0c917d7cea79343e57526a3229e8a999`
的训练视图。数据按 JSONL 流式读取并跨文档打包；checkpoint 原子保存模型、
optimizer、scheduler、AMP scaler、随机数状态和绝对 packed-block 游标，可在
optimizer 边界精确恢复，并只保留最近两个恢复点控制磁盘占用。

## 正式对照结果

两臂均已完成 10,000 optimizer steps 和 122,880,000 seen tokens。下表每列使用
同一套 holdout，因此可以比较两个训练候选；不同列之间的 loss 不直接比较。

| 训练候选 | Official val / test | Team val / test | MCQ | 固定提示词 | 高重复 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Official-only | **2.384 / 2.361** | 3.738 / 3.765 | **20 / 48** | **0.240** | **6** |
| Team-only | 3.073 / 3.098 | **2.941 / 2.974** | 15 / 48 | 0.161 | 10 |

结果显示明显的同域优势：Official-only 在 Official holdout 上更低，Team-only 在
Team holdout 上更低。在固定能力套件上 Official-only 也更强，但两者都未通过完整
能力门；Team 数据的逐来源许可尚未澄清，两者的公开许可门也均未通过。因此没有登记
内部 candidate，也没有公开发布权重。完整指标、失败门项和数据身份见
[实验报告](docs/experiments/2026-07-18-pretraining-dataset-ablation-80m.md)与
[机器可读结果](docs/experiments/2026-07-18-pretraining-dataset-ablation-80m.json)。

## 复现

```bash
uv sync --extra dev --frozen
uv run dvc repro build_official_training_view
uv run dvc repro build_team_training_view
uv run python scripts/preflight_training.py --require-cuda
uv run python scripts/run_dataset_ablation.py
```

DVC 描述相同的 fetch → audit → train → evaluate 依赖图：

```bash
uv run dvc dag
```

完整评测同时检查两套 strict validation/test loss、六类诊断域、48 道本地 MCQ、
固定提示词启发式分数和生成重复度。只有内部能力门全部通过时才登记 candidate；
只有能力门和对应数据许可门同时通过时才允许公开发布。任何发布必须明确写出使用
的是 Official、Team 还是混合数据；本对照实验本身不训练混合臂。

## 最新预训练与训练 Infra 进展

阶段正确的 192 题预训练门、统一数据索引、共享 GPU 容量准入、训练遥测和精确恢复
演练已经落地。基于 Official 父模型的 targeted-only、低学习率 targeted-only 和
80:20 Official/targeted replay 三个 1,000-step pilot 均未达到预先冻结的扩大条件，
因此没有启动正式训练或登记新候选。完整结果、统计检验、吞吐和恢复证据见
[2026-07-20 预训练 pilot 与训练恢复报告](docs/experiments/2026-07-20-pretrain-pilots-and-training-recovery.md)。

构建审计后的 80:20 replay 训练视图：

```bash
uv run dvc repro build_official_targeted_replay_training_view
```

该 replay 视图继承 Team 数据许可限制，不可作为公开发布数据。正式 checkpoint
远端副本、干净环境重建、主机重启演练以及 loss 突升/无进展告警闭环仍未完成，
所以预训练与训练 Infra 的状态仍是进行中。

## 推理与服务

本地 CLI 从 checkpoint 内嵌配置重建模型：

```bash
uv run python scripts/inference.py \
  --checkpoint models/checkpoints/minimind_official_v1_80m/final_model.pt \
  --tokenizer data/raw/minimind_official_v1/tokenizer \
  --prompt "你好，请介绍一下机器学习。"
```

API 通过 `INFERENCE_CHECKPOINT` 和 `INFERENCE_TOKENIZER` 加载同一模型。未配置或
加载失败时 `/inference` 明确返回 `503`，不会返回模拟结果。长训练任务由 Celery
排队，worker 并发固定为 1，避免同机出现多个训练进程。
