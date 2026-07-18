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
