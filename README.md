# MindSurf Pretrain

该分支提供 MiniMind 官方数据与 MindSurf 团队数据的可复现 80M 语言模型预训练
链路。数据、tokenizer、训练步数、评测阈值和 checkpoint 进度均有不可变身份；
训练入口采用单进程单卡，不引入该规模不需要的分布式系统。

## 数据与边界

仓库保留两套并列数据源。团队数据是实验臂，不冒充官方数据：

| 数据臂 | 身份 | Validation / test | 许可状态 |
| --- | --- | ---: | --- |
| Official | `gongjy/minimind_dataset@74aad49` | 2,000 / 2,000 | `CC-BY-NC-4.0` |
| Team | `wyywnab/mindsurf_pretrain_dataset@ab97cc8` | 2,000 / 2,000 | `other`，来源许可待澄清 |

**已发布权重只用 Official 数据训练，无一行 Team 数据**，因此 Team 的许可待澄清
问题不传导到发布物。Team holdout 只用作**异源**评测，度量跨源泛化——同分布
holdout loss 只度量对自家语料的压缩，单独不构成质量声明。两臂固定同一 MiniMind
tokenizer、模型、优化器和评测套件。详细身份和发布措辞见
[数据集对照说明](docs/data/pretraining-dataset-comparison.md)与
[团队数据集说明](docs/data/mindsurf-team-dataset-v1.md)。

## 已发布候选

当前交付物是 **Official 数据单臂、168,000 步**的 89,864,448 参数基座，双 seed
各一份。发布说明、许可传导与全部限定见
[2026-07-22 权重发布说明](docs/releases/2026-07-22-pretrain-80m.md)。

| 配置项 | 数值 |
| --- | ---: |
| Hidden size / layers | 768 / 8 |
| Attention heads / KV heads | 8 / 8 |
| FFN size | 3,584 |
| Vocabulary | 6,400 |
| 参数量 | 89,864,448 |
| Sequence length / batch size | 384 / 32 |
| Optimizer steps | 168,000 |
| Seen tokens | 2,064,384,000（约 23 tokens/参数） |
| Learning rate | `5e-4`，AdamW betas `(0.9, 0.95)` |
| Schedule | WSD：warmup 200 + stable 80% + cosine decay |

模型使用 QK RMSNorm、RoPE、MHA 和 SwiGLU。训练语料经 NFKC 归一、质量过滤、
PII、MinHash 近似去重与**全局置换打散**后预分词为 memmap 块
（5,500,556 块 = 2.117B tokens）。checkpoint 原子保存模型、optimizer、
scheduler、AMP scaler、随机数状态和绝对 packed-block 游标，可在 optimizer
边界精确恢复。

## 门 v3 判定结果

判据是**棘轮**（不得比参照差超过 3×噪声底，参照 = 外部锚点与历史最优取严）
+ **金丝雀**（退役绝对阈值抓灾难回归）+ **生成健康**，取代 v1 照着当时最好模型
描出来的绝对阈值。两个 seed 均 `passed: true`，八项门控判据全过。

| 判据（均有门控资格） | 候选 seed20260511 | 参照 | 余量 | 3×噪声底容差 |
| --- | ---: | ---: | ---: | ---: |
| strict_val | 1.7268 | 1.8364 | +0.1096 | 0.0204 |
| strict_test | 1.7210 | 1.8283 | +0.1074 | 0.0204 |
| 六项 domain 切片 | 1.3229–1.9050 | 1.4431–2.0054 | +0.1005~+0.1719 | 0.0432 |
| generation_health.degenerate | 0 / 100 | 0 / 100 | 0 | 3（金丝雀 20） |

跨源泛化（Team 异源 holdout，同参照同管线）：val 3.1927 → 3.0550，
test 3.2170 → 3.0765，效应 −0.135~−0.141 nats，是双 seed 差（0.0030–0.0045）
的 30–45 倍。

**MCQ 准确率与旧版 generation.repetition 没有门控资格**，只记录不作能力证据：
MCQ v2 经用户审核判定不一致且功效不足（192 题极限 ±7pp），旧 repetition 仪器
早停免检且样本量 n=10。任何能力性表述只能追溯到上表这些有资格的仪器。

## 复现

```bash
uv sync --extra dev --frozen
uv run dvc repro build_official_training_view    # 清洗 → 过滤 → PII → 去重 → 全局打散
uv run python scripts/pretokenize_training_view.py   # → blocks_384.u16 memmap
uv run python scripts/preflight_training.py --require-cuda
uv run python scripts/train.py                        # 预算超语料即拒，不会跑到末尾才炸
```

DVC 描述相同的 fetch → audit → train → evaluate 依赖图（`uv run dvc dag`）。

从 checkpoint 到发布物：

```bash
uv run python scripts/evaluate_candidate.py --checkpoint <ckpt> --output <eval.json>
uv run python scripts/judge_gate_v3.py --evaluation <eval.json> --output <verdict.json>
uv run python scripts/register_candidate.py --name <n> --checkpoint <ckpt> \
  --evaluation <eval.json> --verdict <verdict.json> \
  --training-summary <summary.json> --preflight <preflight.json> --limitation "…"
uv run python scripts/export_release_weights.py --name <n> --output <weights.pt>
```

登记只接受门控判定为通过的候选；发布物的血缘从注册表读出而非重敲，权重文件与
其许可条款一同写入，两者不可分离。任何发布必须写明用的是 Official、Team 还是
混合数据。

## 最新预训练与训练 Infra 进展

阶段正确的预训练门、统一数据索引、共享 GPU 容量准入、训练遥测和精确恢复演练
已经落地。本轮新增：

- **仪器效度前置**：任何度量参与门控前须过不变性、区分度、一致性、噪声底、
  样本量五问并在效度台账留证。生成健康仪器按此重建（固定生成长度消早停免检、
  100 条独立探针消模板繁殖、脚本中立统计量消书写系统代理）。
- **门 v3**：棘轮 + 金丝雀取代照当前最好模型描出的绝对阈值。
- **预算护栏**：`train.py` 构造数据集后立即比对所需 block 与语料容量，超出即拒
  并给出最大可承受步数——172k 预算比语料多要 107 步，此前只在单 epoch 跑到最
  末尾时才暴露，两卡各损失 7 小时。
- **中断 run 的摘要可恢复**：`scripts/summarize_checkpoint.py` 从 checkpoint 的
  `progress` 块反推 `training_summary.json`，并标注来源。
- **候选注册接受后期门的判定**：评测产物内嵌的是评测运行时的当期门，被后来的
  门 v3 重判后需与判定产物配对登记（`--verdict`），已知限定随记录一同登记
  （`--limitation`），不留给正文散文。
- **外部锚点：端口成立，锚点作废。** 上游 MiniMind 已发布的预训练 checkpoint
  已逐位忠实移植（对上游自己的模型代码最大绝对 logit 差 0.0），但它们训练所用的
  tokenizer 与本项目的不是同一个，同为 6,400 词表却有 1,772 个 token 互不存在。
  用本项目 tokenizer 评测，锚点 loss 12.02–12.57，而均匀猜测是 8.76——比随机还差。
  闸门失败即拒、不写产物、不报告任何方向的比较，M1 如实降级为三支柱。见
  [外部锚点报告](docs/experiments/2026-07-22-external-anchor.md)。

历史 targeted-only / replay pilot 均未达到预先冻结的扩大条件，未登记候选；结果
见 [2026-07-20 预训练 pilot 与训练恢复报告](docs/experiments/2026-07-20-pretrain-pilots-and-training-recovery.md)。
`replay` 视图继承 Team 数据许可限制，不可作为公开发布数据。主机重启演练因无权限
仍为阻塞，已用 SIGKILL 注入替代并通过 10 项一致性检查。

## 推理与服务

本地 CLI 从 checkpoint 内嵌配置重建模型：

```bash
uv run python scripts/inference.py \
  --checkpoint mindsurf-pretrain-80m-seed20260721.pt \
  --tokenizer data/raw/minimind_official_v1/tokenizer \
  --prompt "你好，请介绍一下机器学习。"
```

发布的权重文件只带 `model_config`、`model_state_dict` 和 `release` 血缘块，
与完整训练 checkpoint 走同一加载路径。**这是预训练基座，不是指令模型**：它续写
文本，不遵循指令，指令遵循要等 SFT 阶段。

API 通过 `INFERENCE_CHECKPOINT` 和 `INFERENCE_TOKENIZER` 加载同一模型。未配置或
加载失败时 `/inference` 明确返回 `503`，不会返回模拟结果。长训练任务由 Celery
排队，worker 并发固定为 1，避免同机出现多个训练进程。
