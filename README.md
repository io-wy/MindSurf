# MindSurf Pretrain

本分支交付一个 **89,864,448 参数、中文为主的语言模型基座**，以及产出它的完整
可复现管线：数据处理、训练、评测、发布门与权重导出。所有环节的输入输出都有
不可变身份（SHA-256 + manifest 继承链），任何一条结果都能追回到产生它的代码、
数据、配置与 checkpoint。

训练采用单进程单卡。该规模不需要分布式，引入它只会增加故障面。

**这是预训练基座，不是指令模型。** 它续写文本，不遵循指令；指令遵循能力由后续
SFT 阶段提供。

---

## 1. 交付物

| 文件 | 大小 | 说明 |
| --- | ---: | --- |
| `mindsurf-pretrain-80m-seed20260511.pt` | 359 MB | 权重 + 配置 + 血缘 |
| `mindsurf-pretrain-80m-seed20260721.pt` | 359 MB | 同上，另一个随机种子 |
| 同名 `.manifest.json` | < 5 KB | 权重 SHA-256、血缘、许可条款 |

两个 seed 都发布：单个模型的指标无法说明差异是真实的还是随机的，两个才能给出
可比较的基准。

发布的是**权重**而非训练 checkpoint——optimizer、scheduler、AMP scaler 与随机数
状态只有断点续训需要，不随发布分发。

加载：

```python
import torch
from python_starter.core.model import ModelConfig, TransformerLM

blob = torch.load("mindsurf-pretrain-80m-seed20260721.pt", map_location="cpu", weights_only=False)
model = TransformerLM(ModelConfig.from_dict(blob["model_config"]))
model.load_state_dict(blob["model_state_dict"], strict=True)
model.eval()
```

或直接用 CLI：

```bash
uv run python scripts/inference.py \
  --checkpoint mindsurf-pretrain-80m-seed20260721.pt \
  --tokenizer data/raw/minimind_official_v1/tokenizer \
  --prompt "你好，请介绍一下机器学习。"
```

完整发布说明（含全部限定与五向追溯）见
[权重发布说明](docs/releases/2026-07-22-pretrain-80m.md)。

---

## 2. 许可：先看这一节再用

- 训练数据是 `gongjy/minimind_dataset` @ `74aad49f`（ModelScope），许可
  **CC-BY-NC-4.0**。
- **权重继承数据的许可**，因此本权重及其任何衍生物（包括微调结果）同样受
  CC-BY-NC-4.0 约束：使用须标注上述数据来源（BY），且**不得商用**（NC）。
- MiniMind **代码仓库**的 MIT 许可不适用于本权重——那是代码的许可，不是数据的。

许可状态按事实记录，不写成"已核实"：维护者声明已获数据集原作者授权贡献，用途
为非商用学习研究，证据是维护者口头陈述。机器可读原文见
[`configs/release/pretrain_80m_license.json`](configs/release/pretrain_80m_license.json)。

---

## 3. 数据

仓库保留两套并列数据源，团队数据不冒充官方数据：

| 数据臂 | 身份 | Validation / test | 许可状态 |
| --- | --- | ---: | --- |
| Official | `gongjy/minimind_dataset@74aad49` | 2,000 / 2,000 | `CC-BY-NC-4.0` |
| Team | `wyywnab/mindsurf_pretrain_dataset@ab97cc8` | 2,000 / 2,000 | `other`，来源许可待澄清 |

**已发布权重只用 Official 数据训练，无一行 Team 数据**，所以 Team 的许可待澄清
问题不传导到发布物。Team holdout 只用作**异源**评测（见 §5）。

处理链固定为：原始文本 → NFKC 归一与空白折叠 → 质量过滤 → 安全/PII →
MinHash 近似去重 → **全局置换打散** → 预分词 memmap → 流式加载。每个阶段的输出
是一个新身份，输入视图不动，因此任意两阶段可对照。

实测：近似去重命中 444,897 条（5.26%），PII 命中 8,739 行，总丢弃 5.44%；
预分词产出 5,500,556 块 = 2.117B tokens；训练视图 SHA-256
`4222e9ca19e8e2e369beee41dd00594ee87eb8d9b946738c16854506f1da47a2`。

打散必须是**全局置换**：两套语料都按来源排序，窗口打散修不了源排序，尾部是纯
英文与代码，顺序读的部分 epoch 必然有偏。过滤阈值在 ≥50k 真实样本上做过不变性
标定，分语种丢弃率差与被丢样本长度分布都留档——否则一个看似中立的阈值会定向
削薄某一类语料，而总丢弃率上看不出来。

详见[数据集对照说明](docs/data/pretraining-dataset-comparison.md)与
[团队数据集说明](docs/data/mindsurf-team-dataset-v1.md)。

---

## 4. 模型与训练配方

| 配置项 | 数值 |
| --- | ---: |
| 参数量 | 89,864,448 |
| Hidden size / layers | 768 / 8 |
| Attention heads / KV heads | 8 / 8 |
| FFN size | 3,584 |
| Vocabulary | 6,400 |
| Sequence length / batch size | 384 / 32 |
| 完成 optimizer steps | 168,000 |
| Seen tokens | 2,064,384,000（约 23 tokens/参数） |
| Learning rate | `5e-4`，AdamW betas `(0.9, 0.95)`，weight decay `0.01` |
| Schedule | WSD：warmup 200 + stable 80% + cosine decay，min_lr_ratio 0.1 |
| 精度 | bfloat16 autocast，RMSNorm 均方归约上溯 float32 |

结构：RoPE（theta 1e6）、QK RMSNorm、MHA、SwiGLU、权重绑定。

计算最优约 20 tokens/参数，本配方 23，处在合理区间。**改 `max_steps` 就是改整条
LR 轨迹的形状**，任何跨预算的比较都必须写明这一点。

---

## 5. 结果

### 5.1 发布门

一个候选要被登记与发布，必须通过**发布门**
（[`configs/evaluation/pretrain_gate_v3.json`](configs/evaluation/pretrain_gate_v3.json)），
它由三部分组成：

- **棘轮**：每项门控指标不得比参照差超过 3×噪声底。参照系取外部锚点与历史最优
  中更严的一个。
- **金丝雀**：一批宽松的绝对阈值，不再用于区分候选，只用于抓灾难性回归。
- **生成健康**：固定长度贪心生成，100 条独立主题探针，判定退化。

门槛不照着当前最好的模型描绝对值——那样的阈值在下一个候选甩开它之后就失去
区分度，只剩下"全部通过"这一种输出。

两个 seed 均 `passed: true`，八项门控判据全过：

| 判据（均有门控资格） | 候选 seed20260511 | 参照 | 余量 | 3×噪声底容差 |
| --- | ---: | ---: | ---: | ---: |
| strict_val | 1.7268 | 1.8364 | +0.1096 | 0.0204 |
| strict_test | 1.7210 | 1.8283 | +0.1074 | 0.0204 |
| 六项 domain 切片 | 1.3229–1.9050 | 1.4431–2.0054 | +0.1005~+0.1719 | 0.0432 |
| generation_health.degenerate | 0 / 100 | 0 / 100 | 0 | 3（金丝雀 20） |

双 seed 实测差：strict_val 0.0020、strict_test 0.0031、六项 domain 0.0003–0.0051、
退化计数 0。门控余量是该差值的 7–12 倍。

### 5.2 跨源泛化

同分布 holdout 的 loss 只度量模型对自家语料的压缩，单独不构成质量声明。因此在
Team（`wyywnab`，异源）holdout 上用**同一参照、同一管线**复核：

| 模型 | Team val | Team test |
| --- | ---: | ---: |
| 参照 | 3.1927 | 3.2170 |
| 候选 seed20260511 | **3.0580** | **3.0810** |
| 候选 seed20260721 | **3.0550** | **3.0765** |

效应 −0.135~−0.141 nats，是双 seed 差（0.0030–0.0045）的 30–45 倍。方向可报：
**跨源改善**。

### 5.3 明确不主张的

- **MCQ 准确率没有门控资格。** 192 题的分辨极限是 ±7pp，而关心的效应是 3–5pp，
  物理上分不出；人工审核也判定其一致性不足。只记录，不作能力证据。
- **旧版 `generation.repetition` 计数没有门控资格。** 该仪器允许模型提前停止就
  免检——7 字符的完成能拿满分——且样本量只有 10。已由 §5.1 的生成健康仪器取代。
- **外部锚点未成立。** 上游 MiniMind 已发布的预训练 checkpoint 已被逐位忠实移植
  （对上游自己的模型代码最大绝对 logit 差 **0.0**），但它们训练所用的 tokenizer
  与本项目不是同一个：同为 6,400 词表，1,772 个 token 互不存在。用本项目
  tokenizer 评测，锚点 loss 为 12.02 与 12.57，而均匀猜测是 8.76——比随机还差，
  即 token id 含义错位。因此**不报告任何方向的比较**。证据与确立方式见
  [外部锚点报告](docs/experiments/2026-07-22-external-anchor.md)。

**任何能力性表述只能追溯到 §5.1 表中有门控资格的仪器。**

### 5.4 已知限定

1. **实际训练 168,000 步而非计划的 172,000。** 请求的预算比预分词视图能提供的
   多 107 步。step-168000 的 checkpoint 完整，覆盖该 epoch 的 97.7%，且已越过 WSD
   退火起点。`train.py` 现在在训练开始前就拒绝超出语料容量的预算，并给出最大可
   承受步数。
2. **双 seed 只扰动初始化。** 数据加载器顺序读且拒绝 shuffle buffer，两个 seed
   的数据顺序完全相同。语料已在预分词前全局打散，两 seed 覆盖同一批语料。因此
   §5.1 的双 seed 差是噪声底的**下界**，不是估计值，据此得出的结论强度相应降级。
3. **`training_summary.json` 由 checkpoint 反推**（`scripts/summarize_checkpoint.py`），
   字段全部取自 checkpoint 的 `progress` 块，文件内 `derived_from_checkpoint`
   标明来源。
4. **下游可用性未自证**，留待 SFT 阶段回流验证。

---

## 6. 复现

```bash
uv sync --extra dev --frozen
uv run dvc repro build_official_training_view       # 清洗 → 过滤 → PII → 去重 → 全局打散
uv run python scripts/pretokenize_training_view.py  # → blocks_384.u16 memmap
uv run python scripts/preflight_training.py --require-cuda
uv run python scripts/train.py
```

DVC 描述同一张 fetch → audit → train → evaluate 依赖图（`uv run dvc dag`）。

从 checkpoint 到发布物：

```bash
uv run python scripts/evaluate_candidate.py --checkpoint <ckpt> --output <eval.json>
uv run python scripts/judge_gate_v3.py --evaluation <eval.json> --output <verdict.json>
uv run python scripts/register_candidate.py --name <n> --checkpoint <ckpt> \
  --evaluation <eval.json> --verdict <verdict.json> \
  --training-summary <summary.json> --preflight <preflight.json> --limitation "…"
uv run python scripts/export_release_weights.py --name <n> --output <weights.pt>
```

登记只接受门控判定为通过的候选；已知限定随记录一同登记，不留给正文散文。发布物
的血缘从注册表读出而非重新录入，权重文件与其许可条款一同写入，两者不可分离。
任何发布必须写明用的是 Official、Team 还是混合数据。

---

## 7. 训练 Infra

- **Checkpoint 契约**：原子写（临时名 + fsync + rename）；内容含模型、optimizer、
  scheduler、AMP scaler、CPU/CUDA/Python/numpy 随机数状态、global/micro step 与
  绝对 packed-block 游标。恢复逐位一致，SIGTERM 与 SIGKILL 注入均已验证。
- **预算护栏**：构造数据集后立即比对所需 block 与语料容量，超出即拒。
- **共享 GPU 准入**：按卡计算可用显存并申请租约，避免同机多个训练进程互相挤爆。
- **监控四层**：算法（loss、ppl、grad norm、LR）、计算（tokens/s、在线 MFU、
  显存）、硬件（温度、功耗、不可纠正 ECC 即 critical）、数据（等待时长）。告警
  用趋势与联合条件，不用单点阈值——单 micro-batch loss 在 1.5–4.7 之间无故障波动，
  单点阈值必然误报。
- **吞吐**：MFU 28.7–31.3%，该规模属正常区间，吞吐不是瓶颈。
- **发布门禁**：`scripts/release_gate.py` 在提交与推送前审计**文件内容与提交
  消息**，拦截绝对路径、内网地址、凭据、模型与数据块等不该进入共享仓库的内容。

测试：170 项，覆盖率 83.29%（CI 门槛 80%）。CI 另行执行 `ruff check`、
`ruff format --check` 与 `mypy --strict`。

```bash
uv run pytest
uv run ruff check src tests scripts && uv run ruff format --check src tests scripts
uv run mypy src tests scripts
```

---

## 8. 推理与服务

API 通过 `INFERENCE_CHECKPOINT` 与 `INFERENCE_TOKENIZER` 加载同一模型。未配置或
加载失败时 `/inference` 明确返回 `503`，不返回模拟结果。长训练任务由 Celery 排队，
worker 并发固定为 1。

分布式训练、vLLM/TensorRT-LLM、量化、推测解码与多副本路由**暂不建设**：在 89.9M
单卡模型上它们是纯负债。触发条件是单卡放不下，或多卡能显著缩短受约束的关键路径。

---

## 9. 下一阶段

1. **SFT**：以本基座为起点做指令微调，并把下游可用性结果回流到 §5.4 的第 4 条。
2. **外部锚点**：上游若发布基于当前 tokenizer 训练的权重，移植与 tokenizer 闸门
   已在库（`scripts/evaluate_external_anchor.py`），锚点是一条命令的事。
3. **双 seed 真正独立**：按 seed 生成块索引置换（随机访问，成本极低），让未来的
   run 天然同时扰动初始化与数据顺序，把噪声底从下界变成估计值。
