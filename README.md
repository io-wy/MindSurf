# MindSurf Pretrain

`pretrain` 分支用于 MiniMind 小参数量语言模型的预训练研究，重点关注数据质量、
训练效率和可复现评测。目前已完成一轮约 80M 参数规模的受控实验，覆盖学习率、
Attention 结构、FFN 宽度和两阶段续训配方。

## 当前进度

| 项目 | 进展 |
| --- | --- |
| 数据划分 | 完成训练、验证、测试集隔离并记录 SHA-256 |
| 基础训练 | 完成 7 组等预算对照 |
| 架构选择 | 选定 MHA / FFN3584 作为当前候选 |
| 续训实验 | 完成两种数据配比及 seed 42、seed 7 复验 |
| 评测 | 完成严格损失、六类域损失、MCQ、固定提示词和重复度评测 |
| 模型发布 | 暂未发布；最终候选未通过完整能力门 |

## 阶段结果

当前表现最均衡的模型配置为：

| 配置项 | 数值 |
| --- | --- |
| Hidden size | 768 |
| Transformer layers | 8 |
| Attention heads / KV heads | 8 / 8 |
| FFN size | 3584 |
| 参数量 | 89,864,448 |
| 训练步数 | 10,000 |
| Seen tokens | 122,880,000 |
| 学习率 | `5e-4` |
| Batch size / sequence length | 32 / 384 |

该配置在 seed 42 的基础训练中取得：

- strict validation loss：`2.399181`
- strict test loss：`2.415155`
- 训练吞吐：约 `98,841 tokens/s`
- 峰值显存：约 `8.81 GiB`

FFN4096 的 strict mean loss 仅改善 `0.002650`，低于实验预先设定的 `0.01`
有效差异阈值，同时参数量增加约 943 万、吞吐下降约 5.1%，因此没有继续扩大
FFN。

## 续训结论

两阶段续训最终比较了 quality70/English15/math15 和
quality80/English10/math10 两种配方。两组实验使用相同父模型、训练预算和
评测口径，并分别以 seed 42 和 seed 7 复验。

| Seed | 配方 | Strict val | Strict test | MCQ | Fixed score | 高重复样本 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 42 | quality70 control | `2.391499` | `2.409126` | `15/48` | `0.44` | 2 |
| 42 | quality80 replay | `2.390968` | `2.408508` | `15/48` | `0.44` | 2 |
| 7 | quality70 control | `2.396327` | `2.412125` | `17/48` | `0.475` | 1 |
| 7 | quality80 replay | `2.395817` | `2.411612` | `17/48` | `0.3785` | 3 |

quality80 replay 在两个 seed 上的 strict mean loss 平均只改善
`0.000543`，MCQ 没有提升；seed 7 的 fixed score 反而明显下降。因此当前
保留 quality70 control 配方。

## 为什么暂不发布模型

候选模型已经通过 strict validation/test loss 和重复度要求，但仍未达到：

- 六类诊断域的损失阈值；
- MCQ 最低要求 `23/48`。

当前结果用于确定下一轮实验起点，不代表模型已经具备稳定的通用能力。完整实验
记录见：

- [80M 预训练复验证报告](docs/experiments/2026-07-18-minimind-80m-revalidation.md)
- [机器可读结果](docs/experiments/2026-07-18-minimind-80m-revalidation.json)

## 数据与复现

本轮数据规模为：

| Split | Rows |
| --- | ---: |
| Train | 1,265,983 |
| Validation | 2,000 |
| Test | 2,000 |

数据 manifest、各 split、tokenizer 和评测阈值的 SHA-256 均记录在实验报告与
JSON 结果中。训练集已排除与 validation/test 规范化文本哈希重复的样本。

仓库提供 Hydra 配置、PyTorch 训练组件、DVC 目录、评测 CLI、实验追踪和
FastAPI 服务骨架。安装开发环境：

```bash
git clone --branch pretrain https://github.com/io-wy/MindSurf.git
cd MindSurf
uv sync --extra dev
```

## 下一阶段

下一轮工作将围绕新的训练数据展开：

1. 统计来源、文本长度、语言和重复分布；
2. 固化 train/validation/test manifest 并检查交叉污染；
3. 将当前 FFN3584 配方接入统一训练入口；
4. 使用独立 run identity 完成等预算复验；
5. 仅在完整能力门通过后发布模型与推理基准。
