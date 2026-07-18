# 80M 预训练数据源消融实验

## 结论

Official-only 与 Team-only 两个候选均完成了相同的 10,000 optimizer steps 和
122,880,000 seen tokens。结果显示强烈的同域优势，但没有证据支持任何一套数据在
全部能力上普遍更优：

- Official-only 在 Official holdout 上的 validation/test loss 为
  `2.3837 / 2.3612`，低于 Team-only 的 `3.0726 / 3.0979`。
- Team-only 在 Team holdout 上的 validation/test loss 为
  `2.9408 / 2.9743`，低于 Official-only 的 `3.7381 / 3.7645`。
- Official-only 的固定 MCQ 为 `20/48`、提示词分数为 `0.240`；Team-only 分别为
  `15/48` 和 `0.161`。
- 两者都未通过完整能力门，也都未通过公开许可门。没有登记内部 candidate，没有
  公开发布任何权重。

同一 holdout 内的模型差异可以解释为预训练数据差异造成的结果；不同 holdout 之间
本身存在难度和分布差异，不能直接用绝对 loss 排名。

## 受控条件

唯一主要变量是预训练数据源。两臂都从零开始，并固定以下条件：

| 条件 | 数值 |
| --- | ---: |
| 参数量 | 89,864,448 |
| Tokenizer / vocabulary | MiniMind shared / 6,400 |
| Seed | 20260511 |
| Sequence length / batch size | 384 / 32 |
| Optimizer steps | 10,000 |
| Seen tokens | 122,880,000 |
| Learning rate | `5e-4` |
| Schedule | warmup 200 + stable 80% + cosine decay |

两套源数据总行数不同，因此固定 seen tokens 会覆盖不同的数据比例。这是数据源变量的
组成部分，也是本实验不能外推到其他预算的限制。实验只运行一个 seed，结论应视为
受控的单次数据消融，而不是稳定性研究。

## 数据身份

| 数据臂 | 冻结身份 | 训练视图 | 许可门 |
| --- | --- | --- | --- |
| Official | `gongjy/minimind_dataset@74aad49fa444` | 1,265,981 行，SHA `757d1b35…7699` | 未通过 |
| Team | `wyywnab/mindsurf_pretrain_dataset@ab97cc8ba19d` | 2,177,845 行，SHA `37800ee0…a999` | 未通过 |

Team 数据不是 MiniMind 官方数据。它由团队成员整理，源训练 split 经相同 NFKC 规则
移除 11 条等价重复文本，且与 validation/test 的重叠为 0。其数据卡许可字段为
`other`，逐来源许可未澄清前不得公开发布权重。Official 数据卡标记
`CC-BY-NC-4.0`，本实验的公开许可门同样保持关闭。

本实验没有混合两套数据。后续若发布 Official-only、Team-only 或 Mixed 结果，必须
分别写明所用数据身份；Mixed 还必须公布混合比例、采样策略、seen tokens 和新的
manifest。

## 训练资源

| 候选 | Final train loss | Tokens/s | 用时 | Peak reserved |
| --- | ---: | ---: | ---: | ---: |
| Official-only | 2.1108 | 91,238 | 1,346.8 s | 9.69 GB |
| Team-only | 2.5790 | 89,195 | 1,377.7 s | 9.69 GB |

两臂都消费 320,000 个 packed blocks、122,880,000 tokens，并写出 step 9,500、
step 10,000、best 与 final checkpoint。checkpoint 仅保留为私有实验资产，没有放入
Git、Release 或其他公开存储。

## 2×2 交叉评测

| 训练候选 | Official val | Official test | Team val | Team test |
| --- | ---: | ---: | ---: | ---: |
| Official-only | **2.3837** | **2.3612** | 3.7381 | 3.7645 |
| Team-only | 3.0726 | 3.0979 | **2.9408** | **2.9743** |

在 Official holdout 上，Official-only 的 validation/test loss 分别低
`0.6889 / 0.7366`；在 Team holdout 上，Team-only 分别低
`0.7973 / 0.7903`。这说明两套训练数据都显著强化了自身分布，而不是产生一致方向的
通用提升。

## 能力门与发布决定

Official-only 通过 strict validation/test loss 门，但未通过
`english_or_code_heavy`、`math_like`、`quality_pass` 三个诊断域，以及 MCQ、固定
提示词和生成重复度门。Team-only 还未通过 strict validation/test loss，并在
`length_long`、`math_like`、`quality_pass`、MCQ、固定提示词和重复度上失败。

因此：

1. 最低同域 loss：各自匹配的数据候选最低；
2. 实际有效差异：存在明显数据域专门化；
3. 能力门：两个候选均未通过；
4. 许可门：两个候选均未通过；
5. 发布：未登记内部 candidate，未公开发布权重。

完整精确值和失败项见同目录的
`2026-07-18-pretraining-dataset-ablation-80m.json`。
