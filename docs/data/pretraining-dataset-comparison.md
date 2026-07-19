# 预训练数据源对照与发布说明

## 两套数据的身份

Official 臂延续项目此前的 MiniMind 官方基线：

- 数据集：`gongjy/minimind_dataset`
- revision：`74aad49fa4443e7ed640d44bc4e9c7d1fe71ada5`
- 原始文件：`pretrain_t2t_mini.jsonl`
- 原始文件：1,270,238 行，1,241,043,656 bytes
- 原始 SHA-256：
  `6dd6716c84ab36897bdbfc7f88e04f4441c48c1ab7ecee88ce0b0e7d4685560c`
- strict train / validation / test：1,265,983 / 2,000 / 2,000 行
- strict SHA-256：
  `a121ae89a213e8223bc8bf7620a031344d1a348627a727f87f56135957d5edaf` /
  `664772d3a8420fe49104c6c531c7eebeeaf92290dd4ab3d5bb34fe88a2fc1f34` /
  `a09ece1f82cf059b570f0ed27a758c87929bbd04d9518bf118d118b23cfb4daa`
- 数据卡许可：`CC-BY-NC-4.0`

Team 臂是 MindSurf 成员整理的数据，不是 MiniMind 官方数据：

- 数据集：`wyywnab/mindsurf_pretrain_dataset`
- revision：`ab97cc8ba19d08175e5567bd2816ec850b8bb200`
- strict train / validation / test：2,177,856 / 2,000 / 2,000 行
- 数据卡许可字段：`other`
- 团队训练视图在相同 NFKC 规则下进一步移除 11 条等价重复文本。

两者都使用同一固定 MiniMind tokenizer，但来源构成、样本数、文本长度分布、split
形成过程和许可状态不同。loss 只在同一 holdout 上比较；不同 holdout 的绝对 loss
不直接用于宣称某数据源更优。

## 受控实验

数据消融包含 Official 和 Team 两个从零训练臂。以下项目完全相同：

- 89,864,448 参数的 MHA/FFN3584 模型；
- tokenizer 与 vocabulary 6,400；
- seed `20260511`；
- sequence length 384、batch size 32；
- 10,000 optimizer steps、122,880,000 seen tokens；
- learning rate `5e-4` 与同一 WSD schedule；
- 训练实现、checkpoint 语义和评测代码。

唯一主要变量是预训练数据。两个 checkpoint 都在两套 strict holdout 上形成 2×2
交叉评测矩阵，并共用 MCQ 与固定提示词。此前官方数据上的历史结果只作为历史基线，
不能与新 Team 结果伪装成严格单变量对照。

## 发布时必须写明

发布说明必须明确写出以下一种数据身份：

1. `Official-only`：只使用 MiniMind 官方数据，受 `CC-BY-NC-4.0` 及署名要求约束；
2. `Team-only`：只使用团队数据；逐来源许可未澄清前不得公开发布权重；
3. `Mixed`：同时使用两者；必须另外公布混合比例、采样方法、seen tokens 和新
   manifest，本次两臂消融不属于 mixed。

内部能力门与公开许可门相互独立。能力指标通过不代表许可通过；许可可用也不代表
模型达到了发布质量。
