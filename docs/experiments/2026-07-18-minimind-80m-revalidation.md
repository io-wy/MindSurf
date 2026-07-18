# MiniMind 80M 预训练复验证

状态：**实验完成，模型未发布**

本报告记录约 80M 参数规模的受控预训练实验。目标是在固定数据、训练预算和评测
口径下比较学习率、Attention 结构、FFN 宽度及续训数据配比，并确定下一轮实验
的起始配置。

对应的机器可读结果：
[`2026-07-18-minimind-80m-revalidation.json`](2026-07-18-minimind-80m-revalidation.json)。

## 结论

当前候选配方为：

- MHA，hidden size 768，8 层，8 attention heads，8 KV heads，
  FFN 3584；
- 基础训练使用 WSD，学习率 `5e-4`，batch size 32，sequence length 384，
  10,000 optimizer steps，共 122,880,000 seen tokens；
- 第一阶段续训使用 quality70/English15/math15，700 steps，
  sequence length 512，batch size 16，学习率 `8e-7`；
- 第二阶段保持相同数据配比，1,200 steps，学习率 `3e-7`；
- 使用 seed 42 和 seed 7 进行复验。

所有最终候选均未通过完整能力门，因此本轮不发布模型。

## 数据身份

| Artifact | SHA-256 |
| --- | --- |
| Strict manifest | `64e18821cc7c18d96fd149a0480d774fcfd6d30564b96d072a7d8b1d5726270f` |
| Strict train | `a121ae89a213e8223bc8bf7620a031344d1a348627a727f87f56135957d5edaf` |
| Strict validation | `664772d3a8420fe49104c6c531c7eebeeaf92290dd4ab3d5bb34fe88a2fc1f34` |
| Strict test | `a09ece1f82cf059b570f0ed27a758c87929bbd04d9518bf118d118b23cfb4daa` |
| Tokenizer bundle | `7e76729752b4463589393a037bd6a6b1ff370964de1870216dec07a3f1d273ff` |
| Candidate thresholds | `d975c2b1dac4b7021d97efcc0b56fb9ed96b1260faa7b922b61ec45f787d8d5b` |

训练集包含 1,265,983 行，验证集和测试集各 2,000 行。训练集已排除与验证集、
测试集规范化文本哈希重复的样本。

评测门槛：

- strict validation loss 不高于 `2.42`；
- strict test loss 不高于 `2.44`；
- MCQ 不低于 `23/48`；
- fixed prompt score 不低于 `0.4575`；
- 高重复样本不超过 4；
- 六类诊断域损失全部不高于预设阈值。

MCQ 置信区间使用 10,000 次 bootstrap，随机种子为 `20260712`。

## 基础训练对照

所有实验使用相同的数据身份、tokenizer、数据顺序、WSD schedule、
batch size 32、sequence length 384、10,000 optimizer steps 和
122,880,000 seen tokens。

| Run | Strict val | Strict test | Mean | Tokens/s | Peak GiB | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| MHA/FFN3072，LR `3e-4` | `2.438767` | `2.452744` | `2.445755` | `104132.66` | `7.947` | strict loss 退化 |
| MHA/FFN3072，LR `5e-4` | `2.417287` | `2.434266` | `2.425776` | `104245.35` | `7.947` | 学习率参考 |
| MHA/FFN3072，LR `7e-4` | `2.410559` | `2.427583` | `2.419071` | `104237.10` | `7.947` | 与 `5e-4` 差异较小 |
| GQA/FFN3328，LR `5e-4` | `2.428264` | `2.441455` | `2.434859` | `110189.39` | `8.004` | validation loss 退化 |
| MHA/FFN2816，LR `5e-4` | `2.430995` | `2.443201` | `2.437098` | `107355.21` | `8.014` | strict loss 退化 |
| MHA/FFN3584，LR `5e-4` | `2.399181` | `2.415155` | `2.407168` | `98841.02` | `8.811` | 当前候选 |
| MHA/FFN4096，LR `5e-4` | `2.397700` | `2.411336` | `2.404518` | `94004.80` | `8.943` | loss 接近但效率较低 |

FFN3584 有 89,864,448 个参数。FFN4096 有 99,301,632 个参数，strict mean
loss 仅改善 `0.002650`，低于预设的 `0.01` 有效差异阈值。FFN3584 的 MCQ
更高、五类诊断域 loss 更低、吞吐高约 5.1%，因此作为续训父模型。

## 两阶段续训

每个 seed 的 quality70 control 与 quality80 replay 均从同一个 quality70
第一阶段 checkpoint 开始。第二阶段统一使用 1,200 steps、19,200 consumed
blocks、sequence length 512、batch size 16、学习率 `3e-7` 和
9,830,400 seen tokens。

| Seed / arm | Strict val | Strict test | Mean | MCQ | Fixed | Repetition |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 / quality70 control | `2.391499` | `2.409126` | `2.400313` | `15/48` | `0.44` | `2` |
| 42 / quality80 replay | `2.390968` | `2.408508` | `2.399738` | `15/48` | `0.44` | `2` |
| 7 / quality70 control | `2.396327` | `2.412125` | `2.404226` | `17/48` | `0.475` | `1` |
| 7 / quality80 replay | `2.395817` | `2.411612` | `2.403715` | `17/48` | `0.3785` | `3` |

quality80 replay 的 strict mean loss 在 seed 42 和 seed 7 上分别改善
`0.000574` 和 `0.000512`，MCQ 没有变化。seed 7 的 fixed score 从
`0.475` 降至 `0.3785`，高重复样本从 1 增至 3，因此最终保留
quality70 control。

## 结果边界

- 当前配方只代表本轮数据与预算下的候选起点；
- 所有候选均未通过六类诊断域 loss 和 MCQ 门槛；
- 新数据实验需要重新生成数据身份并独立记录结果；
- checkpoint、逐样本评测输出和训练日志未包含在本仓库中。
