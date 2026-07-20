# 2026-07-20 预训练 pilot 与训练恢复结果

本轮只覆盖预训练算法和训练 Infra，不包含推理服务。结论是：能力门、数据接口、
共享 GPU 准入和精确恢复已经得到实测证据，但三个小预算 continuation 都未达到
预先冻结的扩大条件，因此没有启动正式训练、第二 seed 或候选登记。

机器可读结果见
[`2026-07-20-pretrain-pilots-and-training-recovery.json`](2026-07-20-pretrain-pilots-and-training-recovery.json)。

## 数据与评测

- 预训练门与后训练门已拆分；预训练门不再使用指令遵循目标。
- MCQ 从 48 题扩充到冻结的 192 题，覆盖中文事实、数学、代码、英语、长上下文和
  校准推理。结果保存逐题 choice probability、预测和答案。
- 训练污染扫描覆盖 Official 与 Team 共 3,443,826 条训练记录，未发现 MCQ 题干的
  NFKC 规范化精确碰撞。
- 同候选比较同时报告 10,000 次配对 bootstrap 置信区间和精确 McNemar 检验。
- Official、Team、targeted 和 80:20 replay 都通过统一数据索引解析；replay 使用
  新 dataset ID、固定比例、采样 seed、manifest 和 run identity。

80:20 replay 训练视图包含 240,000 条 Official 和 60,000 条 targeted 记录，
SHA-256 为
`c5b2e57d791fd48ad56db8dc8dca29cc6f45287857c9c32a370c994d6f72d652`。
它继承 Team 数据的许可限制，不能作为公开发布数据。

## Pilot 结果

三组实验都从同一个 Official-only checkpoint 初始化，固定 1,000 steps、
12,288,000 seen tokens、sequence length 384、batch size 32 和 seed 42。

| Pilot | 主要变量 | Official strict mean | MCQ | MCQ 差值 | 生成高重复 | 结论 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| targeted | targeted-only，LR `1e-4` | 2.7694 | 31.77% | -5.73 pp | 10 | 停止 |
| targeted-low-lr | 仅将 LR 改为 `1e-5` | 2.4628 | 35.94% | -1.56 pp | 10 | 停止 |
| 80:20 replay | Official:targeted = 80:20，LR `1e-4` | 2.6544 | 30.21% | -7.29 pp | 8 | 停止 |

父模型的 Official strict mean 为 2.3725，MCQ 为 37.50%，高重复计数为 7。
第一组的 MCQ 差值 95% 配对 bootstrap 区间为 `[-10.94, -1.04] pp`，
McNemar `p=0.0347`；80:20 replay 为 `[-12.50, -2.60] pp`，
McNemar `p=0.00661`。低学习率组虽然统计上未显示明确 MCQ 下降，但 strict mean
仍回退 0.0903，且生成高重复从 7 增至 10。

这些结果否证了“在当前训练目标下，短 targeted continuation 能提升能力门且保持
Official 能力”的假设。按冻结的停止条件，不用增加训练步数掩盖失败，也不把失败臂
扩成正式训练。下一轮算法工作应先归因训练目标、样本粒度和重复退化，而不是继续微调
同一数据比例。

## 训练 Infra 结果

- 共享 GPU 准入在启动前合并计算外部进程显存和项目 lease；三组训练均在准入后启动，
  峰值实测显存 9,750 MiB，没有 OOM，也没有抢占外部进程。
- 训练摘要记录 loss、learning rate、gradient norm、tokens/s、数据等待、
  optimizer、checkpoint 写入和峰值显存；三个 pilot 的有效吞吐为
  80,805–83,812 tokens/s。
- 独立采样记录 GPU 利用率、显存、温度、功耗和 checkpoint 磁盘余量；本轮峰值为
  100% 利用率、62°C、354.34 W。
- run 注册表覆盖 queued、running、failed、completed 和同 run 重试历史；
  GPU lease 与 run 状态使用原子文件更新。
- 真实恢复演练在 step 10 对训练子进程发送 SIGTERM，再由新进程从 rolling
  checkpoint 恢复到 step 40。恢复结果与不中断基线在模型、optimizer、scheduler、
  CPU/CUDA RNG、global step、micro step、packed-block 游标和 seen tokens 上完全一致。

恢复一致性依赖 deterministic CUDA 设置和独立 DataLoader generator；后者避免
DataLoader iterator 在进程重启时额外消耗全局 CPU RNG。

## 尚未完成

- 三个 pilot 都未过扩大门，所以预训练候选、第二 seed、正式训练和模型注册仍未完成。
- checkpoint、必要数据视图和评测产物尚无经反查的私有 DVC remote 或对象存储副本。
- 未在空目录中从远端资产完成端到端重建。
- 已完成真实训练进程中断和 worker 重拉起；未在共享服务器上执行主机重启。
- NaN、GPU 容量和磁盘水位已有保护；loss 突升与长时间无进展的自动告警闭环仍待补齐。

因此当前状态应表述为“预训练与训练 Infra 进行中”，不能表述为已经完成。
