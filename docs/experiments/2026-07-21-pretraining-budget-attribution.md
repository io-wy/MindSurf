# 2026-07-21 预训练预算归因与首个过门候选

本轮只覆盖预训练算法与训练 Infra，不含推理服务。结论是：此前所有实验撞的墙是
**训练预算**，不是数据配比。只把 `max_steps` 从 10,000 提到 60,000、其余配方一个字
不改，候选就通过了完整预训练门，此前卡门的四项全部翻过。同时发现官方上游还有一份
比在用语料大 6.7 倍的完整预训练文件，从未被使用过，也从未有人论证过为什么不用。

机器可读结果见 `artifacts/evaluation/minimind_official_v1_budget60k_seed20260511_80m.json`
与同目录的 `_formal_pipeline_outcome.json`。

## 错误归因

父模型（Official-only，10,000 步）在预训练门上失败四项：

| 判据 | 实测 | 门限 |
| --- | ---: | ---: |
| `domain.english_or_code_heavy` | 3.1571 | 2.9033 |
| `domain.math_like` | 2.2346 | 2.0446 |
| `domain.quality_pass` | 2.3701 | 2.2885 |
| `generation.repetition` | 7 | 4 |

`strict_val` 2.3837、`strict_test` 2.3612 与 MCQ 0.3750 本就通过。四项失败里三项是
loss、一项是生成退化，**没有一项依赖 MCQ**。

三条证据把原因指向预算：

1. 父模型 eval loss 在最后 1,000 步仍从 2.4797 掉到 2.3837，而那时 LR 已在退火，
   **没有任何平台期**。
2. 122,880,000 tokens / 89,864,448 参数 = **每参数 1.37 tokens**，计算最优约在 20，
   差约 15 倍。模型还在 loss-数据幂律的陡坡上。
3. `archive/legacy-minimind` 存档里有 **196 个历史 run**，全是 700–1,200 步、
   LR `3e-7`–`8e-7` 的退火，test loss 总共只移动约 0.001。

被排除的其他归因：**数据覆盖**（在用语料与门限参考模型同源）、**tokenizer**（父模型
与参考模型一致）、**评测错位**（strict holdout 与 MCQ 本就通过，失败集中在 domain
loss 与生成退化）。

此前三个 1,000 步 continuation pilot 的结论必须重述为**受混淆**：它们用 LR `1e-4`，
是父模型收敛时 LR（`5e-4 × 0.1 = 5e-5`）的两倍，并且重置了 optimizer 与数据游标。
"targeted 数据有害"这个结论无法从中推出，真实变量是续训机制与预算，不是数据。

## 预算臂

冻结的假设与触发条件见 `configs/experiments/official_budget_scaling_80m.json`，在
占用 GPU 之前提交。

配置差异经逐键比对：**159 个键中 152 个完全相同**。实质差异只有
`training.max_steps` 10,000 → 60,000；`data.epochs` 从缺省到 3 是为了让这个预算跑得
起来（Official 视图单遍约 844,000 packed block，batch 32 下单遍上限约 26,000 步）。
其余差异是 run 名、输出目录与日志频率。架构、LR `5e-4`、warmup 200、`stable_ratio`
0.8、`min_lr_ratio` 0.1、batch 32、seq 384、bf16、weight decay、grad clip、seed
`20260511`、数据集 revision 与 tokenizer 全部未动。

结果：`failures: []`，候选已登记。

| 判据 | 父模型 | 候选 | 门限 |
| --- | ---: | ---: | ---: |
| `strict_val` | 2.3837 | **1.9529** | 2.42 |
| `strict_test` | 2.3612 | **1.9455** | 2.44 |
| `domain.code_like` | 2.1992 | 1.5636 | 2.3882 |
| `domain.english_or_code_heavy` | 3.1571 | **1.8907** | 2.9033 |
| `domain.length_long` | 2.0810 | 1.6103 | 2.6167 |
| `domain.math_like` | 2.2346 | **1.8472** | 2.0446 |
| `domain.quality_pass` | 2.3701 | **2.0626** | 2.2885 |
| `domain.repeat_high` | 2.2172 | 1.8432 | 2.2573 |
| `generation.repetition` | 7 | **1** | 4 |

strict mean 2.3725 → 1.9492。冻结的扩大触发线是改善 0.02，实测 **0.4233**。
seen tokens 737,280,000，每参数 8.2 tokens，2.27 个 epoch，MFU 28.7%。

## 跨数据 holdout 复核

候选在 Team holdout（域外）上 3.7645 → **3.3017**，改善 0.463，与 Official 上的
0.416 同量级。收益不是把 Official holdout 拟合过去，是真实泛化。同域专门化依然成立：
Team 训练的模型在自己域上是 2.9743，候选没有超过它。

## MCQ 与题库审核

MCQ 0.3750 → 0.3542，配对 bootstrap 95% 区间 `[-9.4, +4.7] pp`、
McNemar `p = 0.665`（26 题仅父模型对，22 题仅候选对），统计上与无变化不可区分。

同日维护者按种子 `20260720` 分层抽检 32 题，结论记于
`configs/evaluation/pretrain_mcq_benchmark_v2.review.json`：**标注答案 32/32 全部
正确**，但该套件**不可作为能力、长上下文或推理的正式证据**。主要缺陷：`long_context`
的 32 条记录实为 4 个模板各复制 8 份，全库去重后最多 164 个独立题目模板；两个类别名
与实测能力不符；干扰项与正确答案不在同一语义层级，会系统性高估模型。

审计中原本写着 `human_review.status = "completed"`，但那只由一个命令行开关决定，
没有任何人抽检过。该字段已改为必须指向真实审核文档并嵌入其摘要与
`reviewed_this_benchmark` 标志。MCQ 门限一个未动、仍然阻塞——仪器偏松时放松门限是
反方向的处理——但门配置中已记录该局限，禁止把"MCQ 通过"读作能力结论。

## 数据源发现：mini 与 full

在用语料是 `pretrain_t2t_mini.jsonl`（1.24 GB）。同一上游 revision 还发布着
`pretrain_t2t.jsonl`，**8.28 GB，是在用语料的 6.7 倍**。按 mini 实测 token 密度外推，
完整语料约 21.6 亿 tokens ≈ 每参数 24 tokens，正好落在计算最优附近。

追溯选型理由：**仓库任何地方都没有记录**。引入该 spec 的 commit `a03f6e4` 未说明，
文档只写"延续项目此前的 MiniMind 官方基线"。legacy 存档里完整语料**早已登记在案**
（`asset_sources.json` 中 id 为 `full-pretraining-dataset`，带恢复流程），但整个存档
中 mini 出现 31 次、full 出现 7 次且**全部是资产登记，没有一次用于训练**。

即：一个从未被论证过的继承决策，成了项目的主要瓶颈，而 196 + 3 次实验都在这个瓶颈
之下调配方。本文件记录此事，以免下一轮再被"延续基线"继承一次。

已建立可比的全量臂：`configs/datasets/minimind_official_full_v1.json` 保持冻结的
4,000 条 strict holdout 不变，只把 `train` 指向完整语料。训练视图构建器按 NFKC 文本
摘要从 `train` 中剔除 holdout 行，因此得到的是"完整语料减去冻结 holdout"，现有门限与
此前全部对比保持有效，且 manifest 中的 `holdout_rows_removed` 会直接回答 mini 是否为
full 的子集。

## 训练 Infra 本轮补齐

**告警闭环**（`scripts/watch_training.py`）：非有限 loss、loss 突升、长时间无进展、
显存压力、磁盘水位。两次正式训练全程 `alerts: []`。突升规则用相邻两窗中位数对比而非
最新值对比——父模型逐 1,000 步的单 micro-batch loss 在 1.498 到 4.726 之间波动而无任何
故障，最新值对比会持续误报。这条规则是被真实数据证伪后改的。

**吞吐基线**（`scripts/report_throughput_baseline.py`）：父模型 91,238 tokens/s、
51.8 TFLOPS、MFU **31.3%**；60k 预算臂 83,494 tokens/s、47.4 TFLOPS、28.7%。按 PaLM
口径计 FLOPs，除以设备 bf16 稠密峰值。结论是吞吐不构成瓶颈，不值得投入优化。

**不可捕获中断演练**：主机重启无权限执行，改用 SIGKILL 作为最接近的替代——不运行任何
处理器、不刷新任何缓冲、进程无法拒绝。`interrupted_return_code: -9`，10 项检查全部
通过：模型、optimizer、scheduler、CPU/CUDA RNG、global step、micro step、
packed-block 游标、seen tokens 全部与不中断基线逐位一致。证据见
`artifacts/infra/training_recovery_drill_sigkill.json`。残留未覆盖：内核级掉电语义与
开机自动重启，该项标记为受权限阻塞。

**GPU 准入**：真实并发下验证——训练持 11,000 MiB lease 时，评测以 4,096 MiB 被准入，
projected 16,642 / 24,564 MiB，未 OOM、未抢占外部进程。被杀 run 的死 lease 在下次
申请时按 `owner_pid` 存活自动回收（实测确认）。

**run 注册表**：原先只记录 pid 从不核活，被杀的 run 永远停在 `running` 并挡住重试，
唯一出路是 `allow_retry`——而该开关连真正在运行的 run 也能覆盖。已改为按存活判定：
owner 已死则记为 `failed` 并放行重试，owner 存活仍然阻塞，`completed` 无条件阻塞。

**血缘**：注册表原先只有 checkpoint 与评测哈希，源码 commit 仅存在于 preflight 打到
日志的一行里，候选无法追溯到训练它的代码。preflight 记录现落盘到 checkpoint 目录，
注册时连同训练摘要一并绑定，`source_git_head` 进入记录。

**资产与保留**：13 个资产 3.40 GB，按 SHA-256 内容寻址存放并重新哈希反查，全部通过，
并已复制到异机。未找到任何对象存储凭据；`/data/cubelet` 是 root 所有的容器运行时状态
目录，不是数据盘，不得写入。训练视图与 checkpoint 均为指向权威 checkout 的符号链接，
让 DVC 改写这些路径会危及唯一权威副本，故采用只读镜像而非 DVC remote，耐久性范围
如实记为"另一文件系统 + 另一主机"。保留策略与磁盘预算见
`docs/operations/retention-and-disk-budget.md`。

## 语料有序性与两个由此产生的缺陷

抽样测量两份语料在文件不同位置的内容特征，两份**都是按来源排序的**，不是均匀分布：

| 文件位置 | mini 平均字符 | mini 中文占比 | 全量平均字符 | 全量中文占比 |
| --- | ---: | ---: | ---: | ---: |
| 0% | 278 | 0.94 | 276 | 0.95 |
| 20% | 1,840 | 1.00 | 144（17% 处） | 0.94 |
| 40% | 417 | 0.98 | 419（34% 处） | 0.98 |
| 60% | 397 | 1.00 | 415（50% 处） | 0.98 |
| 80% | 2,101 | 1.00 | 2,078 | 1.00 |
| 97% | 1,099 | **0.00** | 1,109 | **0.00** |

97% 处中文占比为 0，即两份语料的尾部都是纯英文/代码段；80% 处是长文档段。

**缺陷一：部分 epoch 在有序语料上是有偏采样。** 全量臂 60,000 步只消费 0.34 个 epoch，
配置为 `shuffle_buffer: 0` 即纯顺序读，因此它只训练了文件前 34% —— **一行英文/代码都没
读到，长文档段也未进入**。这直接解释了它的 `domain.english_or_code_heavy` 为何显著更差
（2.5566 与 2.5720，对照 mini 臂的 1.8907 与 1.8762）：该 domain 的数据它根本没见过。

因此"重复数据优于全新数据"这一表面结论**不成立**。两个全量臂结果高度一致
（val 2.0489 / 2.0474），说明这是系统性偏差而非随机波动。该实验对"语料规模"这一问题
无效，必须在全局预打散的视图上重跑。

**缺陷二：两个 seed 看到的是同一份数据的同一顺序。** `JsonlPackedDataset` 仅在
`shuffle_buffer > 1` 时使用 `seed`；DataLoader 以 `shuffle=False` 包装 IterableDataset，
其 generator 不参与定序；dropout 为 0.0。因此 seed `20260511` 与 `20260721` 之间**唯一
的差异是权重初始化**。

这使得"第二 seed 确认"弱于清单意图的完整确认，并解释了确认结果的形态：八项 loss 判据
跨 seed 差仅 0.0005–0.0144，而 MCQ（0.3542 与 0.3281）和生成重复（1 与 3）明显波动——
前者被相同的数据顺序锁定，后者对初始化敏感。已登记的两个候选仍在冻结 holdout 上真实
通过，但"预算是唯一变量"这一解释尚未经过数据顺序扰动的检验。

## 数据管线瓶颈

服务器管理员观察到"单核打满、GPU 90%、整机 CPU 大量空闲"，该判断准确，原因不在模型。

实测：两个训练进程各占约一个核（109% 与 91% CPU），28 核主机其余核心空闲；累计数据
等待 390s 与 342s，约占墙钟 6.5%，与观察到的 GPU 利用率缺口一致。MFU 28.7%–31.3% 对
89,864,448 参数、sequence length 384 的模型属正常水位——该规模受显存带宽限制而非算力
限制，因此"利用率高"是预期结果，不是异常。

根因是数据加载使用 `num_workers=0`。这不是疏忽：精确断点恢复要求 packed-block 游标有
唯一确定的顺序，多 worker 会让"已消费到第几个 block"不可复现。代价是 JSON 解析、
分词和打包全部在训练进程内同步完成，并且每个 epoch 都会把整个语料重新分词一遍——
mini 语料的两个候选跑了 2.27 个 epoch，等于多分词了 1.27 遍。

升级路径：预先分词一次，存为内存映射的 uint16 打包数组（词表 6,400 可用 uint16，
21.6 亿 token 约 4.3 GB），训练时直接切片。这既保留 `num_workers=0` 与绝对游标，又
消除瓶颈。本轮不改：两个臂已跑至 56% 与 64%，为回收 6.5% 而重启得不偿失。标记于
`JsonlPackedDataset` 文档字符串，下一次长训练前执行。

显存方面本轮无需处理：每卡占用 9.7 GB / 23 GB，准入控制在启动前计入外部进程用量并
拒绝抢占，两张卡分别留出 9.5 GB 与 7.2 GB。若他人需要更多余量，batch 32 降至 16 配
梯度累积可将激活显存约减半而吞吐基本不变。

## 未完成

- 第二 seed（`seed=20260721`，同预算）确认中，未完成前候选只能表述为"内部过门并登记"，
  不能表述为预训练完成。
- MCQ v3 题库：模板去重、类别改名或重建、干扰项按真实错误模式重写、类别轴拆分。
- 全量语料臂：数据已就位，环境构建中。
- 主机重启演练受权限阻塞，SIGKILL 已作为替代路径穷尽。
- 服务器工作区收敛（当前两个 workspace）。
