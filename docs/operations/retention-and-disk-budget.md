# 保留策略与磁盘预算

训练主机 `mindsurf` 的根文件系统只有 248 GB，一次 80M 单卡训练就能占掉两位数
GB，所以每一类产物都必须有明确的保留期限和删除条件。本文件是这些规则的事实入口。

## 实测尺寸基线

| 产物 | 单份大小 | 依据 |
| --- | ---: | --- |
| `final_model.pt`（含 optimizer、scheduler、AMP、RNG、游标） | 1.08 GB | `minimind_official_v1_80m/final_model.pt`，89,864,448 参数 |
| rolling checkpoint | 1.08 GB | 与 final 同结构 |
| Official 训练视图 JSONL | 1.24 GB | `pretrain_train_nfkc_dedup.jsonl` |
| Official 原始数据 | 2.4 GB | `data/raw/minimind_official_v1/` |
| Team 原始数据 | 5.9 GB | `data/raw/mindsurf_team_v1/` |
| 单个 run 的 tracking 日志 | < 1 MB | `metrics.jsonl` |
| 评测产物 JSON | < 5 MB | `artifacts/evaluation/` |

## 各类产物的保留策略

**rolling checkpoint**：`training.checkpoint_keep_last = 2`，训练器在每次保存后自动
删除更早的 step checkpoint。恢复演练只需要最近一个 rolling checkpoint，保留 2 份是
为了容忍一次写入损坏。

**正式候选的 final checkpoint**：无限期保留，且必须存在经反查的备份副本
（`scripts/backup_assets.py`）。删除条件：已被判定为废弃，结论写入
`docs/experiments/`，且备份副本仍在。

**失败 pilot 的 checkpoint**：与正式候选区分。pilot 的价值在其结论与逐项评测
产物，权重本身被否证后无未来用途；因此在结论写入 `docs/experiments/`、评测
JSON 保留的前提下即可删除，不要求备份副本——在同一磁盘上复制一份不增加任何
耐久性，只消耗被守卫保护的空间。2026-07-21 依此删除三个 targeted pilot 目录，
释放 12.3 GB。

**演练产物**：故障注入和恢复演练的 checkpoint 在演练证据 JSON
（`artifacts/infra/training_recovery_drill*.json`，含 SHA-256 与逐项一致性检查）
写入后即可删除。演练可从冻结配方重跑，证据不可重建，所以留证据不留权重。
2026-07-20 依此清理了三个演练目录，释放 33 GB。

**训练视图与原始数据**：训练视图保留，因为它是 manifest SHA-256 的对象。原始上游
语料可删——`configs/datasets/index.json` 固定了 `dataset_id` 与 `revision`，可按身份
重新拉取，所以它不进备份清单。

**日志与 tracking**：随 run 目录保留，体积可忽略。

**缓存**：`__pycache__`、`.pytest_cache`、HuggingFace 下载缓存随时可删，不作为证据。

## 训练前空间预算

`scripts/preflight_training.py --min-free-bytes` 在 checkpoint 目标目录上强制一个
下限，默认 15 GiB，并把 `free_bytes` 与 `min_free_bytes` 一起写进 preflight 产物。

预算的推导方式：

```
预算 = (checkpoint_keep_last + 1) x checkpoint 大小 + 备份副本 + 余量
     = 3 x 1.08 GB + 1.08 GB + 余量
```

当前配方约需 4.4 GB，15 GiB 默认值留出了三倍余量，同时容纳并行评测与日志。换更大
模型或调高 `checkpoint_keep_last` 时必须显式传 `--min-free-bytes`，不要沿用默认值。

## 运行中的磁盘预警

`scripts/watch_training.py` 每轮轮询 checkpoint 所在文件系统的剩余空间，低于
`--free-bytes-min`（默认 20 GB）时产出 `disk_low` 告警并写入证据 JSON。预警阈值高于
preflight 下限，目的是在训练中途逼近下限之前就暴露问题。

## 不适用的路径

`/data/cubelet` 是 root 所有的容器运行时状态目录（`cubelet.sock`、`storage`），
不是可用数据盘，任何备份或缓存都不得写入。
