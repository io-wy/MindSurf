# 外部锚点：端口成立，锚点作废（2026-07-22）

**结论：M1 的外部锚点无法用任何当前已发布的上游 checkpoint 建立，如实降级为
三支柱。** 降级的原因不是"做不动"，而是一个可量化的事实：上游发布的权重与本项目
用的**不是同一个 tokenizer**，per-token loss 因此不可比。模型端口本身已证明逐位
忠实——正是这个忠实的端口让作废原因可以被测出来，而不是被猜出来。

机器可读证据：`artifacts/evaluation/anchor_upstream_void.json`。

---

## 1. 锚点来源

ModelScope `gongjy/MiniMind2-PyTorch` @ master：

| 文件 | 大小 | SHA-256 | 参数量 |
| --- | ---: | --- | ---: |
| `pretrain_512.pth` | 58,238,592 | `d6b9868ae7626cf14eea85fd4c94a90ba2a92f065ab308fe80ff42c691f91dba` | 25,829,888 |
| `pretrain_768.pth` | 217,941,823 | `cf624cb8f26a4e7bfa474338bbb72a2551e9892e095420864807632420706cfb` | 104,030,976 |

参数量与上游宣称的 26M / 104M 一致。

## 2. 端口忠实性：逐位证明，不是"看起来合理"

上游发布的权重**没有 QK norm 权重**，而本项目的 `CausalSelfAttention` 有
`q_norm`/`k_norm`。把上游权重塞进带未训练 QK norm 的模型会改变函数，产出的
loss 是假的。因此新增 `ModelConfig.qk_norm`（默认 `True`），为 `False` 时两个
norm 变成 `nn.Identity`——**不是空操作开关而是不产生参数**，这样键不匹配会响亮
失败，而不是静默地让训练好的权重穿过未训练的 norm。

证明方法：取上游自己的 `model/model_minimind.py` @ `83e52f6`（`101d7df`
"minimind-3" 加入 QK norm 之前的最后一版，即与这批权重匹配的代码），原生加载，
与本端口在同一组随机 token id 上比对 logits（batch 2 × seq 128，float32，
seed 20260721）：

| checkpoint | 最大绝对 logit 差 |
| --- | ---: |
| `pretrain_512` | **0.0** |
| `pretrain_768` | **0.0** |

容差设为 1e-4，实测为精确相等。逐项核对而非假定的架构事实：half-split NeoX
RoPE 且 cos/sin 构造一致、`rope_theta` 1e6、pre-norm 块接线、SwiGLU
（`gate/up/down` 对 `w1/w2/w3`）、GQA 8 头 / 2 KV 头且 KV 连续重复、embedding
逐位绑定。一个陷阱被抓到：上游 `rms_norm_eps` 是 **1e-5**（其 `config.json`），
不是本项目默认的 1e-6；照默认值加载会得到一个安静走偏的模型。

## 3. 锚点作废：tokenizer 不同源

| | SHA-256 |
| --- | --- |
| 本项目 tokenizer | `71f32c68cf63a15355a8fc171b7594b3d41870fe0ddb54fc6aefa55f73a4a668` |
| 锚点权重所用 tokenizer | `d98595c6aef70d95f72748582fb9b4f53d76dd58c1ae1dd702ad7c84e1caf5e4` |

确立方式（每步都可复核）：

1. 本项目 `data/raw/minimind_official_v1/tokenizer/tokenizer.json` 哈希
   `71f32c68`，与上游 master 的 `model/tokenizer.json` 逐位相同。
2. ModelScope `gongjy/MiniMind2`（与锚点同源仓库的 HF 格式姊妹仓）的
   `tokenizer.json` 为 `d98595c6`（439,547 字节），直接下载并哈希。
3. `d98595c6` 与上游 `83e52f6` 的 `model/tokenizer.json` 逐位相同。
4. `101d7df` 同时替换了模型（加 QK norm）与 tokenizer；锚点没有 QK norm 权重，
   因此位于该提交之前。
5. ModelScope 上不存在 MiniMind3 的 checkpoint 发布（`record not found`），
   所以**没有任何已发布的上游权重与本项目 tokenizer 配对**。

两个 tokenizer 都是 6,400 词表，但映射不同：1,772 个 token 互不存在，4,619 个
两边都有但 id 不同，merge 规则 6,141 对 6,108。

### 实测后果

在本项目 strict_val、本项目管线（`packed_loss`，max_length 384，batch 32，
250 batch）上：

| 锚点 | 用本项目 tokenizer | 用其自身 tokenizer |
| --- | ---: | ---: |
| 26M | **12.5733** | 2.9716 |
| 104M | **12.0227** | 2.7469 |

6,400 词表均匀猜测的 loss 是 **8.7641**。用本项目 tokenizer 时两个锚点都**比
随机猜还差**——这是"自信地答错"，正是 token id 被重映射的定量特征。

**若不核对 tokenizer，这次评测会得出"本项目 89M 候选 1.7248 优于上游 104M 的
12.02"，而这十 nats 的差距完全由 tokenizer 错配制造。** 因此
`scripts/evaluate_external_anchor.py` 的 tokenizer 闸门**失败即拒且不写任何
产物**。

自身 tokenizer 那一列也**不能**当替代锚点：per-token 交叉熵只在固定分词下可比，
同一份 2k holdout 在锚点 tokenizer 下产生 559,488 token、在本项目 tokenizer 下
540,672 token，2.75 与 1.7248 不在同一标尺上。换成 bits-per-character 是**新建
一个仪器**，要走清单 §2 的全套效度与噪声底，不是本项目现成的标尺。

## 4. 处置

- **不报告任何方向的比较。** 锚点作废，候选与锚点之间既不声称优也不声称劣。
- **M1 如实降级为三支柱**：跨源泛化、生成健康、下游可用性（后者留待 SFT）。
  发布说明中所有结论都相对本项目历史最优参照，属内部纵向比较，其局限已在
  发布说明明示。
- **端口与闸门保留**（`scripts/evaluate_external_anchor.py`，9 项测试）。上游若
  发布基于当前 tokenizer 训练的权重，锚点就是一条命令的事。
- 未确立的事项如实记录：因为没有做任何比较，本轮**未确认**噪声底是否适用于
  锚点式比较。

## 5. 一个副作用与它的取舍

`ModelConfig.to_dict()` 在 `qk_norm` 取默认值时**省略该键**。原因：
`trainer.py` 恢复时对 `checkpoint["model_config"]` 与 `config.to_dict()` 做
**精确字典相等**比较，无条件写出新键会让此前所有 checkpoint 无法恢复。省略是
向后兼容的最小代价，代价是序列化配置不显式列出该默认值；`qk_norm=False` 时键
照常写出，所以锚点 checkpoint 的配置里能看到它。已用真实候选验证：存储配置无
`qk_norm`，往返相等，加载后 89,864,448 参数且 QK norm 在位；锚点走同一加载器
得到 104,030,976 参数且无 QK norm。
