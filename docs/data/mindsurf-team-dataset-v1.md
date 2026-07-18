# MindSurf 团队预训练数据集 v1

## 身份

- Provider：ModelScope
- Dataset：`wyywnab/mindsurf_pretrain_dataset`
- Revision：`ab97cc8ba19d08175e5567bd2816ec850b8bb200`
- 数据维护者：MindSurf 团队
- 与 MiniMind 的关系：使用固定版本的 MiniMind tokenizer；数据本身不是 MiniMind
  官方数据集
- Dataset license 字段：`other`
- 声明 token 数：2,000,004,621

机器可读文件身份、大小和行数位于
`configs/datasets/mindsurf_team_v1.json`。

## Split 构造

发布的 split metadata 记录了以下过程：

- 原始 2,182,066 行；
- 规范化文本去重后 2,181,856 条，移除 210 条重复；
- 使用 seed `20260511`，按 `sha256(seed:text_sha1)` 最小值选择 holdout；
- validation/test 各 2,000 条；
- 训练集 2,177,856 条，并提供预先 shuffle 的训练文件。

审计程序重新验证每个文件的大小和 SHA-256、JSONL schema、行数、规范化文本去重、
source/id 重复、validation/test 的发布哈希集合，以及 train/holdout 交叉污染。

发布的 shuffled train 中有 11 条文本只在 NFKC 规范化后等价。训练管线保留第一次
出现并移除后续等价记录，不改变其余行的顺序。最终训练视图为 2,177,845 行、
6,305,991,620 bytes，SHA-256 为
`37800ee01d5294e3d40765ec686fce7d0c917d7cea79343e57526a3229e8a999`。

## 来源与许可风险

抽样可见的 `source_key` 包括 ChineseWebText2.0、Ultra-FineWeb、
open-web-math-pro、stack-edu、chinese-cosmopedia、BAAI CCI3-HQ、公开技术文档、
MDN 和 EPUB 书籍。该列表用于数据分布审计，不等于逐来源许可证明。

在完成以下材料前，`public_release_license_ready` 必须保持 `false`：

1. 每个 `source_key` 到原始数据集、版本和许可证的映射；
2. 文档与 EPUB 的采集范围、再分发依据和退出机制；
3. 训练后权重发布是否符合所有上游条款的书面审查；
4. 数据卡中可由外部使用者验证的来源和限制。

许可门不阻止内部工程验证，但会独立阻止模型进入 public stage。
