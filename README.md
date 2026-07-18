# MindSurf

MindSurf 是团队维护的小模型预训练、评测与服务工程。当前仓库以
decoder-only Transformer 为核心，提供可复现配置、训练与评测 CLI、
实验追踪、异步任务和 FastAPI 服务骨架。

项目仍处于工程化与实验验证阶段。任何模型只有在数据身份、严格评测、
能力门和发布门全部通过后，才能进入部署链路。

## 当前能力

- PyTorch decoder-only Transformer：RoPE、RMSNorm、SwiGLU、GQA/MHA；
- Hydra 配置驱动的预训练、SFT 与 DPO 入口；
- DVC 数据与模型产物布局；
- W&B 与 MLflow 实验记录，外部服务不可用时降级运行；
- 本地推理与困惑度评测 CLI；
- FastAPI 健康检查、推理和实验管理接口；
- Celery 异步训练任务；
- PostgreSQL、Redis、MLflow 的 Docker Compose 开发环境；
- pytest、ruff、mypy、coverage 与 GitHub Actions 质量门。

## 分支纪律

- `main`：稳定分支；
- `pretrain`：预训练方向的集成基线；
- `feature/*`、`fix/*`：短期工作分支。

禁止直接向 `main` 或 `pretrain` 提交。所有改动从最新集成基线创建短期
分支，经测试、审查和 Pull Request 合并；禁止用另一套仓库树整体替换
集成分支。

## 快速开始

要求 Python 3.11+、[uv](https://docs.astral.sh/uv/)；数据库、Redis 和
MLflow 可按需通过 Docker 启动。

```bash
git clone https://github.com/io-wy/MindSurf.git
cd MindSurf
git switch pretrain

uv sync --extra dev
cp .env.example .env
```

启动本地依赖与 API：

```bash
docker compose up -d postgres redis mlflow
uv run alembic upgrade head
uv run uvicorn python_starter.api.main:app --reload
```

API 文档默认位于 `http://localhost:8000/docs`。

## 训练与评测

配置按模型、训练阶段和数据拆分，入口由 Hydra 组合：

```bash
# 预训练
uv run python scripts/train.py training=pretrain model=minimind

# 监督微调
uv run python scripts/train.py training=sft model=minimind

# 本地评测
uv run python scripts/evaluate.py \
  --checkpoint models/checkpoints/latest.pt \
  --data data/processed/val.jsonl

# 本地推理
uv run python scripts/inference.py \
  --checkpoint models/checkpoints/latest.pt \
  --prompt "Hello"
```

训练前应冻结 tokenizer、训练/验证/测试数据身份、随机种子和评测口径。
checkpoint 通过 `Trainer.save_checkpoint()` 写入，禁止原地修改已有权重。

## 项目结构

```text
MindSurf/
├── configs/                     # Hydra 模型、训练和数据配置
├── data/                        # DVC 管理的数据目录
├── models/                      # DVC 管理的模型产物
├── scripts/                     # 训练、推理、评测和预处理入口
├── src/python_starter/
│   ├── core/                    # 模型、数据集、tokenizer、trainer
│   ├── experiments/             # W&B、MLflow 与模型注册
│   ├── api/                     # FastAPI 路由、schema 与 ORM
│   ├── tasks/                   # Celery 异步任务
│   └── infrastructure/          # 配置、数据库、Redis 与日志
├── tests/                       # pytest 测试
├── docs/experiments/            # 已冻结的实验结论
├── dvc.yaml
├── docker-compose.yml
└── pyproject.toml
```

## 已冻结实验

- [MiniMind 80M controlled revalidation](docs/experiments/2026-07-18-minimind-80m-revalidation.md)
  发布了原始严格数据集上的受控对照结果及机器可读摘要。
- 该实验的所有最终候选都未通过完整能力门，因此没有模型获准进入基础设施
  晋级或发布。
- 替换数据集必须建立新的 manifest、哈希、泄漏审计和独立 run identity，
  不得把不同数据身份的指标混入同一比较表。

## 工程质量门

提交前运行：

```bash
uv run ruff check src tests scripts
uv run mypy src tests scripts
uv run pytest
```

Pull Request 应说明变更范围、验证命令、数据与配置身份以及已知限制。
外部服务必须在测试中 mock；需要真实服务或 GPU 的测试分别标记为
`integration`、`slow` 或 `gpu`。

## 配置与安全

- 密钥只放在本地 `.env`，不得提交 `.env`、私钥或 `secrets/`；
- 生产环境必须更换至少 32 字符的 `SECRET_KEY`；
- 数据与 checkpoint 通过 DVC 或受控制品仓库管理，不直接提交大文件；
- 生产容器使用非 root 用户；
- 数据库、Redis 或追踪服务不可用时，应明确记录降级状态，不能把降级运行
  误报为完整验证通过。
