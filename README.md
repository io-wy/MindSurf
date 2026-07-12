# Python ML Starter

A production-ready Python starter for neural network development, inspired by [minimind](https://github.com/jingyaogong/minimind). It provides a complete engineering foundation for training, evaluating, and serving transformer-based language models.

**Highlights:**
- Hydra-driven configuration for reproducible experiments
- Unified experiment tracking (W&B + MLflow) with graceful degradation
- FastAPI inference service with fail-open DB/Redis
- Async training jobs via Celery
- Full CI/CD with ruff, mypy, pytest, and Docker multi-platform builds
- CPU/GPU dual Dockerfile variants

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Package Manager | [uv](https://docs.astral.sh/uv/) |
| ML Framework | PyTorch 2.x |
| API | FastAPI + Uvicorn |
| Database | PostgreSQL + SQLAlchemy 2.0 (async) + Alembic |
| Cache/Queue | Redis + Celery |
| Config | Hydra + pydantic-settings |
| Experiment Tracking | Weights & Biases + MLflow |
| Data Versioning | DVC |
| Code Quality | ruff + mypy + pre-commit |
| Testing | pytest + coverage (80% threshold) |
| Containers | Docker + docker-compose (CPU/GPU) |

---

## First-Time Setup

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Docker & Docker Compose (optional, for services)

### 1. Clone & Configure Environment

```bash
cd python_starter
cp .env.example .env
# Edit .env with your settings (see Environment Variables below)
```

### 2. Install Dependencies

```bash
uv sync --extra dev
```

This creates a `.venv` virtual environment and installs all production + dev dependencies. The lock file `uv.lock` ensures reproducible installs across machines.

### 3. Start Development Services

```bash
docker compose up -d postgres redis mlflow
```

This starts:
- **PostgreSQL** on port `5432`
- **Redis** on port `6379`
- **MLflow UI** on port `5000`

### 4. Run Database Migrations

```bash
uv run alembic upgrade head
```

Migrations are stored in `alembic/versions/`. To create a new migration after changing ORM models:

```bash
uv run alembic revision --autogenerate -m "describe change"
```

### 5. Start API Server

```bash
uv run uvicorn python_starter.api.main:app --reload
```

Visit http://localhost:8000/docs for interactive API documentation.

### 6. Start Celery Worker (optional, for async training)

```bash
uv run celery -A python_starter.tasks.celery_app worker --loglevel=info
```

---

## VS Code Development Environment

This starter includes a fully configured `.vscode/` directory. Open the project in VS Code and it will prompt you to install recommended extensions.

### Recommended Extensions

| Extension | Purpose |
|-----------|---------|
| `ms-python.python` | Python language support |
| `ms-python.debugpy` | Debugger |
| `charliermarsh.ruff` | Linting and formatting (replaces black/isort/flake8/pylint) |
| `ms-python.mypy-type-checker` | Type checking |
| `redhat.vscode-yaml` | YAML validation |
| `tamasfe.even-better-toml` | TOML editing |
| `GitHub.vscode-github-actions` | GitHub Actions workflow support |
| `ms-azuretools.vscode-docker` | Docker integration |

### Workspace Settings

These settings are automatically applied when you open the project:

| Setting | What It Does |
|---------|-------------|
| `python.defaultInterpreterPath` | Points to `.venv` so VS Code finds the uv-managed environment |
| `python.analysis.extraPaths` | Adds `src/` to Python path for import resolution |
| `python.analysis.typeCheckingMode` | Enables basic type checking inline |
| `editor.formatOnSave` | Auto-formats on every save |
| `editor.defaultFormatter` | Uses ruff for all files |
| `editor.codeActionsOnSave` | Auto-fixes ruff issues and organizes imports on save |
| `files.exclude` | Hides cache dirs (`__pycache__`, `.pytest_cache`, `wandb`, `mlruns`, etc.) |
| `search.exclude` | Excludes `.venv`, `uv.lock`, `coverage.xml` from search |
| `python.testing.pytestEnabled` | Enables pytest test discovery in the Testing panel |
| `yaml.schemas` | Validates GitHub Actions workflow YAML against schema |

### Debug Configurations

Press `F5` to launch any of these:

| Configuration | What It Runs |
|---------------|-------------|
| **FastAPI: debug server** | Uvicorn with `--reload` on port 8000 |
| **Python: Current File** | Debug the currently open Python file |
| **Python: Train Script** | `scripts/train.py` with default Hydra args |
| **Python: Inference Script** | `scripts/inference.py` with sample args |
| **Python: Pytest Current File** | Run and debug the current test file |
| **Celery: Worker Debug** | Celery worker with concurrency=1 for easier debugging |

### Tasks

Open Command Palette (`Ctrl+Shift+P`) → `Tasks: Run Task`:

| Task | Command |
|------|---------|
| `uv: sync` | `uv sync --extra dev` |
| `uv: lock` | `uv lock` |
| `ruff: check` | Lint all source code |
| `ruff: format` | Format all source code |
| `mypy: typecheck` | Type check all source code |
| `pytest: all` | Run full test suite |
| `pytest: coverage` | Run tests with HTML + terminal coverage |
| `alembic: migrate` | Create auto-generated migration (prompts for message) |
| `alembic: upgrade` | Apply pending migrations |
| `docker: up` | Start postgres + redis + mlflow |
| `docker: down` | Stop all docker compose services |
| `pre-commit: run all` | Run all pre-commit hooks on all files |

---

## Project Structure

```
python_starter/
├── .github/workflows/        # CI/CD pipelines
│   ├── ci.yml                # Main orchestrator: quality → test → build-docker
│   ├── quality.yml           # ruff + mypy checks
│   ├── test.yml              # pytest with coverage
│   ├── build-docker.yml      # CPU/GPU image builds
│   └── release.yml           # GitHub release on version tags
│
├── .vscode/                  # VS Code workspace configuration
│   ├── extensions.json       # Recommended extensions
│   ├── settings.json         # Workspace settings (formatting, paths, excludes)
│   ├── launch.json           # Debug configurations
│   ├── tasks.json            # Run tasks (lint, test, docker, alembic)
│   └── mcp.json              # MCP server config for VS Code
│
├── configs/                  # Hydra YAML configurations
│   ├── default.yaml          # Default config composition (model + training + data)
│   ├── model/                # Model architectures (minimind, minimind_small)
│   ├── training/             # Training configs (pretrain, sft, dpo)
│   └── data/                 # Dataset configs
│
├── data/                     # Datasets (DVC tracked, .gitignored)
│   ├── raw/                  # Original data
│   └── processed/            # Preprocessed data
│
├── docker/                   # Container definitions
│   ├── Dockerfile.cpu        # CPU-optimized multi-stage build
│   └── Dockerfile.gpu        # CUDA 12.2 base image
│
├── models/                   # Model checkpoints (DVC tracked, .gitignored)
├── notebooks/                # Jupyter notebooks (optional)
├── scripts/                  # CLI entry points
│   ├── train.py              # Unified training (Hydra-driven)
│   ├── inference.py          # Local inference CLI
│   ├── evaluate.py           # Evaluation / perplexity
│   └── preprocess.py         # Data cleaning / filtering
│
├── src/python_starter/       # Main source package (src-layout)
│   ├── api/                  # FastAPI application
│   │   ├── main.py           # App factory + lifespan
│   │   ├── dependencies.py   # FastAPI dependency injection
│   │   ├── models.py         # SQLAlchemy ORM models
│   │   ├── routers/          # API route handlers
│   │   │   ├── health.py     # /health, /health/ready
│   │   │   ├── inference.py  # /inference
│   │   │   └── experiments.py# /experiments CRUD + jobs
│   │   └── schemas/          # Pydantic request/response models
│   │
│   ├── core/                 # ML core code
│   │   ├── model.py          # TransformerLM (decoder-only, RoPE, SwiGLU)
│   │   ├── trainer.py        # Generic training loop (AMP, cosine schedule)
│   │   ├── dataset.py        # TextDataset, SFTDataset
│   │   ├── tokenizer.py      # HuggingFace tokenizer wrapper
│   │   └── utils.py          # Seed, device detection, param counting
│   │
│   ├── experiments/          # Experiment tracking
│   │   ├── tracker.py        # Unified W&B + MLflow wrapper
│   │   └── registry.py       # MLflow model registry operations
│   │
│   ├── tasks/                # Celery async tasks
│   │   ├── celery_app.py     # Celery instance configuration
│   │   └── training.py       # Async training job task
│   │
│   └── infrastructure/       # Shared infrastructure
│       ├── config.py         # pydantic-settings (env vars + .env)
│       ├── database.py       # Async PostgreSQL (SQLAlchemy 2.0, fail-open)
│       ├── redis_client.py   # Redis client (fail-open)
│       └── logging.py        # structlog configuration
│
├── tests/                    # pytest suite
│   ├── conftest.py           # Fixtures (mock DB, fake Redis, API client)
│   ├── test_api.py           # API endpoint tests
│   ├── test_model.py         # Model architecture tests
│   └── test_trainer.py       # Trainer tests
│
├── alembic/                  # Database migrations
│   ├── env.py                # Async migration environment
│   └── versions/             # Generated migration scripts
│
├── .env.example              # Environment variable template
├── .gitignore                # Git ignore rules
├── .dockerignore             # Docker build context exclusions
├── .dvcignore                # DVC ignore rules
├── .pre-commit-config.yaml   # Pre-commit hooks (ruff, trailing-whitespace)
├── .mcp.json                 # MCP server configuration
├── AGENTS.md                 # Repository guidelines for AI agents
├── pyproject.toml            # uv dependencies + ruff + mypy + pytest config
├── alembic.ini               # Alembic configuration
├── docker-compose.yml        # Dev services (postgres, redis, mlflow, api, worker)
├── docker-compose.gpu.yml    # GPU overlay for docker-compose
├── dvc.yaml                  # DVC pipeline (preprocess → train → evaluate)
└── README.md                 # This file
```

### Key Configuration Files

| File | Purpose | Developer Notes |
|------|---------|-----------------|
| `pyproject.toml` | Single source of truth for dependencies, tool configs (ruff, mypy, pytest, coverage) | Edit this to add packages or change tool settings |
| `.env` / `.env.example` | Runtime secrets and settings (DB URLs, API keys, ports) | Copy `.env.example` → `.env` and customize |
| `configs/*.yaml` | Hydra configuration for training experiments | Override via CLI: `training.learning_rate=1e-4` |
| `alembic.ini` | Database migration settings | Auto-generated; rarely needs manual edits |
| `dvc.yaml` | Data version control pipeline stages | Run `dvc repro` to execute the full pipeline |
| `.pre-commit-config.yaml` | Git hooks that run before each commit | Install with `uv run pre-commit install` |
| `.mcp.json` | MCP (Model Context Protocol) server config | Enables AI tools to interact with the codebase |

---

## Environment Variables

Copy `.env.example` to `.env` and customize:

| Variable | Default | Description |
|----------|---------|-------------|
| `ENV` | `development` | `development` / `test` / `production` |
| `DEBUG` | `false` | Enable debug logging |
| `API_HOST` | `0.0.0.0` | FastAPI bind address |
| `API_PORT` | `8000` | FastAPI port |
| `POSTGRES_URL` | `postgresql+asyncpg://dev:dev@localhost:5432/appdb` | Async PostgreSQL connection |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection |
| `CELERY_BROKER_URL` | `redis://localhost:6379/1` | Celery message broker |
| `CELERY_RESULT_BACKEND` | `redis://localhost:6379/2` | Celery result store |
| `WANDB_PROJECT` | `python-starter` | Weights & Biases project name |
| `WANDB_API_KEY` | — | W&B API key (optional) |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server URI |
| `MLFLOW_EXPERIMENT_NAME` | `default` | Default MLflow experiment |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device selection |
| `DEFAULT_DEVICE` | `auto` | `auto` / `cuda` / `cpu` / `mps` |
| `SECRET_KEY` | *(dev default)* | JWT/signing key (change in production!) |

---

## Coding Style & Naming Conventions

- **Language:** Python 3.11+ with strict type hints (`from __future__ import annotations` at the top of every file).
- **Linting:** `ruff` is the source of truth; keep code warning-free. `mypy --strict` for type checking.
- **Imports:** `isort` style (handled by ruff). Group order: stdlib → third-party → first-party (`python_starter`).
- **Modules:** `snake_case` filenames. Packages: lowercase.
- **Classes:** `PascalCase`. Functions/variables: `snake_case`. Constants: `UPPER_SNAKE_CASE`.
- **Private internals:** prefix with underscore (`_internal_fn`).
- **Avoid mutable defaults;** prefer immutable data structures.

---

## ML Development Guidelines

- **Model configs** live in `configs/model/*.yaml`; training configs in `configs/training/*.yaml`.
- Use `ModelConfig` dataclass for model hyperparameters; pass config objects, not raw kwargs.
- Training scripts must be **Hydra-driven** (`@hydra.main`) for reproducible experiments.
- Always set random seeds via `set_seed()` for reproducibility.
- Log metrics to both W&B and MLflow via `ExperimentTracker`.
- Save checkpoints with `Trainer.save_checkpoint()`; never modify checkpoints in-place.

---

## Testing Guidelines

- Test files: `tests/test_*.py`.
- Fixtures in `tests/conftest.py`: mocked DB (aiosqlite), fake Redis (fakeredis), API client.
- **Minimum coverage threshold: 80%** (enforced by CI).
- Mock external services (W&B, MLflow) in unit tests.
- Use `pytest.mark.slow` for integration tests that require real services.

---

## Commit & Pull Request Guidelines

- Prefer **Conventional Commits**: `feat:`, `fix:`, `docs:`, `refactor:`, `chore:`, `ci:`, `test:`.
- Link issues in the footer: `Closes #123`.
- PRs should include: brief scope/intent, validation steps, and pass `uv run ruff check && uv run mypy && uv run pytest`.
- Keep changes focused; avoid unrelated refactors.

---

## Training

### Pretraining

```bash
uv run scripts/train.py training=pretrain model=minimind data=default
```

### Supervised Fine-Tuning

```bash
uv run scripts/train.py training=sft model=minimind_small data=default
```

### Custom Overrides

```bash
uv run scripts/train.py training=pretrain model=minimind data=default \
    training.num_epochs=5 training.learning_rate=1e-4
```

### Via Celery (Async)

Submit a training job through the API:

```bash
curl -X POST http://localhost:8000/experiments/jobs \
  -H "Content-Type: application/json" \
  -d '{"experiment_name": "my-exp", "config_overrides": {"training.num_epochs": 3}}'
```

---

## Inference

```bash
uv run scripts/inference.py \
    --checkpoint models/checkpoints/pretrain/final_model.pt \
    --prompt "Once upon a time" \
    --max-length 128 \
    --temperature 0.7 \
    --top-p 0.9
```

Or use the API:

```bash
curl -X POST http://localhost:8000/inference \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "max_length": 32}'
```

---

## Evaluation

```bash
uv run scripts/evaluate.py \
    --checkpoint models/checkpoints/pretrain/final_model.pt \
    --data data/raw/test.txt
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness probe |
| `/health/ready` | GET | Readiness probe (DB + Redis status) |
| `/inference` | POST | Run model inference |
| `/inference/models` | GET | List available models |
| `/experiments` | POST | Create experiment |
| `/experiments` | GET | List experiments (paginated) |
| `/experiments/{id}` | GET | Get experiment detail |
| `/experiments/{id}/status` | PATCH | Update experiment status |
| `/experiments/{id}/metrics` | POST | Update experiment metrics |
| `/experiments/{id}/models` | POST | Register a trained model |
| `/experiments/{id}/models` | GET | List models for an experiment |
| `/experiments/jobs` | POST | Submit async training job |
| `/experiments/jobs/{id}` | GET | Get training job status |

---

## Development Commands

```bash
# Code quality
uv run ruff check src tests scripts
uv run ruff format src tests scripts
uv run mypy src tests scripts

# Tests
uv run pytest
uv run pytest --cov-report=html   # Open htmlcov/index.html
uv run pytest -m "not slow"       # Skip slow integration tests

# Pre-commit hooks
uv run pre-commit install
uv run pre-commit run --all-files

# Database migrations
uv run alembic revision --autogenerate -m "Add new table"
uv run alembic upgrade head
uv run alembic downgrade -1
```

---

## Docker

### CPU Mode

```bash
docker compose up --build
```

### GPU Mode

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

### Services Overview

The `docker-compose.yml` defines these services:

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| `postgres` | postgres:17-alpine | 5432 | Application database |
| `redis` | redis:7-alpine | 6379 | Cache + Celery broker |
| `mlflow` | ghcr.io/mlflow/mlflow | 5000 | Experiment tracking UI |
| `api` | Dockerfile.cpu | 8000 | FastAPI server |
| `worker` | Dockerfile.cpu | — | Celery training worker |

---

## Security & Configuration Tips

- Use `.env` for secrets; never commit `.env*` or `secrets/` files.
- `SECRET_KEY` must be changed in production (min 32 chars).
- Database and Redis connections are **fail-open**: the API starts even if dependencies are unavailable, printing warnings and degrading gracefully.
- Docker: use non-root user in production images.

---

## License

MIT
