# Repository Guidelines

## Project Structure & Module Organization

- `src/python_starter/core/` — ML core: model architectures, training loops, datasets, tokenizers.
- `src/python_starter/api/` — FastAPI application: routers, schemas, dependencies, ORM models.
- `src/python_starter/experiments/` — Experiment tracking wrappers (W&B, MLflow) and model registry.
- `src/python_starter/tasks/` — Celery async tasks for offloading long-running training jobs.
- `src/python_starter/infrastructure/` — Config, database, Redis, logging (shared across all layers).
- `scripts/` — CLI entry points for training, inference, evaluation, and preprocessing.
- `configs/` — Hydra YAML configurations for models, training, and data.
- `tests/` — pytest suite with fixtures for DB/Redis mocking.
- Root configs: `pyproject.toml`, `alembic.ini`, `docker-compose.yml`, `dvc.yaml`.

## Build, Test, and Development Commands

- `uv sync --extra dev` — Install all dependencies including dev tools.
- `uv run uvicorn python_starter.api.main:app --reload` — Start API dev server.
- `uv run pytest` — Run full test suite.
- `uv run pytest --cov-report=html` — Run tests with HTML coverage report.
- `uv run ruff check src tests scripts` — Lint code.
- `uv run ruff format src tests scripts` — Format code.
- `uv run mypy src tests scripts` — Type check.
- `uv run alembic upgrade head` — Run database migrations.
- `uv run celery -A python_starter.tasks.celery_app worker --loglevel=info` — Start Celery worker.
- `docker compose up -d postgres redis mlflow` — Start development services.
- `uv run scripts/train.py training=pretrain model=minimind` — Run training with Hydra.

## Coding Style & Naming Conventions

- Language: Python 3.11+ with strict type hints (`from __future__ import annotations`).
- Linting: `ruff` is the source of truth; keep code warning-free. `mypy --strict` for type checking.
- Imports: `isort` style (ruff handles it). Group: stdlib → third-party → first-party (`python_starter`).
- Modules: snake_case filenames. Packages: lowercase.
- Classes: PascalCase. Functions/variables: snake_case. Constants: UPPER_SNAKE_CASE.
- Private internals: prefix with underscore (`_internal_fn`).
- Avoid mutable defaults; prefer immutable data structures.

## ML Code Guidelines

- Model configs live in `configs/model/*.yaml`; training configs in `configs/training/*.yaml`.
- Use `ModelConfig` dataclass for model hyperparameters; pass config objects, not raw kwargs.
- Training scripts must be Hydra-driven (`@hydra.main`) for reproducible experiments.
- Always set random seeds via `set_seed()` for reproducibility.
- Log metrics to both W&B and MLflow via `ExperimentTracker`.
- Save checkpoints with `Trainer.save_checkpoint()`; never modify checkpoints in-place.

## Testing Guidelines

- Test files: `tests/test_*.py`. Co-locate complex integration tests in `tests/`.
- Fixtures in `tests/conftest.py`: mocked DB (aiosqlite), fake Redis (fakeredis), API client.
- Minimum coverage threshold: 80% (enforced by CI).
- Mock external services (W&B, MLflow) in unit tests.
- Use `pytest.mark.slow` for integration tests that require real services.

## Commit & Pull Request Guidelines

- Prefer Conventional Commits: `feat:`, `fix:`, `docs:`, `refactor:`, `chore:`, `ci:`, `test:`.
- Link issues in the footer: `Closes #123`.
- PRs should include: brief scope/intent, validation steps, and pass `uv run ruff check && uv run mypy && uv run pytest`.
- Keep changes focused; avoid unrelated refactors.

## Security & Configuration Tips

- Use `.env` for secrets; never commit `.env*` or `secrets/` files.
- `SECRET_KEY` must be changed in production (min 32 chars).
- Database and Redis connections are fail-open: the API starts even if dependencies are unavailable.
- JWT tokens (if added later) should use `SECRET_KEY` with HS256.
- Docker: use non-root user in production images.
