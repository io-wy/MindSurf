"""Experiment management endpoints.

CRUD operations for experiments and model registry.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from python_starter.api.dependencies import DBManagerDep, DBSession, SettingsDep
from python_starter.api.models import Experiment, RegisteredModel
from python_starter.api.schemas.models import (
    ExperimentCreate,
    ExperimentListResponse,
    ExperimentResponse,
    ExperimentStatus,
    ModelRegisterRequest,
    ModelResponse,
    TrainingJobRequest,
    TrainingJobResponse,
)
from python_starter.infrastructure.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("", response_model=ExperimentResponse, status_code=status.HTTP_201_CREATED)
async def create_experiment(
    data: ExperimentCreate,
    db: DBSession,
) -> Experiment:
    """Create a new experiment."""
    experiment = Experiment(
        name=data.name,
        description=data.description,
        status=ExperimentStatus.CREATED.value,
        model_config_snapshot=data.model_config_snapshot,
        training_config_snapshot=data.training_config_snapshot,
    )
    db.add(experiment)
    await db.flush()
    await db.refresh(experiment)

    logger.info("experiment_created", experiment_id=experiment.id, name=experiment.name)
    return experiment


@router.get("", response_model=ExperimentListResponse)
async def list_experiments(
    db: DBSession,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status_filter: ExperimentStatus | None = Query(None, alias="status"),
) -> ExperimentListResponse:
    """List experiments with pagination and optional status filter."""
    query = select(Experiment)
    count_query = select(func.count()).select_from(Experiment)

    if status_filter:
        query = query.where(Experiment.status == status_filter.value)
        count_query = count_query.where(Experiment.status == status_filter.value)

    query = query.order_by(desc(Experiment.created_at))
    query = query.offset((page - 1) * page_size).limit(page_size)

    result = await db.execute(query)
    items = result.scalars().all()

    count_result = await db.execute(count_query)
    total = count_result.scalar_one()

    return ExperimentListResponse(
        items=list(items),
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{experiment_id}", response_model=ExperimentResponse)
async def get_experiment(
    experiment_id: int,
    db: DBSession,
) -> Experiment:
    """Get a single experiment by ID."""
    result = await db.execute(
        select(Experiment).where(Experiment.id == experiment_id)
    )
    experiment = result.scalar_one_or_none()

    if experiment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found",
        )
    return experiment


@router.patch("/{experiment_id}/status")
async def update_experiment_status(
    experiment_id: int,
    status: ExperimentStatus,
    db: DBSession,
) -> ExperimentResponse:
    """Update experiment status (e.g., running -> completed)."""
    result = await db.execute(
        select(Experiment).where(Experiment.id == experiment_id)
    )
    experiment = result.scalar_one_or_none()

    if experiment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found",
        )

    experiment.status = status.value
    logger.info(
        "experiment_status_updated",
        experiment_id=experiment_id,
        new_status=status.value,
    )
    return experiment


@router.post("/{experiment_id}/metrics")
async def update_experiment_metrics(
    experiment_id: int,
    metrics: dict,
    db: DBSession,
) -> ExperimentResponse:
    """Update experiment metrics (e.g., from training loop)."""
    result = await db.execute(
        select(Experiment).where(Experiment.id == experiment_id)
    )
    experiment = result.scalar_one_or_none()

    if experiment is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found",
        )

    current_metrics = dict(experiment.metrics) if experiment.metrics else {}
    current_metrics.update(metrics)
    experiment.metrics = current_metrics

    logger.info(
        "experiment_metrics_updated",
        experiment_id=experiment_id,
        metrics=metrics,
    )
    return experiment


# =============================================================================
# Model Registry
# =============================================================================

@router.post("/{experiment_id}/models", response_model=ModelResponse, status_code=status.HTTP_201_CREATED)
async def register_model(
    experiment_id: int,
    data: ModelRegisterRequest,
    db: DBSession,
) -> RegisteredModel:
    """Register a trained model associated with an experiment."""
    # Verify experiment exists
    exp_result = await db.execute(
        select(Experiment).where(Experiment.id == experiment_id)
    )
    if exp_result.scalar_one_or_none() is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment {experiment_id} not found",
        )

    model = RegisteredModel(
        experiment_id=experiment_id,
        name=data.name,
        artifact_path=data.artifact_path,
        metrics=data.metrics,
        parameters=data.parameters,
    )
    db.add(model)
    await db.flush()
    await db.refresh(model)

    logger.info(
        "model_registered",
        model_id=model.id,
        experiment_id=experiment_id,
        name=model.name,
    )
    return model


@router.get("/{experiment_id}/models", response_model=list[ModelResponse])
async def list_experiment_models(
    experiment_id: int,
    db: DBSession,
) -> list[RegisteredModel]:
    """List all models registered for an experiment."""
    result = await db.execute(
        select(RegisteredModel)
        .where(RegisteredModel.experiment_id == experiment_id)
        .order_by(desc(RegisteredModel.registered_at))
    )
    return list(result.scalars().all())


# =============================================================================
# Training Jobs
# =============================================================================

@router.post("/jobs", response_model=TrainingJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def submit_training_job(
    data: TrainingJobRequest,
    settings: SettingsDep,
) -> TrainingJobResponse:
    """Submit an asynchronous training job via Celery.

    Returns immediately with a job ID. Poll GET /jobs/{job_id} for status.
    """
    # TODO: Integrate with Celery task submission
    # from python_starter.tasks.training import run_training_task
    # task = run_training_task.delay(...)

    logger.info(
        "training_job_submitted",
        experiment_name=data.experiment_name,
        config_overrides=data.config_overrides,
    )

    return TrainingJobResponse(
        job_id="placeholder-job-id",
        status="queued",
        message=f"Training job '{data.experiment_name}' has been queued",
    )


@router.get("/jobs/{job_id}")
async def get_training_job_status(
    job_id: str,
) -> dict:
    """Get the status of a submitted training job."""
    # TODO: Query Celery task result backend
    return {
        "job_id": job_id,
        "status": "pending",
        "result": None,
    }
