"""Pydantic schemas for API request/response validation."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class HealthStatus(str, Enum):
    """Health check status values."""

    OK = "ok"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class HealthResponse(BaseModel):
    """Health check response."""

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "status": "ok",
            "version": "0.1.0",
            "timestamp": "2026-04-30T12:00:00Z",
        }
    })

    status: HealthStatus = Field(..., description="Overall service health status")
    version: str = Field(..., description="Application version")
    timestamp: datetime = Field(..., description="Response timestamp")
    services: dict[str, bool] = Field(
        default_factory=dict,
        description="Individual service health (database, redis, etc.)",
    )


class InferenceRequest(BaseModel):
    """Model inference request."""

    text: str = Field(..., min_length=1, description="Input text for inference")
    max_length: int = Field(default=128, ge=1, le=2048, description="Maximum output length")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Nucleus sampling parameter")


class InferenceResponse(BaseModel):
    """Model inference response."""

    text: str = Field(..., description="Generated text")
    input_tokens: int = Field(..., description="Number of input tokens")
    output_tokens: int = Field(..., description="Number of output tokens")
    generation_time_ms: float = Field(..., description="Generation time in milliseconds")


class ExperimentStatus(str, Enum):
    """Experiment lifecycle status."""

    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ExperimentCreate(BaseModel):
    """Create experiment request."""

    name: str = Field(..., min_length=1, max_length=255, description="Experiment name")
    description: str | None = Field(default=None, description="Experiment description")
    model_config_snapshot: dict = Field(
        default_factory=dict, description="Model configuration snapshot"
    )
    training_config_snapshot: dict = Field(
        default_factory=dict, description="Training configuration snapshot"
    )


class ExperimentResponse(BaseModel):
    """Experiment detail response."""

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(..., description="Experiment ID")
    name: str = Field(..., description="Experiment name")
    description: str | None = Field(default=None, description="Experiment description")
    status: ExperimentStatus = Field(..., description="Current status")
    model_config_snapshot: dict = Field(default_factory=dict)
    training_config_snapshot: dict = Field(default_factory=dict)
    metrics: dict = Field(default_factory=dict, description="Aggregated metrics")
    created_at: datetime = Field(...)
    updated_at: datetime = Field(...)


class ExperimentListResponse(BaseModel):
    """Paginated experiment list response."""

    items: list[ExperimentResponse] = Field(...)
    total: int = Field(..., description="Total number of experiments")
    page: int = Field(...)
    page_size: int = Field(...)


class ModelRegisterRequest(BaseModel):
    """Register a trained model request."""

    experiment_id: int = Field(..., description="Associated experiment ID")
    name: str = Field(..., description="Model name/version")
    artifact_path: str = Field(..., description="Path to model artifacts")
    metrics: dict = Field(default_factory=dict, description="Model evaluation metrics")
    parameters: dict = Field(default_factory=dict, description="Model hyperparameters")


class ModelResponse(BaseModel):
    """Registered model response."""

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(...)
    experiment_id: int = Field(...)
    name: str = Field(...)
    artifact_path: str = Field(...)
    metrics: dict = Field(default_factory=dict)
    parameters: dict = Field(default_factory=dict)
    registered_at: datetime = Field(...)


class TrainingJobRequest(BaseModel):
    """Submit a training job request."""

    experiment_name: str = Field(..., description="Name for the new experiment")
    config_overrides: dict = Field(
        default_factory=dict, description="Hydra config overrides"
    )
    dataset_path: str | None = Field(default=None, description="Path to dataset")


class TrainingJobResponse(BaseModel):
    """Training job submission response."""

    job_id: str = Field(..., description="Celery task ID")
    experiment_id: int | None = Field(default=None)
    status: str = Field(..., description="Job status")
    message: str = Field(...)
