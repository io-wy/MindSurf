"""Unified experiment tracking for W&B and MLflow.

Provides a single interface that delegates to both backends simultaneously.
Either can be disabled by not configuring its API key / URI.
"""

from __future__ import annotations

from typing import Any

import mlflow
import wandb

from python_starter.infrastructure.config import Settings
from python_starter.infrastructure.logging import get_logger

logger = get_logger(__name__)


class ExperimentTracker:
    """Unified tracker that logs to W&B and/or MLflow."""

    def __init__(self, settings: Settings, experiment_name: str | None = None) -> None:
        self.settings = settings
        self.experiment_name = experiment_name or settings.mlflow_experiment_name
        self._wandb_run: wandb.sdk.wandb_run.Run | None = None
        self._mlflow_active = False

    def start(self, run_name: str | None = None, config: dict[str, Any] | None = None) -> None:
        """Initialize tracking backends."""
        # W&B
        if self.settings.wandb_api_key:
            try:
                self._wandb_run = wandb.init(
                    project=self.settings.wandb_project,
                    name=run_name,
                    config=config,
                    reinit=True,
                )
                logger.info("wandb_initialized", project=self.settings.wandb_project)
            except Exception as e:
                logger.warning("wandb_init_failed", error=str(e))

        # MLflow
        if self.settings.mlflow_tracking_uri:
            try:
                mlflow.set_tracking_uri(self.settings.mlflow_tracking_uri)
                mlflow.set_experiment(self.experiment_name)
                mlflow.start_run(run_name=run_name)
                if config:
                    mlflow.log_params(config)
                self._mlflow_active = True
                logger.info("mlflow_initialized", uri=self.settings.mlflow_tracking_uri)
            except Exception as e:
                logger.warning("mlflow_init_failed", error=str(e))

    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters."""
        if self._wandb_run:
            wandb.config.update(params)
        if self._mlflow_active:
            for k, v in params.items():
                mlflow.log_param(k, v)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log scalar metrics."""
        if self._wandb_run:
            wandb.log(metrics, step=step)
        if self._mlflow_active:
            for k, v in metrics.items():
                mlflow.log_metric(k, v, step=step)

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Log a file artifact."""
        if self._wandb_run:
            artifact = wandb.Artifact(name="model-artifacts", type="model")
            artifact.add_file(local_path)
            wandb.log_artifact(artifact)
        if self._mlflow_active:
            mlflow.log_artifact(local_path, artifact_path=artifact_path)

    def finish(self) -> None:
        """Close tracking sessions."""
        if self._wandb_run:
            wandb.finish()
            self._wandb_run = None
        if self._mlflow_active:
            mlflow.end_run()
            self._mlflow_active = False
        logger.info("tracking_finished")
