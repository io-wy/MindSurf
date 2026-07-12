"""MLflow model registry operations.

Convenience wrappers for registering, versioning, and promoting models.
"""

from __future__ import annotations

from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from python_starter.infrastructure.config import Settings
from python_starter.infrastructure.logging import get_logger

logger = get_logger(__name__)


class ModelRegistry:
    """Interface to MLflow model registry."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        if settings.mlflow_tracking_uri:
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        self.client = MlflowClient()

    def register_model(
        self,
        model_path: str,
        name: str,
        tags: dict[str, Any] | None = None,
    ) -> str:
        """Register a model artifact to the model registry.

        Args:
            model_path: Local path or run artifact URI.
            name: Registered model name.
            tags: Optional tags to attach.

        Returns:
            Version string of the registered model.
        """
        result = mlflow.register_model(model_uri=model_path, name=name, tags=tags)
        logger.info(
            "model_registered",
            name=name,
            version=result.version,
        )
        return result.version

    def transition_stage(
        self,
        name: str,
        version: str,
        stage: str,
    ) -> None:
        """Transition a model version to a new stage (e.g., Staging -> Production).

        Args:
            name: Registered model name.
            version: Model version.
            stage: Target stage (Staging, Production, Archived).
        """
        self.client.transition_model_version_stage(
            name=name, version=version, stage=stage
        )
        logger.info("model_stage_transitioned", name=name, version=version, stage=stage)

    def get_latest_version(self, name: str, stage: str | None = None) -> str | None:
        """Get the latest version of a registered model.

        Args:
            name: Registered model name.
            stage: Optional stage filter.

        Returns:
            Version string or None if not found.
        """
        try:
            if stage:
                versions = self.client.get_latest_versions(name, stages=[stage])
            else:
                versions = self.client.search_model_versions(f"name='{name}'")
            if versions:
                return str(max(int(v.version) for v in versions))
        except Exception as e:
            logger.warning("get_latest_version_failed", name=name, error=str(e))
        return None

    def load_model(self, name: str, version: str | None = None, stage: str | None = None) -> Any:
        """Load a registered model for inference.

        Args:
            name: Registered model name.
            version: Specific version. If None, uses stage.
            stage: Stage to load from (e.g., "Production").

        Returns:
            Loaded model object.
        """
        if version:
            model_uri = f"models:/{name}/{version}"
        elif stage:
            model_uri = f"models:/{name}/{stage}"
        else:
            latest = self.get_latest_version(name)
            if latest is None:
                raise ValueError(f"No versions found for model {name}")
            model_uri = f"models:/{name}/{latest}"

        logger.info("loading_registered_model", uri=model_uri)
        return mlflow.pyfunc.load_model(model_uri)
