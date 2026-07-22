"""MLflow model registry operations.

Convenience wrappers for registering, versioning, and promoting models.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import mlflow
from mlflow.tracking import MlflowClient

from python_starter.core.data_contract import sha256_file, write_json_atomic
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
        return str(result.version)

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
        self.client.transition_model_version_stage(name=name, version=version, stage=stage)
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


class LocalCandidateRegistry:
    """Durable local registry whose stages are controlled by evaluation gates."""

    def __init__(self, path: str | Path = "artifacts/model_registry.json") -> None:
        self.path = Path(path)

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": 1, "models": []}
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if value.get("schema_version") != 1 or not isinstance(value.get("models"), list):
            raise ValueError(f"invalid local model registry: {self.path}")
        return cast(dict[str, Any], value)

    def register(
        self,
        *,
        name: str,
        checkpoint_path: str | Path,
        evaluation_path: str | Path,
        training_summary_path: str | Path | None = None,
        preflight_path: str | Path | None = None,
        verdict_path: str | Path | None = None,
        limitations: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Register an internally passing candidate with immutable identities."""
        checkpoint = Path(checkpoint_path).resolve(strict=True)
        evaluation = Path(evaluation_path).resolve(strict=True)
        evaluation_data = json.loads(evaluation.read_text(encoding="utf-8"))
        registry = self._read()
        record: dict[str, Any] = {
            "name": name,
            "stage": "candidate",
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "evaluation_path": str(evaluation),
            "evaluation_sha256": sha256_file(evaluation),
            "registered_at": datetime.now(UTC).isoformat(),
        }

        if verdict_path is None:
            # The gate embedded in the evaluation is whichever gate was current
            # when the evaluation ran.
            gate = evaluation_data.get("gate", {})
            if gate.get("internal_candidate_passed") is not True:
                raise ValueError("only an internally passing candidate can be registered")
            record["gate_source"] = "evaluation"
            record["public_release_eligible"] = gate.get("public_release_passed") is True
        else:
            # A standalone verdict re-judges an existing evaluation under a
            # later gate, so the two have to be checked as a pair: a passing
            # verdict for some other evaluation says nothing about this
            # candidate. Release eligibility is not a gate v3 concept -- it is
            # a license decision, made and recorded outside this registry.
            verdict = Path(verdict_path).resolve(strict=True)
            verdict_data = json.loads(verdict.read_text(encoding="utf-8"))
            judged = verdict_data.get("evaluation")
            if not isinstance(judged, str) or Path(judged).name != evaluation.name:
                raise ValueError(f"verdict does not judge this evaluation: {judged!r}")
            if verdict_data.get("passed") is not True:
                raise ValueError("only an internally passing candidate can be registered")
            record["gate_source"] = "verdict"
            record["verdict_path"] = str(verdict)
            record["verdict_sha256"] = sha256_file(verdict)
            record["gate_config"] = verdict_data.get("gate")
            record["gate_reference"] = verdict_data.get("reference")
            record["public_release_eligible"] = False

        if limitations:
            record["limitations"] = list(limitations)
        # Data identity already reaches the record through the evaluation
        # provenance. The training summary carries the run's seed and consumed
        # tokens, and the preflight record carries the source commit, which is
        # otherwise only ever printed to a log.
        for key, value in (
            ("training_summary", training_summary_path),
            ("preflight", preflight_path),
        ):
            if value is None:
                continue
            resolved = Path(value).resolve(strict=True)
            record[f"{key}_path"] = str(resolved)
            record[f"{key}_sha256"] = sha256_file(resolved)
            if key == "preflight":
                source = json.loads(resolved.read_text(encoding="utf-8")).get("source", {})
                record["source_git_head"] = source.get("git_head")
        if any(
            item.get("checkpoint_sha256") == record["checkpoint_sha256"]
            for item in registry["models"]
        ):
            raise ValueError("checkpoint is already registered")
        registry["models"].append(record)
        write_json_atomic(self.path, registry)
        return record

    def promote_public(self, checkpoint_sha256: str) -> dict[str, Any]:
        """Promote only a candidate whose evaluation includes the license gate."""
        registry = self._read()
        for record in registry["models"]:
            if record.get("checkpoint_sha256") != checkpoint_sha256:
                continue
            if record.get("public_release_eligible") is not True:
                raise ValueError("candidate is not eligible for public release")
            record["stage"] = "public"
            record["promoted_at"] = datetime.now(UTC).isoformat()
            write_json_atomic(self.path, registry)
            return cast(dict[str, Any], record)
        raise KeyError(checkpoint_sha256)
