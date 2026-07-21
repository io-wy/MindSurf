"""Unified experiment tracking for W&B and MLflow.

Provides a single interface that delegates to both backends simultaneously.
Either can be disabled by not configuring its API key / URI.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import wandb

from python_starter.infrastructure.config import Settings
from python_starter.infrastructure.logging import get_logger

logger = get_logger(__name__)


def _config_fingerprint(config: dict[str, Any]) -> str:
    """Stable digest of a run's resolved config, for spotting silent drift."""
    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_head() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _environment_snapshot() -> dict[str, Any]:
    """Record what a result cannot be reproduced without.

    The standard asks for code version, dependency state and platform spec.
    Each is cheap here and impossible to recover later: a result whose torch
    build, GPU model or determinism setting is unknown cannot be re-derived,
    only re-run and hoped over.
    """
    snapshot: dict[str, Any] = {
        "git_head": _git_head(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor_count": os.cpu_count(),
    }
    try:
        import torch

        snapshot["torch"] = torch.__version__
        snapshot["cuda"] = torch.version.cuda
        snapshot["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic)
        snapshot["cublas_workspace_config"] = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if torch.cuda.is_available():
            snapshot["gpu_name"] = torch.cuda.get_device_name(0)
            snapshot["gpu_total_bytes"] = torch.cuda.get_device_properties(0).total_memory
            snapshot["driver_cuda"] = torch.version.cuda
    except ImportError:
        snapshot["torch"] = None
    try:
        frozen = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True
        ).stdout
        snapshot["dependencies_sha256"] = hashlib.sha256(frozen.encode("utf-8")).hexdigest()
        snapshot["dependency_count"] = len([line for line in frozen.splitlines() if line.strip()])
    except (OSError, subprocess.CalledProcessError):
        snapshot["dependencies_sha256"] = None
    return snapshot


class ExperimentTracker:
    """Always log locally, with optional W&B and MLflow mirrors."""

    def __init__(
        self,
        settings: Settings,
        experiment_name: str | None = None,
        local_dir: str | Path = "artifacts/runs",
    ) -> None:
        self.settings = settings
        self.experiment_name = experiment_name or settings.mlflow_experiment_name
        self.local_dir = Path(local_dir)
        self.run_dir: Path | None = None
        self.metrics_path: Path | None = None
        self._wandb_run: wandb.sdk.wandb_run.Run | None = None
        self._mlflow_active = False

    def start(self, run_name: str | None = None, config: dict[str, Any] | None = None) -> None:
        """Initialize tracking backends."""
        run_id = run_name or (
            f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
        )
        self.run_dir = self.local_dir / run_id
        resumed = self.run_dir.exists()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        if resumed:
            # An existing directory means either a genuine resume or a second
            # experiment reusing the run name. The two are indistinguishable
            # from here, and guessing "resume" silently interleaves two runs'
            # metrics in one file and keeps the first run's config as the
            # record of the second. Record the ambiguity instead of hiding it.
            self._append_local(
                {
                    "type": "resume",
                    "config_fingerprint": _config_fingerprint(config or {}),
                    "started_at": datetime.now(UTC).isoformat(),
                }
            )
        else:
            self._write_json_atomic(
                self.run_dir / "run.json",
                {
                    "run_id": run_id,
                    "experiment_name": self.experiment_name,
                    "started_at": datetime.now(UTC).isoformat(),
                    "config": config or {},
                    "config_fingerprint": _config_fingerprint(config or {}),
                    "environment": _environment_snapshot(),
                },
            )

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
        self._append_local({"type": "params", "params": params})
        if self._wandb_run:
            wandb.config.update(params)  # type: ignore[no-untyped-call]
        if self._mlflow_active:
            for k, v in params.items():
                mlflow.log_param(k, v)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log scalar metrics."""
        self._append_local({"type": "metrics", "step": step, "metrics": metrics})
        if self._wandb_run:
            wandb.log(metrics, step=step)
        if self._mlflow_active:
            for k, v in metrics.items():
                mlflow.log_metric(k, v, step=step)

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Log a file artifact."""
        self._append_local(
            {
                "type": "artifact",
                "local_path": local_path,
                "artifact_path": artifact_path,
            }
        )
        if self._wandb_run:
            artifact = wandb.Artifact(name="model-artifacts", type="model")
            artifact.add_file(local_path)
            wandb.log_artifact(artifact)
        if self._mlflow_active:
            mlflow.log_artifact(local_path, artifact_path=artifact_path)

    def finish(self) -> None:
        """Close tracking sessions."""
        # An explicit end time, so reading it does not mean scanning the metrics
        # log for its last row and hoping the run did not die mid-write.
        self._append_local({"type": "finish", "ended_at": datetime.now(UTC).isoformat()})
        if self._wandb_run:
            wandb.finish()
            self._wandb_run = None
        if self._mlflow_active:
            mlflow.end_run()
            self._mlflow_active = False
        logger.info("tracking_finished")

    def _append_local(self, payload: dict[str, Any]) -> None:
        if self.metrics_path is None:
            raise RuntimeError("tracker.start() must be called before logging")
        event = {
            "timestamp": datetime.now(UTC).isoformat(),
            **payload,
        }
        with self.metrics_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
        temporary = path.with_suffix(f"{path.suffix}.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
