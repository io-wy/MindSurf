"""API endpoint tests."""

from __future__ import annotations

import pytest
from httpx import AsyncClient

from python_starter.api.schemas.models import ExperimentStatus


@pytest.mark.asyncio
async def test_health_endpoint(api_client: AsyncClient) -> None:
    response = await api_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


@pytest.mark.asyncio
async def test_readiness_probe(api_client: AsyncClient) -> None:
    response = await api_client.get("/health/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("ok", "degraded")
    assert "services" in data


@pytest.mark.asyncio
async def test_create_experiment(api_client: AsyncClient) -> None:
    payload = {"name": "test-exp", "description": "A test experiment"}
    response = await api_client.post("/experiments", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "test-exp"
    assert data["status"] == ExperimentStatus.CREATED.value


@pytest.mark.asyncio
async def test_list_experiments(api_client: AsyncClient) -> None:
    # Create an experiment first
    await api_client.post("/experiments", json={"name": "list-test"})

    response = await api_client.get("/experiments")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    assert "total" in data
    assert data["page"] == 1


@pytest.mark.asyncio
async def test_experiment_crud_metrics_and_model_registry(api_client: AsyncClient) -> None:
    created = await api_client.post(
        "/experiments",
        json={"name": "crud", "model_config_snapshot": {"layers": 1}},
    )
    experiment_id = created.json()["id"]

    fetched = await api_client.get(f"/experiments/{experiment_id}")
    assert fetched.status_code == 200
    assert fetched.json()["model_config_snapshot"] == {"layers": 1}

    updated = await api_client.patch(
        f"/experiments/{experiment_id}/status",
        params={"experiment_status": "running"},
    )
    assert updated.status_code == 200
    assert updated.json()["status"] == "running"

    metrics = await api_client.post(
        f"/experiments/{experiment_id}/metrics",
        json={"loss": 2.0},
    )
    assert metrics.status_code == 200
    assert metrics.json()["metrics"]["loss"] == 2.0

    registered = await api_client.post(
        f"/experiments/{experiment_id}/models",
        json={
            "experiment_id": experiment_id,
            "name": "candidate",
            "artifact_path": "models/candidate.pt",
            "metrics": {"loss": 2.0},
        },
    )
    assert registered.status_code == 201
    models = await api_client.get(f"/experiments/{experiment_id}/models")
    assert models.status_code == 200
    assert models.json()[0]["name"] == "candidate"

    filtered = await api_client.get("/experiments", params={"status": "running"})
    assert filtered.status_code == 200
    assert filtered.json()["total"] == 1


@pytest.mark.asyncio
async def test_experiment_not_found_paths(api_client: AsyncClient) -> None:
    assert (await api_client.get("/experiments/999")).status_code == 404
    assert (
        await api_client.patch(
            "/experiments/999/status",
            params={"experiment_status": "failed"},
        )
    ).status_code == 404
    assert (
        await api_client.post("/experiments/999/metrics", json={"loss": 1.0})
    ).status_code == 404
    assert (
        await api_client.post(
            "/experiments/999/models",
            json={
                "experiment_id": 999,
                "name": "missing",
                "artifact_path": "missing.pt",
            },
        )
    ).status_code == 404


@pytest.mark.asyncio
async def test_inference_requires_a_configured_model(api_client: AsyncClient) -> None:
    payload = {"text": "Hello world", "max_length": 32}
    response = await api_client.post("/inference", json=payload)
    assert response.status_code == 503
    assert "INFERENCE_CHECKPOINT" in response.json()["detail"]


@pytest.mark.asyncio
async def test_inference_model_listing_without_model(api_client: AsyncClient) -> None:
    response = await api_client.get("/inference/models")
    assert response.status_code == 200
    assert response.json() == {"models": []}


@pytest.mark.asyncio
async def test_training_job_submission_and_status(
    api_client: AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from python_starter.tasks import celery_app as celery_module
    from python_starter.tasks import training

    class _Task:
        id = "job-1"

    class _Result:
        state = "SUCCESS"
        result = {"status": "completed"}
        info = None

        def ready(self) -> bool:
            return True

    monkeypatch.setattr(training.run_training_task, "delay", lambda *_: _Task())
    monkeypatch.setattr(celery_module.celery_app, "AsyncResult", lambda *_: _Result())

    submitted = await api_client.post(
        "/experiments/jobs",
        json={"experiment_name": "job", "config_overrides": {"training": {"max_steps": 1}}},
    )
    assert submitted.status_code == 202
    assert submitted.json()["job_id"] == "job-1"

    status = await api_client.get("/experiments/jobs/job-1")
    assert status.status_code == 200
    assert status.json()["status"] == "success"
