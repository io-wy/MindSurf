"""Celery application configuration.

Sets up the Celery app with Redis broker and result backend.
"""

from __future__ import annotations

import os

from celery import Celery

from python_starter.infrastructure.config import get_settings

settings = get_settings()

celery_app = Celery(
    "python_starter",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["python_starter.tasks.training"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600 * 24,  # 24 hours max per task
    worker_prefetch_multiplier=1,
)
