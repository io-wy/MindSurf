"""Model inference endpoints.

Provides REST API for running inference on trained models.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, status

from python_starter.api.dependencies import SettingsDep
from python_starter.api.schemas.models import InferenceRequest, InferenceResponse
from python_starter.infrastructure.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("", response_model=InferenceResponse)
async def run_inference(
    request: InferenceRequest,
    settings: SettingsDep,
) -> InferenceResponse:
    """Run model inference on input text.

    This is a placeholder implementation. In a real setup:
    1. Load the model from app.state or MLflow registry
    2. Tokenize input
    3. Run forward pass
    4. Decode output

    For now, returns a mock response to demonstrate the API contract.
    """
    logger.info(
        "inference_request",
        input_length=len(request.text),
        max_length=request.max_length,
        temperature=request.temperature,
    )

    # TODO: Replace with actual model loading and inference
    # model = request.app.state.model  # or load from MLflow
    start = time.perf_counter()

    # Placeholder: echo the input with a mock response
    mock_output = f"[Model output for: {request.text[:50]}...]"
    elapsed_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "inference_complete",
        output_length=len(mock_output),
        elapsed_ms=elapsed_ms,
    )

    return InferenceResponse(
        text=mock_output,
        input_tokens=len(request.text.split()),
        output_tokens=len(mock_output.split()),
        generation_time_ms=elapsed_ms,
    )


@router.get("/models")
async def list_available_models(settings: SettingsDep) -> dict:
    """List models available for inference."""
    # TODO: Query MLflow model registry or local model directory
    return {
        "models": [
            {
                "name": "minimind",
                "version": "v1",
                "path": "models/minimind/latest",
                "description": "Default minimind model",
            }
        ]
    }
