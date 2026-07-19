"""REST inference backed by the checkpoint loaded during application startup."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool

from python_starter.api.schemas.models import InferenceRequest, InferenceResponse
from python_starter.core.inference import InferenceEngine
from python_starter.infrastructure.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.post("", response_model=InferenceResponse)
async def run_inference(
    request: InferenceRequest,
    http_request: Request,
) -> InferenceResponse:
    """Generate text with the configured checkpoint without blocking the event loop."""
    engine: InferenceEngine | None = getattr(
        http_request.app.state,
        "inference_engine",
        None,
    )
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Inference model is not loaded. Configure INFERENCE_CHECKPOINT "
                "and INFERENCE_TOKENIZER."
            ),
        )

    result = await run_in_threadpool(
        engine.generate,
        request.text,
        max_new_tokens=request.max_length,
        temperature=request.temperature,
        top_p=request.top_p,
    )
    logger.info(
        "inference_complete",
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        elapsed_ms=result.generation_time_ms,
    )
    return InferenceResponse(
        text=result.text,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        generation_time_ms=result.generation_time_ms,
    )


@router.get("/models")
async def list_available_models(request: Request) -> dict[str, list[dict[str, Any]]]:
    """List the model actually resident in this API process."""
    engine: InferenceEngine | None = getattr(request.app.state, "inference_engine", None)
    if engine is None:
        return {"models": []}
    return {
        "models": [
            {
                "checkpoint": str(engine.checkpoint_path),
                "checkpoint_sha256": engine.checkpoint_sha256,
                "tokenizer": str(engine.tokenizer_path),
                "model_config": engine.model.config.to_dict(),
            }
        ]
    }
