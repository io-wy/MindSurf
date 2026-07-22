"""Load merged or PEFT post-training checkpoints for inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class InferenceLibraries:
    """Runtime model/tokenizer classes used by the inference loader."""

    auto_model_cls: type[Any]
    auto_tokenizer_cls: type[Any]
    peft_model_cls: type[Any]


@dataclass(frozen=True)
class PostTrainingInferenceRuntime:
    """A model and tokenizer ready for generation."""

    model: Any
    tokenizer: Any


def _load_inference_libraries() -> InferenceLibraries:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return InferenceLibraries(
        auto_model_cls=AutoModelForCausalLM,
        auto_tokenizer_cls=AutoTokenizer,
        peft_model_cls=PeftModel,
    )


def load_post_training_model(
    model_name_or_path: str,
    adapter_name_or_path: str | None = None,
    device: Any = "cpu",
    trust_remote_code: bool = False,
    libraries: InferenceLibraries | None = None,
) -> PostTrainingInferenceRuntime:
    """Load a merged model or attach a PEFT adapter to its base model."""
    runtime_libraries = libraries or _load_inference_libraries()
    tokenizer_path = adapter_name_or_path or model_name_or_path
    tokenizer = runtime_libraries.auto_tokenizer_cls.from_pretrained(
        tokenizer_path,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = runtime_libraries.auto_model_cls.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    if adapter_name_or_path is not None:
        model = runtime_libraries.peft_model_cls.from_pretrained(
            model,
            adapter_name_or_path,
        )
    model.to(device)
    model.eval()
    return PostTrainingInferenceRuntime(model=model, tokenizer=tokenizer)


def generate_response(
    runtime: PostTrainingInferenceRuntime,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Generate and decode only the continuation after ``prompt``."""
    import torch

    inputs = runtime.tokenizer(prompt, return_tensors="pt").to(runtime.model.device)
    prompt_length = inputs["input_ids"].shape[-1]
    generation_options: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": runtime.tokenizer.pad_token_id,
        "eos_token_id": runtime.tokenizer.eos_token_id,
    }
    if temperature > 0:
        generation_options["temperature"] = temperature
        generation_options["top_p"] = top_p

    with torch.no_grad():
        output = runtime.model.generate(**inputs, **generation_options)
    continuation = output[0, prompt_length:]
    decoded = runtime.tokenizer.decode(
        continuation,
        skip_special_tokens=True,
    )
    return str(decoded).strip()
