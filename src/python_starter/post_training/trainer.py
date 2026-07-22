"""TRL trainer selection and PEFT configuration helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from python_starter.post_training.data import PostTrainingDatasets


class PostTrainingConfigError(ValueError):
    """Raised when post-training configuration is invalid."""


@dataclass(frozen=True)
class TrainerSpec:
    """Names of the TRL classes used by a post-training stage."""

    trainer_name: str
    config_name: str


@dataclass(frozen=True)
class TrainingLibraries:
    """Runtime classes used to assemble a TRL trainer."""

    trl_module: Any
    auto_tokenizer_cls: type[Any]
    auto_model_cls: type[Any]
    lora_config_cls: type[Any]


@dataclass(frozen=True)
class PostTrainingRuntime:
    """Objects required to train and persist a post-trained model."""

    trainer: Any
    tokenizer: Any
    output_dir: Path
    resume_from_checkpoint: str | None
    save_merged_model: bool = False
    merged_output_dir: Path | None = None


_TRAINER_SPECS = {
    "sft": TrainerSpec(trainer_name="SFTTrainer", config_name="SFTConfig"),
    "dpo": TrainerSpec(trainer_name="DPOTrainer", config_name="DPOConfig"),
}


def get_trainer_spec(stage: str) -> TrainerSpec:
    """Return the TRL trainer/config pair for ``stage``."""
    normalized = stage.strip().lower()
    try:
        return _TRAINER_SPECS[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(_TRAINER_SPECS))
        raise PostTrainingConfigError(
            f"Unsupported stage {stage!r}; expected one of: {supported}"
        ) from exc


def normalize_peft_options(config: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return kwargs for ``LoraConfig`` or ``None`` for full fine-tuning."""
    if not bool(config.get("enabled", False)):
        return None
    return {key: value for key, value in config.items() if key != "enabled"}


def prepare_tokenizer(tokenizer: Any, stage: str) -> Any:
    """Set the padding contract required by the selected TRL trainer."""
    get_trainer_spec(stage)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise PostTrainingConfigError("Tokenizer must define either pad_token or eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" if stage.strip().lower() == "dpo" else "right"
    return tokenizer


def create_training_config(
    stage: str,
    config: Mapping[str, Any],
    trl_module: Any = None,
) -> Any:
    """Create the stage-specific TRL training arguments."""
    spec = get_trainer_spec(stage)
    if trl_module is None:
        import trl as trl_module

    config_class = getattr(trl_module, spec.config_name)
    options = {key: value for key, value in config.items() if key != "stage"}
    return config_class(**options)


def create_peft_config(
    config: Mapping[str, Any],
    lora_config_cls: type[Any] | None = None,
) -> Any:
    """Create a PEFT ``LoraConfig`` when adapter training is enabled."""
    options = normalize_peft_options(config)
    if options is None:
        return None
    if lora_config_cls is None:
        from peft import LoraConfig

        lora_config_cls = LoraConfig
    return lora_config_cls(**options)


def _load_training_libraries() -> TrainingLibraries:
    import trl
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer

    return TrainingLibraries(
        trl_module=trl,
        auto_tokenizer_cls=AutoTokenizer,
        auto_model_cls=AutoModelForCausalLM,
        lora_config_cls=LoraConfig,
    )


def _required_mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise PostTrainingConfigError(f"Configuration section {key!r} is required")
    return value


def create_post_training_runtime(
    config: Mapping[str, Any],
    datasets: PostTrainingDatasets,
    libraries: TrainingLibraries | None = None,
) -> PostTrainingRuntime:
    """Assemble a configured SFT or DPO trainer without starting training."""
    stage = str(config.get("stage", "")).strip().lower()
    spec = get_trainer_spec(stage)
    model_config = _required_mapping(config, "model")
    training_options = dict(_required_mapping(config, "training"))
    peft_options = _required_mapping(config, "peft")
    artifacts_config = config.get("artifacts", {})
    if not isinstance(artifacts_config, Mapping):
        raise PostTrainingConfigError("Configuration section 'artifacts' must be a mapping")
    runtime_libraries = libraries or _load_training_libraries()

    model_name_or_path = str(model_config.get("name_or_path", "")).strip()
    if not model_name_or_path:
        raise PostTrainingConfigError("model.name_or_path is required")
    tokenizer_name_or_path = str(model_config.get("tokenizer_name_or_path") or model_name_or_path)
    trust_remote_code = bool(model_config.get("trust_remote_code", False))

    tokenizer = runtime_libraries.auto_tokenizer_cls.from_pretrained(
        tokenizer_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    prepare_tokenizer(tokenizer, stage)

    training_options.setdefault("trust_remote_code", trust_remote_code)
    training_config = create_training_config(
        stage,
        training_options,
        trl_module=runtime_libraries.trl_module,
    )
    peft_config = create_peft_config(
        peft_options,
        lora_config_cls=runtime_libraries.lora_config_cls,
    )
    trainer_class = getattr(runtime_libraries.trl_module, spec.trainer_name)

    trainer_kwargs: dict[str, Any] = {
        "model": model_name_or_path,
        "args": training_config,
        "train_dataset": datasets.train,
        "eval_dataset": datasets.eval,
        "processing_class": tokenizer,
        "peft_config": peft_config,
    }
    if stage == "dpo":
        ref_model = None
        ref_model_name_or_path = model_config.get("ref_model_name_or_path")
        if ref_model_name_or_path:
            ref_model = runtime_libraries.auto_model_cls.from_pretrained(
                str(ref_model_name_or_path),
                trust_remote_code=trust_remote_code,
            )
        trainer_kwargs["ref_model"] = ref_model

    trainer = trainer_class(**trainer_kwargs)
    output_dir = Path(str(training_config.output_dir))
    resume_from_checkpoint = getattr(training_config, "resume_from_checkpoint", None)
    save_merged_model = bool(artifacts_config.get("save_merged_model", False))
    merged_subdir = str(artifacts_config.get("merged_subdir", "merged"))
    return PostTrainingRuntime(
        trainer=trainer,
        tokenizer=tokenizer,
        output_dir=output_dir,
        resume_from_checkpoint=resume_from_checkpoint,
        save_merged_model=save_merged_model,
        merged_output_dir=output_dir / merged_subdir if save_merged_model else None,
    )


def run_post_training(runtime: PostTrainingRuntime) -> None:
    """Train and persist a reloadable model/tokenizer directory."""
    runtime.output_dir.mkdir(parents=True, exist_ok=True)
    runtime.trainer.train(resume_from_checkpoint=runtime.resume_from_checkpoint)
    output_dir = str(runtime.output_dir)
    runtime.trainer.save_model(output_dir)
    runtime.tokenizer.save_pretrained(output_dir)

    if getattr(runtime, "save_merged_model", False):
        merge_and_unload = getattr(runtime.trainer.model, "merge_and_unload", None)
        merged_output_dir = getattr(runtime, "merged_output_dir", None)
        if callable(merge_and_unload) and merged_output_dir is not None:
            merged_output_dir.mkdir(parents=True, exist_ok=True)
            merged_model = merge_and_unload()
            merged_model.save_pretrained(
                str(merged_output_dir),
                safe_serialization=True,
            )
            runtime.tokenizer.save_pretrained(str(merged_output_dir))
