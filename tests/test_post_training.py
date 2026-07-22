"""Tests for the isolated post-training module."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from python_starter.post_training.data import (
    PostTrainingDataError,
    PostTrainingDatasets,
    load_post_training_datasets,
    load_preference_records,
    load_sft_records,
)
from python_starter.post_training.inference import (
    InferenceLibraries,
    PostTrainingInferenceRuntime,
    generate_response,
    load_post_training_model,
)
from python_starter.post_training.trainer import (
    PostTrainingConfigError,
    PostTrainingRuntime,
    TrainingLibraries,
    create_peft_config,
    create_post_training_runtime,
    create_training_config,
    get_trainer_spec,
    normalize_peft_options,
    prepare_tokenizer,
    run_post_training,
)


class PostTrainingDataTests(unittest.TestCase):
    """Exercise the local JSONL contract before TRL sees the data."""

    def test_sft_prompt_response_is_normalized_to_prompt_completion(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "sft.jsonl"
            path.write_text(
                json.dumps({"prompt": "Question", "response": "Answer"}) + "\n",
                encoding="utf-8",
            )

            records = load_sft_records(path)

        self.assertEqual(records, [{"prompt": "Question", "completion": "Answer"}])

    def test_dpo_requires_prompt_chosen_and_rejected(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "dpo.jsonl"
            path.write_text(
                json.dumps({"prompt": "Question", "chosen": "Good"}) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(PostTrainingDataError, "rejected"):
                load_preference_records(path)

    def test_invalid_json_reports_the_source_line(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "broken.jsonl"
            path.write_text('{"prompt": "Question"\n', encoding="utf-8")

            with self.assertRaisesRegex(PostTrainingDataError, "broken.jsonl:1"):
                load_sft_records(path)

    def test_dataset_bundle_keeps_evaluation_optional(self) -> None:
        class Dataset:
            @classmethod
            def from_list(cls, records: list[dict[str, object]]) -> list[dict[str, object]]:
                return records

        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "sft.jsonl"
            path.write_text(
                json.dumps({"prompt": "Question", "response": "Answer"}) + "\n",
                encoding="utf-8",
            )

            datasets = load_post_training_datasets(
                stage="sft",
                train_path=path,
                eval_path=None,
                dataset_cls=Dataset,
            )

        self.assertEqual(
            datasets.train,
            [{"prompt": "Question", "completion": "Answer"}],
        )
        self.assertIsNone(datasets.eval)


class PostTrainingConfigTests(unittest.TestCase):
    """Keep TRL and PEFT selection explicit and deterministic."""

    def test_sft_selects_sft_trainer_and_config(self) -> None:
        spec = get_trainer_spec("sft")

        self.assertEqual(spec.trainer_name, "SFTTrainer")
        self.assertEqual(spec.config_name, "SFTConfig")

    def test_dpo_selects_dpo_trainer_and_config(self) -> None:
        spec = get_trainer_spec("dpo")

        self.assertEqual(spec.trainer_name, "DPOTrainer")
        self.assertEqual(spec.config_name, "DPOConfig")

    def test_unknown_stage_is_rejected(self) -> None:
        with self.assertRaisesRegex(PostTrainingConfigError, "Unsupported stage"):
            get_trainer_spec("ppo")

    def test_disabled_peft_returns_no_options(self) -> None:
        self.assertIsNone(normalize_peft_options({"enabled": False, "r": 16}))

    def test_enabled_peft_removes_control_flag(self) -> None:
        options = normalize_peft_options(
            {
                "enabled": True,
                "r": 16,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "v_proj"],
            }
        )

        self.assertEqual(
            options,
            {
                "r": 16,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "v_proj"],
            },
        )

    def test_dpo_tokenizer_uses_left_padding_and_eos_as_pad(self) -> None:
        class Tokenizer:
            pad_token = None
            eos_token = "<eos>"
            padding_side = "right"

        tokenizer = Tokenizer()

        prepare_tokenizer(tokenizer, "dpo")

        self.assertEqual(tokenizer.pad_token, "<eos>")
        self.assertEqual(tokenizer.padding_side, "left")

    def test_sft_tokenizer_keeps_right_padding(self) -> None:
        class Tokenizer:
            pad_token = "<pad>"
            eos_token = "<eos>"
            padding_side = "left"

        tokenizer = Tokenizer()

        prepare_tokenizer(tokenizer, "sft")

        self.assertEqual(tokenizer.pad_token, "<pad>")
        self.assertEqual(tokenizer.padding_side, "right")

    def test_training_config_excludes_stage_selector(self) -> None:
        class Config:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs

        class TrlModule:
            SFTConfig = Config
            DPOConfig = Config

        config = create_training_config(
            "dpo",
            {"stage": "dpo", "output_dir": "models/dpo", "beta": 0.1},
            trl_module=TrlModule,
        )

        self.assertEqual(
            config.kwargs,
            {"output_dir": "models/dpo", "beta": 0.1},
        )

    def test_peft_config_uses_normalized_lora_options(self) -> None:
        class LoraConfig:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs

        config = create_peft_config(
            {"enabled": True, "r": 8, "target_modules": "all-linear"},
            lora_config_cls=LoraConfig,
        )

        self.assertIsNotNone(config)
        self.assertEqual(
            config.kwargs,
            {"r": 8, "target_modules": "all-linear"},
        )

    def test_sft_runtime_wires_model_tokenizer_data_and_lora(self) -> None:
        class Tokenizer:
            pad_token = None
            eos_token = "<eos>"
            padding_side = "left"

            def __init__(self) -> None:
                self.path = ""
                self.load_kwargs: dict[str, object] = {}

            @classmethod
            def from_pretrained(cls, path: str, **kwargs: object) -> Tokenizer:
                tokenizer = cls()
                tokenizer.path = path
                tokenizer.load_kwargs = kwargs
                return tokenizer

        class Config:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs
                self.output_dir = kwargs["output_dir"]
                self.resume_from_checkpoint = kwargs.get("resume_from_checkpoint")

        class Trainer:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs

        class LoraConfig:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs

        libraries = TrainingLibraries(
            trl_module=SimpleNamespace(
                SFTConfig=Config,
                DPOConfig=Config,
                SFTTrainer=Trainer,
                DPOTrainer=Trainer,
            ),
            auto_tokenizer_cls=Tokenizer,
            auto_model_cls=object,
            lora_config_cls=LoraConfig,
        )
        datasets = PostTrainingDatasets(train=[{"text": "hello"}])

        runtime = create_post_training_runtime(
            {
                "stage": "sft",
                "model": {
                    "name_or_path": "tiny-model",
                    "tokenizer_name_or_path": "tiny-tokenizer",
                    "trust_remote_code": False,
                },
                "training": {"output_dir": "models/sft"},
                "peft": {"enabled": True, "r": 8},
            },
            datasets,
            libraries=libraries,
        )

        self.assertEqual(runtime.tokenizer.path, "tiny-tokenizer")
        self.assertEqual(runtime.tokenizer.padding_side, "right")
        self.assertEqual(runtime.trainer.kwargs["model"], "tiny-model")
        self.assertEqual(runtime.trainer.kwargs["train_dataset"], datasets.train)
        self.assertEqual(runtime.trainer.kwargs["peft_config"].kwargs, {"r": 8})

    def test_dpo_runtime_loads_explicit_reference_model(self) -> None:
        class Tokenizer:
            pad_token = "<pad>"
            eos_token = "<eos>"
            padding_side = "right"

            @classmethod
            def from_pretrained(cls, path: str, **kwargs: object) -> Tokenizer:
                return cls()

        class Model:
            @classmethod
            def from_pretrained(cls, path: str, **kwargs: object) -> tuple[str, dict[str, object]]:
                return path, kwargs

        class Config:
            def __init__(self, **kwargs: object) -> None:
                self.output_dir = kwargs["output_dir"]
                self.resume_from_checkpoint = None

        class Trainer:
            def __init__(self, **kwargs: object) -> None:
                self.kwargs = kwargs

        libraries = TrainingLibraries(
            trl_module=SimpleNamespace(
                SFTConfig=Config,
                DPOConfig=Config,
                SFTTrainer=Trainer,
                DPOTrainer=Trainer,
            ),
            auto_tokenizer_cls=Tokenizer,
            auto_model_cls=Model,
            lora_config_cls=object,
        )

        runtime = create_post_training_runtime(
            {
                "stage": "dpo",
                "model": {
                    "name_or_path": "sft-model",
                    "tokenizer_name_or_path": "sft-model",
                    "ref_model_name_or_path": "reference-model",
                    "trust_remote_code": False,
                },
                "training": {"output_dir": "models/dpo"},
                "peft": {"enabled": False},
            },
            PostTrainingDatasets(train=[{"prompt": "Q", "chosen": "A", "rejected": "B"}]),
            libraries=libraries,
        )

        self.assertEqual(runtime.tokenizer.padding_side, "left")
        self.assertEqual(
            runtime.trainer.kwargs["ref_model"],
            ("reference-model", {"trust_remote_code": False}),
        )

    def test_run_trains_and_saves_model_with_tokenizer(self) -> None:
        events: list[tuple[str, object]] = []

        class Trainer:
            def train(self, resume_from_checkpoint: object = None) -> None:
                events.append(("train", resume_from_checkpoint))

            def save_model(self, output_dir: str) -> None:
                events.append(("model", output_dir))

        class Tokenizer:
            def save_pretrained(self, output_dir: str) -> None:
                events.append(("tokenizer", output_dir))

        with TemporaryDirectory() as tmp_dir:
            runtime = PostTrainingRuntime(
                trainer=Trainer(),
                tokenizer=Tokenizer(),
                output_dir=Path(tmp_dir) / "final",
                resume_from_checkpoint="models/checkpoint-10",
            )

            run_post_training(runtime)

            expected_output = str(Path(tmp_dir) / "final")
            self.assertEqual(
                events,
                [
                    ("train", "models/checkpoint-10"),
                    ("model", expected_output),
                    ("tokenizer", expected_output),
                ],
            )

    def test_run_optionally_saves_merged_adapter_checkpoint(self) -> None:
        events: list[tuple[str, object]] = []

        class MergedModel:
            def save_pretrained(self, output_dir: str, safe_serialization: bool) -> None:
                events.append(("merged", (output_dir, safe_serialization)))

        class AdapterModel:
            def merge_and_unload(self) -> MergedModel:
                events.append(("merge", None))
                return MergedModel()

        class Trainer:
            model = AdapterModel()

            def train(self, resume_from_checkpoint: object = None) -> None:
                pass

            def save_model(self, output_dir: str) -> None:
                pass

        class Tokenizer:
            def save_pretrained(self, output_dir: str) -> None:
                events.append(("tokenizer", output_dir))

        with TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / "adapter"
            merged_output_dir = output_dir / "merged"
            runtime = PostTrainingRuntime(
                trainer=Trainer(),
                tokenizer=Tokenizer(),
                output_dir=output_dir,
                resume_from_checkpoint=None,
                save_merged_model=True,
                merged_output_dir=merged_output_dir,
            )

            run_post_training(runtime)

            self.assertIn(("merge", None), events)
            self.assertIn(
                ("merged", (str(merged_output_dir), True)),
                events,
            )
            self.assertIn(("tokenizer", str(merged_output_dir)), events)


class PostTrainingInferenceTests(unittest.TestCase):
    """Verify merged and adapter checkpoints share one loading API."""

    def test_adapter_loading_uses_base_model_and_adapter_tokenizer(self) -> None:
        events: list[tuple[str, object]] = []

        class Model:
            def to(self, device: str) -> Model:
                events.append(("device", device))
                return self

            def eval(self) -> None:
                events.append(("eval", None))

        class AutoModel:
            @classmethod
            def from_pretrained(cls, path: str, **kwargs: object) -> Model:
                events.append(("base", (path, kwargs)))
                return Model()

        class PeftModel:
            @classmethod
            def from_pretrained(cls, model: Model, path: str) -> Model:
                events.append(("adapter", path))
                return model

        class Tokenizer:
            pad_token = None
            eos_token = "<eos>"
            padding_side = "right"

            @classmethod
            def from_pretrained(cls, path: str, **kwargs: object) -> Tokenizer:
                events.append(("tokenizer", (path, kwargs)))
                return cls()

        runtime = load_post_training_model(
            model_name_or_path="base-model",
            adapter_name_or_path="adapter-model",
            device="cpu",
            libraries=InferenceLibraries(
                auto_model_cls=AutoModel,
                auto_tokenizer_cls=Tokenizer,
                peft_model_cls=PeftModel,
            ),
        )

        self.assertIn(("adapter", "adapter-model"), events)
        self.assertIn(("tokenizer", ("adapter-model", {"trust_remote_code": False})), events)
        self.assertIn(("device", "cpu"), events)
        self.assertIn(("eval", None), events)
        self.assertEqual(runtime.tokenizer.pad_token, "<eos>")
        self.assertEqual(runtime.tokenizer.padding_side, "left")

    def test_generation_decodes_only_new_tokens(self) -> None:
        import torch

        class Inputs(dict[str, torch.Tensor]):
            def to(self, device: str) -> Inputs:
                return self

        class Tokenizer:
            eos_token_id = 1
            pad_token_id = 0

            def __call__(self, prompt: str, return_tensors: str) -> Inputs:
                return Inputs(input_ids=torch.tensor([[2, 3]]))

            def decode(self, token_ids: torch.Tensor, skip_special_tokens: bool) -> str:
                self.decoded_ids = token_ids.tolist()
                return "Answer"

        class Model:
            device = "cpu"

            def generate(self, **kwargs: object) -> torch.Tensor:
                return torch.tensor([[2, 3, 4, 5]])

        tokenizer = Tokenizer()
        response = generate_response(
            PostTrainingInferenceRuntime(model=Model(), tokenizer=tokenizer),
            prompt="Question",
            max_new_tokens=2,
            temperature=0.7,
            top_p=0.9,
        )

        self.assertEqual(response, "Answer")
        self.assertEqual(tokenizer.decoded_ids, [4, 5])


if __name__ == "__main__":
    unittest.main()
