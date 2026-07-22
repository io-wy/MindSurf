"""Generation-health instrument tests.

The instrument this replaces failed §2.1 of the completion checklist on two
counts, and both are pinned here: early stopping was a free pass, and the
statistic it used was script-relative.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import torch
from transformers import PreTrainedTokenizerBase

from python_starter.core.evaluation import evaluate_generation_health


class _StubTokenizer:
    """Character-level stand-in; token id 0 is the end-of-sequence marker."""

    eos_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [ord(value) for value in text]

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return "".join(chr(value) for value in ids if not (skip_special_tokens and value == 0))


class _ScriptedModel:
    """Emits a fixed continuation, so the instrument is tested and not a model."""

    def __init__(self, continuation: str, stop_after: int | None = None) -> None:
        self.continuation = continuation
        self.stop_after = stop_after

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        eos_token_id: int | None,
    ) -> torch.Tensor:
        produced: list[int] = []
        for index in range(max_new_tokens):
            if self.stop_after is not None and index == self.stop_after:
                produced.append(0)
                # Honouring eos here is exactly the behaviour under test: the
                # instrument must keep going so short and long models are
                # measured on the same amount of text.
                if eos_token_id is not None:
                    break
                continue
            produced.append(ord(self.continuation[index % len(self.continuation)]))
        return torch.tensor([input_ids[0].tolist() + produced], dtype=torch.long)


def _probes(path: Path, count: int = 4) -> Path:
    path.write_text(
        "\n".join(
            json.dumps({"id": f"p{index}", "category": "zh", "prompt": "开头"})
            for index in range(count)
        ),
        encoding="utf-8",
    )
    return path


def _run(model: Any, path: Path, new_tokens: int = 40) -> dict[str, Any]:
    # The stub implements only the encode/decode surface the instrument uses;
    # requiring a real PreTrainedTokenizerBase here would put a model's
    # vocabulary between the test and the behaviour under test.
    tokenizer = cast(PreTrainedTokenizerBase, _StubTokenizer())
    return evaluate_generation_health(
        model, tokenizer, path, torch.device("cpu"), new_tokens=new_tokens
    )


def test_an_early_stopping_model_is_still_measured_on_full_length(tmp_path: Path) -> None:
    """The defect that made the old count incomparable across models.

    A model emitting end-of-sequence after two tokens used to produce a
    completion too short to contain any repetition and scored perfectly. Here
    it must be judged on the same token budget as everyone else.
    """
    probes = _probes(tmp_path / "probes.jsonl")

    result = _run(_ScriptedModel("重复重复", stop_after=2), probes)

    assert result["items"][0]["generated_tokens"] == 40
    assert result["degenerate_count"] == result["probes"]


def test_natural_stopping_is_reported_separately_not_folded_in(tmp_path: Path) -> None:
    probes = _probes(tmp_path / "probes.jsonl")

    stops = _run(_ScriptedModel("这是一段没有重复的自然文本内容各不相同", stop_after=5), probes)
    never = _run(_ScriptedModel("这是一段没有重复的自然文本内容各不相同"), probes)

    assert stops["natural_stop_rate"] == 1.0
    assert never["natural_stop_rate"] == 0.0
    # Whether a model stops is a real property, but it must not change the
    # degeneracy verdict for otherwise identical text.
    assert stops["degenerate_count"] == never["degenerate_count"] == 0


def test_degenerate_output_is_caught(tmp_path: Path) -> None:
    probes = _probes(tmp_path / "probes.jsonl")

    result = _run(_ScriptedModel("啊"), probes)

    assert result["degenerate_count"] == result["probes"]
    assert result["mean_top_bigram_mass"] > 0.9


def test_varied_output_passes(tmp_path: Path) -> None:
    probes = _probes(tmp_path / "probes.jsonl")

    varied = "深度学习模型训练需要算力与语料数据管线决定有效样本量而评测决定结论"
    result = _run(_ScriptedModel(varied), probes)

    assert result["degenerate_count"] == 0


def test_statistic_is_script_neutral(tmp_path: Path) -> None:
    """Chinese and English prose must land on the same side of the line.

    The replaced statistic counted how much of a script's bigram inventory a
    text used, which scores an alphabet of 26 letters far higher than Chinese
    for reasons unrelated to degeneration.
    """
    probes = _probes(tmp_path / "probes.jsonl", count=1)

    chinese = _run(_ScriptedModel("深度学习模型训练需要算力与高质量语料支撑"), probes)
    english = _run(_ScriptedModel("training a language model needs compute and text"), probes)

    assert chinese["degenerate_count"] == english["degenerate_count"] == 0
    assert abs(chinese["mean_top_bigram_mass"] - english["mean_top_bigram_mass"]) < 0.1


def test_per_category_rates_are_reported(tmp_path: Path) -> None:
    path = tmp_path / "probes.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"id": "a", "category": "zh", "prompt": "开头"}),
                json.dumps({"id": "b", "category": "code", "prompt": "开头"}),
            ]
        ),
        encoding="utf-8",
    )

    result = _run(_ScriptedModel("啊"), path)

    # A bias confined to one genre must be visible in the artifact rather than
    # averaged away into the overall rate.
    assert set(result["by_category"]) == {"zh", "code"}
    assert result["by_category"]["zh"]["probes"] == 1
