"""Runtime evaluation tests with a tiny deterministic model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

from python_starter.core.evaluation import (
    DOMAIN_NAMES,
    classify_domains,
    compare_mcq_items,
    domain_losses,
    evaluate_fixed_prompts,
    evaluate_gate,
    evaluate_mcq,
    packed_loss,
    repeated_bigram_ratio,
    run_candidate_evaluation,
    source_tree_sha256,
    wilson_interval,
)
from python_starter.core.model import ModelConfig, TransformerLM


class _TinyTokenizer:
    eos_token_id = 1
    bos_token_id = 2
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [3 + (ord(value) % 25) for value in text]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        assert skip_special_tokens
        return "南京 blue light sky data 学习 def return 香蕉 不确定。建议一。建议二。"


def _model(max_seq_len: int = 64) -> TransformerLM:
    torch.manual_seed(3)
    return TransformerLM(
        ModelConfig(
            vocab_size=32,
            n_embed=16,
            n_layer=1,
            n_head=4,
            max_seq_len=max_seq_len,
        )
    ).eval()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_domain_classification_and_repetition() -> None:
    text = "def function(): return 2 + 2 " * 100
    domains = classify_domains(text)
    assert "code_like" in domains
    assert "english_or_code_heavy" in domains
    assert "math_like" in domains
    assert "repeat_high" in domains
    assert repeated_bigram_ratio("abababababab") > 0.45
    assert repeated_bigram_ratio("short") == 0.0


def test_loss_mcq_and_fixed_prompt_evaluators(tmp_path: Path) -> None:
    tokenizer = _TinyTokenizer()
    model = _model()
    data = tmp_path / "heldout.jsonl"
    _write_jsonl(
        data,
        [
            {"text": "def function(): return 2 + 2 " * 100, "source_key": "code", "id": 1},
            {
                "text": "This is a sufficiently varied English quality document. " * 50,
                "source_key": "web",
                "id": 2,
            },
        ],
    )

    strict = packed_loss(
        model,
        tokenizer,  # type: ignore[arg-type]
        data,
        torch.device("cpu"),
        max_length=8,
        batch_size=2,
        max_batches=2,
    )
    assert strict["tokens"] == 32
    assert float(strict["loss"]) > 0

    domains, counts = domain_losses(
        model,
        tokenizer,  # type: ignore[arg-type]
        data,
        torch.device("cpu"),
        max_length=8,
        batch_size=2,
        max_blocks=1,
    )
    assert {"code_like", "english_or_code_heavy", "math_like", "repeat_high"} <= domains.keys()
    assert counts["code_like"] == 1

    mcq_path = tmp_path / "mcq.jsonl"
    _write_jsonl(
        mcq_path,
        [
            {
                "id": "one",
                "category": "fact",
                "prompt": "Answer:",
                "choices": [" A", " B"],
                "answer": 0,
            }
        ],
    )
    mcq = evaluate_mcq(
        model,
        tokenizer,  # type: ignore[arg-type]
        mcq_path,
        torch.device("cpu"),
        max_length=32,
    )
    assert mcq["total"] == 1
    assert len(mcq["items"][0]["choice_nll"]) == 2
    assert sum(mcq["items"][0]["choice_probability"]) == pytest.approx(1.0)
    assert mcq["by_category"]["fact"]["total"] == 1
    assert len(mcq["wilson_interval_95"]) == 2

    fixed_path = tmp_path / "fixed.jsonl"
    _write_jsonl(
        fixed_path,
        [
            {"id": "zh_fact_nanjing", "category": "fact", "prompt": "城市？"},
            {"id": "unknown", "category": "other", "prompt": "其他"},
        ],
    )
    fixed = evaluate_fixed_prompts(
        model,
        tokenizer,  # type: ignore[arg-type]
        fixed_path,
        torch.device("cpu"),
        max_new_tokens=2,
    )
    assert fixed["mean_score"] > 0
    assert len(fixed["items"]) == 2


def test_source_tree_identity_changes_with_content(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    file_path = source / "a.py"
    file_path.write_text("one\n", encoding="utf-8")
    first = source_tree_sha256(tmp_path, [source])
    file_path.write_text("two\n", encoding="utf-8")
    assert source_tree_sha256(tmp_path, [source]) != first


def test_mcq_statistics_are_paired_and_deterministic() -> None:
    baseline = [
        {"id": "a", "correct": True},
        {"id": "b", "correct": False},
        {"id": "c", "correct": False},
        {"id": "d", "correct": True},
    ]
    candidate = [
        {"id": "a", "correct": True},
        {"id": "b", "correct": True},
        {"id": "c", "correct": True},
        {"id": "d", "correct": False},
    ]
    comparison = compare_mcq_items(
        baseline,
        candidate,
        bootstrap_samples=1000,
        seed=7,
    )
    assert comparison["accuracy_delta"] == pytest.approx(0.25)
    assert comparison["mcnemar_exact"]["candidate_only_correct"] == 2
    assert comparison["mcnemar_exact"]["baseline_only_correct"] == 1
    lower, upper = wilson_interval(3, 4)
    assert 0 < lower < 0.75 < upper < 1


def test_candidate_bundle_orchestration_and_gate(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    import python_starter.core.evaluation as evaluation

    model = _model()
    tokenizer = _TinyTokenizer()
    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer_path.mkdir()
    (tokenizer_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    validation = tmp_path / "validation.jsonl"
    test = tmp_path / "test.jsonl"
    validation.write_text("{}\n", encoding="utf-8")
    test.write_text("{}\n", encoding="utf-8")
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "dataset_spec_sha256": "a" * 64,
                "splits": {"train": {"sha256": "b" * 64}},
                "gates": {"public_release_license_ready": False},
            }
        ),
        encoding="utf-8",
    )
    thresholds_path = tmp_path / "thresholds.json"
    thresholds = {
        "strict_val_max": 2.5,
        "strict_test_max": 2.5,
        "domain_max": dict.fromkeys(DOMAIN_NAMES, 2.5),
        "mcq_accuracy_min": 0.5,
        "fixed_score_min": 0.5,
        "repetition_count_max": 2,
    }
    thresholds_path.write_text(json.dumps(thresholds), encoding="utf-8")
    mcq_path = tmp_path / "mcq.jsonl"
    fixed_path = tmp_path / "fixed.jsonl"
    mcq_path.write_text("{}\n", encoding="utf-8")
    fixed_path.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "result.json"

    monkeypatch.setattr(
        evaluation,
        "load_checkpoint_model",
        lambda *_: (model, {"progress": {"global_step": 10}}, torch.device("cpu")),
    )
    monkeypatch.setattr(evaluation, "load_tokenizer", lambda *_: tokenizer)
    monkeypatch.setattr(
        evaluation,
        "packed_loss",
        lambda *_, **__: {"loss": 2.0, "perplexity": 7.3, "tokens": 8, "batches": 1},
    )
    monkeypatch.setattr(
        evaluation,
        "domain_losses",
        lambda *_, **__: (dict.fromkeys(DOMAIN_NAMES, 2.0), dict.fromkeys(DOMAIN_NAMES, 1)),
    )
    monkeypatch.setattr(
        evaluation,
        "evaluate_mcq",
        lambda *_, **__: {"accuracy": 1.0, "items": [], "correct": 1, "total": 1},
    )
    monkeypatch.setattr(
        evaluation,
        "evaluate_fixed_prompts",
        lambda *_, **__: {"mean_score": 1.0, "repetition_count": 0, "items": []},
    )

    result = run_candidate_evaluation(
        root=tmp_path,
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        validation_path=validation,
        test_path=test,
        audit_path=audit_path,
        thresholds_path=thresholds_path,
        mcq_path=mcq_path,
        fixed_prompts_path=fixed_path,
        output_path=output,
        device_name="cpu",
        max_length=8,
        batch_size=1,
        strict_batches=1,
        domain_blocks=1,
        fixed_new_tokens=1,
    )
    assert result["gate"]["internal_candidate_passed"] is True
    assert result["gate"]["public_release_passed"] is False
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == 2

    failed_metrics = {
        "strict_val": {"loss": float("nan")},
        "strict_test": {"loss": 3.0},
        "domain": {},
        "mcq": {},
        "fixed_prompts": {},
    }
    failed = evaluate_gate(failed_metrics, thresholds, license_ready=True)
    assert failed["internal_candidate_passed"] is False
