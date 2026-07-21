"""Fail-closed candidate evaluation for the team pretraining run."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from python_starter.core.data_contract import iter_jsonl, sha256_file, write_json_atomic
from python_starter.core.dataset import JsonlPackedDataset, collate_fn
from python_starter.core.inference import load_checkpoint_model
from python_starter.core.model import TransformerLM
from python_starter.core.quality_filters import top_bigram_mass
from python_starter.core.tokenizer import load_tokenizer

DOMAIN_NAMES = (
    "code_like",
    "english_or_code_heavy",
    "length_long",
    "math_like",
    "quality_pass",
    "repeat_high",
)


def repeated_bigram_ratio(text: str) -> float:
    """Measure repeated non-whitespace character bigrams."""
    characters = [value for value in text if not value.isspace()]
    if len(characters) < 8:
        return 0.0
    bigrams = ["".join(characters[index : index + 2]) for index in range(len(characters) - 1)]
    return 1.0 - len(set(bigrams)) / len(bigrams)


def _ascii_alpha_ratio(text: str) -> float:
    characters = [value for value in text if not value.isspace()]
    if not characters:
        return 0.0
    return sum(value.isascii() and value.isalpha() for value in characters) / len(characters)


def classify_domains(text: str) -> set[str]:
    """Assign stable diagnostic domains from observable text properties."""
    lowered = text.lower()
    domains: set[str] = set()
    if any(
        marker in lowered
        for marker in (
            "def ",
            "class ",
            "import ",
            "function ",
            "public static",
            "#include",
            "```",
        )
    ):
        domains.add("code_like")
    if _ascii_alpha_ratio(text) >= 0.55:
        domains.add("english_or_code_heavy")
    if len(text) >= 2048:
        domains.add("length_long")
    if re.search(r"(?:\\[a-z]+|[=+\-*/^]|\d{2,})", text):
        domains.add("math_like")
    repetition = repeated_bigram_ratio(text)
    if repetition >= 0.2:
        domains.add("repeat_high")
    if len(text) >= 200 and repetition < 0.2:
        domains.add("quality_pass")
    return domains


def _autocast(device: torch.device) -> AbstractContextManager[Any]:
    return torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    )


@torch.inference_mode()
def packed_loss(
    model: TransformerLM,
    tokenizer: PreTrainedTokenizerBase,
    data_path: Path,
    device: torch.device,
    *,
    max_length: int,
    batch_size: int,
    max_batches: int,
) -> dict[str, float | int]:
    """Calculate weighted next-token loss on deterministic packed blocks."""
    dataset = JsonlPackedDataset(data_path, tokenizer, max_length=max_length)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    nll_sum = 0.0
    tokens = 0
    batches = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        with _autocast(device):
            logits, _ = model(input_ids)
            loss_sum = functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
        token_count = int((labels != -100).sum().item())
        nll_sum += float(loss_sum.item())
        tokens += token_count
        batches += 1
        if batches >= max_batches:
            break
    if not tokens:
        raise ValueError(f"no evaluation tokens found in {data_path}")
    return {
        "loss": nll_sum / tokens,
        "perplexity": math.exp(min(nll_sum / tokens, 20)),
        "tokens": tokens,
        "batches": batches,
    }


def _collect_domain_blocks(
    data_path: Path,
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int,
    max_blocks: int,
) -> dict[str, list[list[int]]]:
    blocks: dict[str, list[list[int]]] = defaultdict(list)
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("tokenizer must define eos_token_id")

    for _, row in iter_jsonl(data_path):
        text = row.get("text")
        if not isinstance(text, str):
            continue
        domains = classify_domains(text)
        if not domains:
            continue
        tokens = tokenizer.encode(text, add_special_tokens=False) + [eos_token_id]
        for offset in range(0, len(tokens) - 1, max_length):
            values = tokens[offset : offset + max_length + 1]
            if len(values) < 2:
                continue
            for domain in domains:
                if len(blocks[domain]) < max_blocks:
                    blocks[domain].append(values)
        if all(len(blocks[name]) >= max_blocks for name in DOMAIN_NAMES):
            break
    return blocks


@torch.inference_mode()
def domain_losses(
    model: TransformerLM,
    tokenizer: PreTrainedTokenizerBase,
    data_path: Path,
    device: torch.device,
    *,
    max_length: int,
    batch_size: int,
    max_blocks: int,
) -> tuple[dict[str, float], dict[str, int]]:
    """Evaluate property-derived domain slices from the held-out test split."""
    blocks = _collect_domain_blocks(
        data_path,
        tokenizer,
        max_length=max_length,
        max_blocks=max_blocks,
    )
    losses: dict[str, float] = {}
    counts: dict[str, int] = {}
    pad_token_id = tokenizer.pad_token_id or 0
    for domain in DOMAIN_NAMES:
        values = blocks.get(domain, [])
        nll_sum = 0.0
        token_count = 0
        for offset in range(0, len(values), batch_size):
            batch_values = values[offset : offset + batch_size]
            width = max(len(item) for item in batch_values)
            input_ids = torch.full(
                (len(batch_values), width - 1),
                pad_token_id,
                dtype=torch.long,
                device=device,
            )
            labels = torch.full_like(input_ids, -100)
            for row_index, item in enumerate(batch_values):
                item_tensor = torch.tensor(item, dtype=torch.long, device=device)
                input_ids[row_index, : len(item) - 1] = item_tensor[:-1]
                labels[row_index, : len(item) - 1] = item_tensor[1:]
            with _autocast(device):
                logits, _ = model(input_ids)
                loss_sum = functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
            nll_sum += float(loss_sum.item())
            token_count += int((labels != -100).sum().item())
        if token_count:
            losses[domain] = nll_sum / token_count
            counts[domain] = len(values)
    return losses, counts


@torch.inference_mode()
def evaluate_mcq(
    model: TransformerLM,
    tokenizer: PreTrainedTokenizerBase,
    path: Path,
    device: torch.device,
    *,
    max_length: int,
) -> dict[str, Any]:
    """Score MCQ choices by mean continuation negative log-likelihood."""
    rows: list[dict[str, Any]] = []
    correct = 0
    for line_number, item in iter_jsonl(path):
        prompt = item.get("prompt")
        choices = item.get("choices")
        answer = item.get("answer")
        if not isinstance(prompt, str) or not isinstance(choices, list):
            raise ValueError(f"{path}:{line_number}: invalid MCQ schema")
        if not isinstance(answer, int) or not 0 <= answer < len(choices):
            raise ValueError(f"{path}:{line_number}: invalid answer index")

        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        scores: list[float] = []
        for choice in choices:
            choice_ids = tokenizer.encode(str(choice), add_special_tokens=False)
            full_ids = (prompt_ids + choice_ids)[-max_length:]
            prompt_tokens_kept = max(0, len(full_ids) - len(choice_ids))
            input_ids = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
            labels = torch.tensor([full_ids[1:]], dtype=torch.long, device=device)
            mask_until = max(0, prompt_tokens_kept - 1)
            labels[:, :mask_until] = -100
            with _autocast(device):
                logits, _ = model(input_ids)
                score = functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100,
                )
            scores.append(float(score.item()))
        prediction = min(range(len(scores)), key=scores.__getitem__)
        logits = torch.tensor([-value for value in scores], dtype=torch.float64)
        probabilities = torch.softmax(logits, dim=0).tolist()
        is_correct = prediction == answer
        correct += int(is_correct)
        rows.append(
            {
                "id": item.get("id", ""),
                "category": item.get("category", ""),
                "answer": answer,
                "prediction": prediction,
                "correct": is_correct,
                "choice_nll": scores,
                "choice_log_probability": [-value for value in scores],
                "choice_probability": probabilities,
            }
        )
    by_category: dict[str, dict[str, int | float]] = {}
    for row in rows:
        category = str(row["category"])
        summary = by_category.setdefault(category, {"correct": 0, "total": 0, "accuracy": 0.0})
        summary["correct"] = int(summary["correct"]) + int(bool(row["correct"]))
        summary["total"] = int(summary["total"]) + 1
    for summary in by_category.values():
        summary["accuracy"] = int(summary["correct"]) / max(int(summary["total"]), 1)
    interval = wilson_interval(correct, len(rows))
    return {
        "correct": correct,
        "total": len(rows),
        "accuracy": correct / max(len(rows), 1),
        "wilson_interval_95": list(interval),
        "by_category": by_category,
        "items": rows,
    }


def wilson_interval(
    correct: int, total: int, *, z: float = 1.959963984540054
) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for a binomial proportion."""
    if total <= 0 or not 0 <= correct <= total:
        raise ValueError("correct and total must describe a non-empty binomial sample")
    proportion = correct / total
    denominator = 1 + z**2 / total
    centre = (proportion + z**2 / (2 * total)) / denominator
    radius = (
        z * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2)) / denominator
    )
    return max(0.0, centre - radius), min(1.0, centre + radius)


def compare_mcq_items(
    baseline_items: Iterable[Mapping[str, Any]],
    candidate_items: Iterable[Mapping[str, Any]],
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 20260720,
) -> dict[str, Any]:
    """Compare paired MCQ predictions with bootstrap and exact McNemar evidence."""
    baseline = {str(item.get("id")): bool(item.get("correct")) for item in baseline_items}
    candidate = {str(item.get("id")): bool(item.get("correct")) for item in candidate_items}
    if not baseline or baseline.keys() != candidate.keys():
        raise ValueError("paired MCQ comparison requires the same non-empty item IDs")
    ids = sorted(baseline)
    baseline_values = [int(baseline[item_id]) for item_id in ids]
    candidate_values = [int(candidate[item_id]) for item_id in ids]
    baseline_only = sum(
        base == 1 and contender == 0
        for base, contender in zip(baseline_values, candidate_values, strict=True)
    )
    candidate_only = sum(
        base == 0 and contender == 1
        for base, contender in zip(baseline_values, candidate_values, strict=True)
    )
    discordant = baseline_only + candidate_only
    if discordant:
        tail = (
            sum(
                math.comb(discordant, value)
                for value in range(0, min(baseline_only, candidate_only) + 1)
            )
            / 2**discordant
        )
        mcnemar_p = min(1.0, 2 * tail)
    else:
        mcnemar_p = 1.0

    rng = random.Random(seed)
    deltas = []
    for _ in range(bootstrap_samples):
        sample = [rng.randrange(len(ids)) for _ in ids]
        delta = sum(candidate_values[index] - baseline_values[index] for index in sample) / len(ids)
        deltas.append(delta)
    deltas.sort()
    lower_index = int(0.025 * (bootstrap_samples - 1))
    upper_index = int(0.975 * (bootstrap_samples - 1))
    baseline_accuracy = sum(baseline_values) / len(ids)
    candidate_accuracy = sum(candidate_values) / len(ids)
    return {
        "schema_version": 1,
        "items": len(ids),
        "baseline_accuracy": baseline_accuracy,
        "candidate_accuracy": candidate_accuracy,
        "accuracy_delta": candidate_accuracy - baseline_accuracy,
        "paired_bootstrap": {
            "samples": bootstrap_samples,
            "seed": seed,
            "confidence_interval_95": [deltas[lower_index], deltas[upper_index]],
        },
        "mcnemar_exact": {
            "baseline_only_correct": baseline_only,
            "candidate_only_correct": candidate_only,
            "discordant": discordant,
            "p_value_two_sided": mcnemar_p,
        },
    }


def _keyword_score(prompt_id: str, completion: str) -> float:
    normalized = " ".join(completion.split())
    lowered = normalized.lower()
    if not normalized:
        return 0.0
    keyword_groups: dict[str, tuple[tuple[str, ...], ...]] = {
        "zh_basic_self": (("人工智能", "语言模型", "助手"), ("帮", "回答", "解释", "写")),
        "zh_explain_ml": (("数据", "例子", "学习"), ("规律", "模型", "算法")),
        "math_steps": (("8",), ("3x", "24", "29", "5")),
        "code_python_fib": (("def ", "def fib", "def fibonacci"), ("return",), ("for ", "while ")),
        "reason_compare": (("mha",), ("gqa",), ("mqa",), ("kv", "速度", "推理"), ("质量", "取舍")),
        "english_basic": (("blue",), ("light",), ("scatter",), ("sky",)),
        "safety_uncertain": (("不知道", "不确定", "无法确认"), ("查", "来源", "资料")),
    }
    if prompt_id == "zh_fact_nanjing":
        return float("南京" in normalized and "杭州" not in normalized)
    if prompt_id == "long_context_recall":
        return float("香蕉" in normalized and "苹果" not in normalized and "橙子" not in normalized)
    if prompt_id == "bad_case_repeat":
        sentences = {value for value in re.split(r"[。！？!?；;]\s*", normalized) if value}
        return min(len(sentences) / 5, 1.0)
    groups = keyword_groups.get(prompt_id)
    if groups is None:
        return 0.5
    matched = sum(any(keyword.lower() in lowered for keyword in group) for group in groups)
    return matched / len(groups)


@torch.inference_mode()
def evaluate_generation_health(
    model: TransformerLM,
    tokenizer: PreTrainedTokenizerBase,
    path: Path,
    device: torch.device,
    *,
    new_tokens: int,
    degeneracy_max: float = 0.2,
) -> dict[str, Any]:
    """Measure greedy-decoding degeneracy at a fixed generation length.

    Two defects in the earlier probe made its count incomparable across models.

    Early stopping was a free pass: generation halted at the end-of-sequence
    token, so a model that emitted it after seven characters produced a
    completion with no repeated bigrams at all and scored a perfect zero, while
    a model that wrote a paragraph was measured on the paragraph. Generation
    here always runs the full ``new_tokens`` budget, so every model is judged on
    the same amount of text. Whether a model stops on its own is a real and
    separate property, reported as ``natural_stop_rate`` rather than folded into
    the degeneracy figure.

    The degeneracy statistic is the share of bigrams taken by the single most
    frequent one, not the fraction of bigrams that repeat. The latter counts how
    much of a script's bigram inventory a text uses, and an alphabet has 26
    letters where Chinese has thousands, so it scores English far higher for
    reasons that have nothing to do with degeneration. Measured over 60,000 real
    corpus rows the concentration statistic sits at a 90th percentile of 0.047
    for Chinese and 0.048 for English.

    The legacy repeated-bigram ratio is still recorded per item, so the two can
    be compared on the same completions rather than across evaluations.
    """
    eos_token_id = tokenizer.eos_token_id
    rows: list[dict[str, Any]] = []
    for _, item in iter_jsonl(path):
        prompt = str(item["prompt"])
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        output = model.generate(
            input_ids,
            max_new_tokens=new_tokens,
            temperature=0,
            eos_token_id=None,
        )
        completion_ids = output[0, len(prompt_ids) :].tolist()
        natural_stop_index = (
            completion_ids.index(eos_token_id)
            if eos_token_id is not None and eos_token_id in completion_ids
            else None
        )
        completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
        if not isinstance(completion, str):
            raise TypeError("tokenizer returned a batched decode result")
        rows.append(
            {
                "id": item.get("id", ""),
                "category": item.get("category", ""),
                "prompt": prompt,
                "completion": completion,
                "generated_tokens": len(completion_ids),
                "natural_stop_token_index": natural_stop_index,
                "top_bigram_mass": top_bigram_mass(completion),
                "repeated_bigram_ratio": repeated_bigram_ratio(completion),
            }
        )

    degenerate = [row for row in rows if float(row["top_bigram_mass"]) > degeneracy_max]
    stopped = [row for row in rows if row["natural_stop_token_index"] is not None]
    return {
        "probes": len(rows),
        "new_tokens": new_tokens,
        "degeneracy_max": degeneracy_max,
        "degenerate_count": len(degenerate),
        "degenerate_rate": len(degenerate) / max(len(rows), 1),
        "natural_stop_rate": len(stopped) / max(len(rows), 1),
        "mean_top_bigram_mass": (
            sum(float(row["top_bigram_mass"]) for row in rows) / max(len(rows), 1)
        ),
        "by_category": _degeneracy_by_category(rows, degeneracy_max),
        "items": rows,
    }


def _degeneracy_by_category(rows: list[dict[str, Any]], limit: float) -> dict[str, Any]:
    """Per-category rates, so a script or genre bias is visible in the artifact."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["category"]), []).append(row)
    return {
        name: {
            "probes": len(items),
            "degenerate_count": sum(float(item["top_bigram_mass"]) > limit for item in items),
            "mean_top_bigram_mass": (
                sum(float(item["top_bigram_mass"]) for item in items) / max(len(items), 1)
            ),
        }
        for name, items in sorted(grouped.items())
    }


@torch.inference_mode()
def evaluate_fixed_prompts(
    model: TransformerLM,
    tokenizer: PreTrainedTokenizerBase,
    path: Path,
    device: torch.device,
    *,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Generate deterministic fixed-prompt samples and heuristic regression scores."""
    rows: list[dict[str, Any]] = []
    for _, item in iter_jsonl(path):
        prompt = str(item["prompt"])
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0,
            eos_token_id=tokenizer.eos_token_id,
        )
        completion_ids = output[0, len(prompt_ids) :].tolist()
        completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
        if not isinstance(completion, str):
            raise TypeError("tokenizer returned a batched decode result")
        repetition = repeated_bigram_ratio(completion)
        score = _keyword_score(str(item.get("id", "")), completion)
        if repetition > 0.45:
            score *= 0.7
        rows.append(
            {
                "id": item.get("id", ""),
                "category": item.get("category", ""),
                "prompt": prompt,
                "completion": completion,
                "score": score,
                "repeated_bigram_ratio": repetition,
            }
        )
    return {
        "mean_score": sum(float(row["score"]) for row in rows) / max(len(rows), 1),
        "repetition_count": sum(float(row["repeated_bigram_ratio"]) > 0.45 for row in rows),
        "items": rows,
    }


def source_tree_sha256(root: Path, paths: Iterable[Path]) -> str:
    """Hash relevant source/config files with their repository-relative names."""
    digest = hashlib.sha256()
    files: list[tuple[str, Path]] = []
    for input_path in paths:
        for path in input_path.rglob("*") if input_path.is_dir() else [input_path]:
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            try:
                label = path.relative_to(root).as_posix()
            except ValueError:
                relative = (
                    path.relative_to(input_path)
                    if input_path.is_dir()
                    else Path(path.name)
                )
                label = (Path(input_path.name) / relative).as_posix()
            files.append((label, path))
    for label, path in sorted(files):
        digest.update(label.encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def evaluate_gate(
    metrics: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    *,
    license_ready: bool,
) -> dict[str, Any]:
    """Apply a frozen stage-specific threshold configuration."""
    gate_kind = thresholds.get("gate_kind", "legacy")
    if gate_kind == "pretrain":
        return _evaluate_pretrain_gate(metrics, thresholds, license_ready=license_ready)
    if gate_kind == "posttrain":
        return _evaluate_posttrain_gate(metrics, thresholds)
    return _evaluate_legacy_gate(metrics, thresholds, license_ready=license_ready)


def _base_pretrain_failures(
    metrics: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> list[str]:
    failures: list[str] = []

    for split in ("strict_val", "strict_test"):
        value = metrics.get(split, {}).get("loss")
        maximum = thresholds.get(f"{split}_max")
        if (
            not isinstance(value, (int, float))
            or not isinstance(maximum, (int, float))
            or not math.isfinite(value)
            or value > maximum
        ):
            failures.append(split)

    domain_metrics = metrics.get("domain", {})
    domain_thresholds = thresholds.get("domain_max", {})
    for name in DOMAIN_NAMES:
        value = domain_metrics.get(name)
        maximum = domain_thresholds.get(name)
        if (
            not isinstance(value, (int, float))
            or not isinstance(maximum, (int, float))
            or not math.isfinite(value)
            or value > maximum
        ):
            failures.append(f"domain.{name}")

    return failures


def _evaluate_pretrain_gate(
    metrics: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    *,
    license_ready: bool,
) -> dict[str, Any]:
    failures = _base_pretrain_failures(metrics, thresholds)
    mcq = metrics.get("mcq", {})
    mcq_accuracy = mcq.get("accuracy")
    minimum_accuracy = thresholds.get("mcq_accuracy_min")
    if (
        not isinstance(mcq_accuracy, (int, float))
        or not isinstance(minimum_accuracy, (int, float))
        or mcq_accuracy < minimum_accuracy
    ):
        failures.append("mcq.accuracy")
    minimum_lower_bound = thresholds.get("mcq_wilson_lower_bound_min")
    if minimum_lower_bound is not None:
        interval = mcq.get("wilson_interval_95")
        if (
            not isinstance(minimum_lower_bound, (int, float))
            or not isinstance(interval, list)
            or len(interval) != 2
            or not isinstance(interval[0], (int, float))
            or interval[0] <= minimum_lower_bound
        ):
            failures.append("mcq.wilson_lower_bound")

    generation = metrics.get("generation", metrics.get("fixed_prompts", {}))
    repetition_count = generation.get("repetition_count")
    if not isinstance(repetition_count, int) or repetition_count > thresholds.get(
        "repetition_count_max", -1
    ):
        failures.append("generation.repetition")

    internal_candidate_passed = not failures
    return {
        "schema_version": 2,
        "gate_kind": "pretrain",
        "internal_candidate_passed": internal_candidate_passed,
        "public_release_passed": internal_candidate_passed and license_ready,
        "public_release_license_ready": license_ready,
        "failures": failures,
    }


def _evaluate_posttrain_gate(
    metrics: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    failures: list[str] = []
    fixed_score = metrics.get("fixed_prompts", {}).get("mean_score")
    if not isinstance(fixed_score, (int, float)) or fixed_score < thresholds.get(
        "fixed_score_min", math.inf
    ):
        failures.append("fixed_prompts")
    repetition_count = metrics.get("fixed_prompts", {}).get("repetition_count")
    if not isinstance(repetition_count, int) or repetition_count > thresholds.get(
        "repetition_count_max", -1
    ):
        failures.append("repetition")
    return {
        "schema_version": 1,
        "gate_kind": "posttrain",
        "posttrain_passed": not failures,
        "failures": failures,
    }


def _evaluate_legacy_gate(
    metrics: Mapping[str, Any],
    thresholds: Mapping[str, Any],
    *,
    license_ready: bool,
) -> dict[str, Any]:
    failures = _base_pretrain_failures(metrics, thresholds)
    mcq_accuracy = metrics.get("mcq", {}).get("accuracy")
    if not isinstance(mcq_accuracy, (int, float)) or mcq_accuracy < thresholds.get(
        "mcq_accuracy_min", math.inf
    ):
        failures.append("mcq")

    fixed_score = metrics.get("fixed_prompts", {}).get("mean_score")
    if not isinstance(fixed_score, (int, float)) or fixed_score < thresholds.get(
        "fixed_score_min", math.inf
    ):
        failures.append("fixed_prompts")

    repetition_count = metrics.get("fixed_prompts", {}).get("repetition_count")
    if not isinstance(repetition_count, int) or repetition_count > thresholds.get(
        "repetition_count_max", -1
    ):
        failures.append("repetition")

    internal_candidate_passed = not failures
    return {
        "schema_version": 1,
        "internal_candidate_passed": internal_candidate_passed,
        "public_release_passed": internal_candidate_passed and license_ready,
        "public_release_license_ready": license_ready,
        "failures": failures,
    }


def run_candidate_evaluation(
    *,
    root: Path,
    checkpoint_path: Path,
    tokenizer_path: Path,
    validation_path: Path,
    test_path: Path,
    audit_path: Path,
    thresholds_path: Path,
    mcq_path: Path,
    fixed_prompts_path: Path,
    output_path: Path,
    device_name: str = "auto",
    max_length: int = 384,
    batch_size: int = 32,
    strict_batches: int = 250,
    domain_blocks: int = 250,
    fixed_new_tokens: int = 128,
    generation_prompts_path: Path | None = None,
    health_prompts_path: Path | None = None,
    health_new_tokens: int = 128,
    posttrain_thresholds_path: Path | None = None,
) -> dict[str, Any]:
    """Produce a complete evidence bundle and release decision."""
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "passed":
        raise ValueError("dataset audit must pass before candidate evaluation")
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
    model, checkpoint, device = load_checkpoint_model(checkpoint_path, device_name)
    tokenizer = load_tokenizer(str(tokenizer_path))

    strict_val = packed_loss(
        model,
        tokenizer,
        validation_path,
        device,
        max_length=max_length,
        batch_size=batch_size,
        max_batches=strict_batches,
    )
    strict_test = packed_loss(
        model,
        tokenizer,
        test_path,
        device,
        max_length=max_length,
        batch_size=batch_size,
        max_batches=strict_batches,
    )
    domain, domain_counts = domain_losses(
        model,
        tokenizer,
        test_path,
        device,
        max_length=max_length,
        batch_size=batch_size,
        max_blocks=domain_blocks,
    )
    mcq = evaluate_mcq(
        model,
        tokenizer,
        mcq_path,
        device,
        max_length=max_length,
    )
    fixed = evaluate_fixed_prompts(
        model,
        tokenizer,
        fixed_prompts_path,
        device,
        max_new_tokens=fixed_new_tokens,
    )
    generation = (
        evaluate_fixed_prompts(
            model,
            tokenizer,
            generation_prompts_path,
            device,
            max_new_tokens=fixed_new_tokens,
        )
        if generation_prompts_path is not None
        else fixed
    )
    # Recorded unconditionally when probes are supplied. The legacy blocks stay
    # so the old and new instruments can be compared on one checkpoint instead
    # of across evaluations.
    generation_health = (
        evaluate_generation_health(
            model,
            tokenizer,
            health_prompts_path,
            device,
            new_tokens=health_new_tokens,
        )
        if health_prompts_path is not None
        else None
    )

    tokenizer_identity = source_tree_sha256(root, [tokenizer_path])
    provenance = {
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_step": checkpoint.get("progress", {}).get("global_step"),
        "checkpoint_training_view": checkpoint.get("run_config", {}).get("training_view", {}),
        "tokenizer_sha256": tokenizer_identity,
        "source_sha256": source_tree_sha256(
            root,
            [root / "src", root / "scripts", root / "configs"],
        ),
        "data_sha256": {
            "manifest": audit.get("dataset_spec_sha256"),
            "train": audit.get("splits", {}).get("train", {}).get("sha256"),
            "validation": sha256_file(validation_path),
            "test": sha256_file(test_path),
        },
        "evaluation_dataset_id": audit.get("dataset_id"),
        "evaluation_dataset_revision": audit.get("dataset_revision"),
        "evaluation_data_sha256": {
            "mcq": sha256_file(mcq_path),
            "fixed_prompts": sha256_file(fixed_prompts_path),
            "generation_prompts": (
                sha256_file(generation_prompts_path)
                if generation_prompts_path is not None
                else sha256_file(fixed_prompts_path)
            ),
            "thresholds": sha256_file(thresholds_path),
            "posttrain_thresholds": (
                sha256_file(posttrain_thresholds_path)
                if posttrain_thresholds_path is not None
                else None
            ),
        },
    }
    metrics = {
        "strict_val": strict_val,
        "strict_test": strict_test,
        "domain": domain,
        "domain_blocks": domain_counts,
        "mcq": mcq,
        "generation": generation,
        "fixed_prompts": fixed,
        **({"generation_health": generation_health} if generation_health else {}),
        "provenance": provenance,
    }
    license_ready = audit.get("gates", {}).get("public_release_license_ready") is True
    gate = evaluate_gate(metrics, thresholds, license_ready=license_ready)
    gates: dict[str, Any] = {"pretrain": gate}
    posttrain_thresholds = None
    if posttrain_thresholds_path is not None:
        posttrain_thresholds = json.loads(posttrain_thresholds_path.read_text(encoding="utf-8"))
        gates["posttrain"] = evaluate_gate(
            metrics,
            posttrain_thresholds,
            license_ready=False,
        )
    result = {
        "schema_version": 2,
        "metrics": metrics,
        "thresholds": {
            "pretrain": thresholds,
            "posttrain": posttrain_thresholds,
        },
        "gates": gates,
        "gate": gate,
    }
    write_json_atomic(output_path, result)
    return result
