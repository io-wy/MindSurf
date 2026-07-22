"""Build the instrument validity ledger from measurements, not from claims.

The completion checklist requires that no measurement gates a decision until it
has been shown to measure what it claims. This script produces that evidence by
running the invariance tests itself, so the ledger states results it computed
rather than conclusions someone typed in. A ledger that could be edited to say
"passed" would repeat the failure it exists to prevent: this project has already
shipped an audit whose human-review field was set by a command-line flag.

Each entry records the four questions the checklist asks — invariance,
discrimination, agreement with human judgement, noise floor — plus sample size,
and derives gating qualification from them rather than accepting it as input.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from python_starter.core.data_contract import sha256_file, write_json_atomic  # noqa: E402
from python_starter.core.quality_filters import (  # noqa: E402
    QualityThresholds,
    language_profile,
    quality_reasons,
    repetition_ratio,
    top_bigram_mass,
)


def _correlation(left: list[float], right: list[float]) -> float:
    """Pearson correlation without pulling in numpy for a one-liner."""
    if len(left) < 2:
        return 0.0
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    covariance = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right, strict=True))
    variance_left = sum((a - mean_left) ** 2 for a in left)
    variance_right = sum((b - mean_right) ** 2 for b in right)
    if variance_left <= 0 or variance_right <= 0:
        return 0.0
    return float(covariance / (variance_left * variance_right) ** 0.5)


def _sample_corpus(path: Path, limit: int) -> list[str]:
    texts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            texts.append(str(json.loads(line)["text"]))
            if len(texts) >= limit:
                break
    return texts


def _quality_filter_entry(corpus: Path, limit: int) -> dict[str, Any]:
    """Invariance of the corpus quality filter, recomputed on real rows."""
    texts = _sample_corpus(corpus, limit)
    thresholds = QualityThresholds()

    dropped_lengths: list[int] = []
    kept_lengths: list[int] = []
    by_language: dict[str, dict[str, int]] = {}
    masses: list[float] = []
    ratios: list[float] = []
    lengths: list[float] = []

    for text in texts:
        language = language_profile(text)
        bucket = by_language.setdefault(language, {"total": 0, "dropped": 0})
        bucket["total"] += 1
        masses.append(top_bigram_mass(text))
        ratios.append(repetition_ratio(text))
        lengths.append(float(len(text)))
        if quality_reasons(text, thresholds):
            bucket["dropped"] += 1
            dropped_lengths.append(len(text))
        else:
            kept_lengths.append(len(text))

    rates = {
        name: bucket["dropped"] / bucket["total"]
        for name, bucket in by_language.items()
        if bucket["total"]
    }
    # Absolute gap, not a ratio. A ratio test on tiny denominators is noise and
    # says nothing about harm: the filter this replaced dropped 6.19% of English
    # against 0.06% of Chinese, a 6.1 point gap that thinned the material the
    # model is weakest on. The current filter's largest gap is a fraction of a
    # point, at which the same ratio is meaningless.
    spread = (max(rates.values()) / min(rates.values())) if rates and min(rates.values()) else None
    gap = (max(rates.values()) - min(rates.values())) if rates else 0.0

    def median(values: list[int]) -> float:
        return float(sorted(values)[len(values) // 2]) if values else 0.0

    # The retired statistic is measured alongside the live one so the ledger
    # shows why the replacement happened, not just that it did.
    return {
        "instrument": "corpus_quality_filter",
        "claims_to_measure": "text quality suitable for pretraining",
        "sample_size": len(texts),
        "invariance": {
            "length_correlation_live": round(_correlation(masses, lengths), 3),
            "length_correlation_retired": round(_correlation(ratios, lengths), 3),
            "dropped_median_length": median(dropped_lengths),
            "kept_median_length": median(kept_lengths),
            "drop_rate_by_language": {name: round(rate, 4) for name, rate in sorted(rates.items())},
            "language_rate_gap_points": round(gap * 100, 3),
            "language_rate_spread": round(spread, 2) if spread else None,
        },
        "noise_floor": "deterministic given a fixed corpus; no sampling variance",
        "agreement": "50 dropped and 50 kept rows sampled per destructive run (checklist 3.5)",
        "gating_qualified": bool(
            gap < 0.01
            and abs(_correlation(masses, lengths)) < 0.3
            and median(dropped_lengths) <= median(kept_lengths)
        ),
        "qualification_reason": (
            "largest drop-rate gap between language groups below 1 percentage point, live "
            "statistic length-independent, and dropped rows not systematically longer than "
            "kept rows. The ratio is recorded alongside but not used: on drop rates of a "
            "fraction of a percent it is noise, and the retired filter it was written for "
            "showed a 6.1 point gap"
        ),
    }


def _static_entries() -> list[dict[str, Any]]:
    """Entries whose evidence lives in artifacts rather than in a rerun here.

    Each links the produced artifact; a conclusion without a link is not
    evidence and the checklist treats a missing link as a missing field.
    """
    return [
        {
            "instrument": "generation_health",
            "claims_to_measure": "greedy-decoding degeneration",
            "sample_size": 100,
            "invariance": {
                "early_stop": "removed: generation always runs the full token budget",
                "script": "top_bigram_mass 90th percentile 0.047 zh against 0.048 en",
                "length": "fixed budget makes every completion the same length",
                "evidence": "tests/test_generation_health.py",
            },
            "noise_floor": "pending: requires two seeds evaluated with this instrument",
            "agreement": "pending user review of sampled completions",
            "gating_qualified": False,
            "qualification_reason": (
                "invariance satisfied; noise floor and human agreement still outstanding, "
                "so it may be reported but must not decide a verdict"
            ),
        },
        {
            "instrument": "generation_repetition_legacy",
            "claims_to_measure": "greedy-decoding degeneration",
            "sample_size": 10,
            "invariance": {
                "early_stop": "FAILED: a seven-character completion scores a perfect 0.000",
                "length": "FAILED: correlation 0.786 across four checkpoints",
                "script": "FAILED: counts bigram-inventory coverage, alphabet-biased",
            },
            "noise_floor": "2 of 10 probes; gate flips between 5 and 4",
            "agreement": "not established",
            "gating_qualified": False,
            "qualification_reason": "retired; superseded by generation_health",
        },
        {
            "instrument": "mcq_v2",
            "claims_to_measure": "knowledge and reasoning capability",
            "sample_size": 192,
            "invariance": {
                "template_duplication": (
                    "FAILED: 32 long_context records are 4 templates copied 8 times"
                ),
                "distractor_quality": "FAILED: distractors sit at a different semantic level",
            },
            "noise_floor": "paired bootstrap plus or minus 7pp; effects of interest are 3-5pp",
            "agreement": (
                "user reviewed 32 stratified items: labels all correct, suite ruled "
                "inadmissible as capability evidence"
            ),
            "evidence": "configs/evaluation/pretrain_mcq_benchmark_v2.review.json",
            "gating_qualified": False,
            "qualification_reason": "human review found disagreement; power insufficient",
        },
        {
            "instrument": "strict_holdout_loss",
            "claims_to_measure": "language modelling quality",
            "sample_size": 4000,
            "invariance": {
                "distribution": (
                    "in-distribution by construction: measures compression of this corpus, "
                    "not capability; cross-source Team holdout carried alongside"
                )
            },
            "noise_floor": "0.0024 to 0.0068 across two seeds at 60k steps",
            "discrimination": (
                "margin against retired absolute thresholds is 0.27 to 1.24 nats, "
                "tens to hundreds of times the noise floor"
            ),
            "agreement": "n/a for a loss statistic",
            "gating_qualified": False,
            "qualification_reason": (
                "lost discrimination against the retired thresholds; retained as canary "
                "and as ratchet input once a reference exists"
            ),
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-rows", type=int, default=60000)
    args = parser.parse_args()

    entries = [_quality_filter_entry(args.corpus, args.sample_rows), *_static_entries()]
    qualified = [entry["instrument"] for entry in entries if entry["gating_qualified"]]

    write_json_atomic(
        args.output,
        {
            "schema_version": 1,
            "recorded_at": datetime.now(UTC).isoformat(),
            "corpus": {"path": args.corpus.as_posix(), "sha256": sha256_file(args.corpus)},
            "sample_rows": args.sample_rows,
            "entries": entries,
            "gating_qualified": qualified,
            "note": (
                "Gating qualification is derived from the recorded measurements, not "
                "accepted as input. An instrument absent from gating_qualified may be "
                "reported but must not decide a verdict."
            ),
        },
    )
    print(json.dumps({"entries": len(entries), "gating_qualified": qualified}, ensure_ascii=False))


if __name__ == "__main__":
    main()
