"""Quality and PII filters for the pretraining corpus.

The pipeline previously went from raw text straight to deduplication and then
to tokenisation, with no quality stage and no PII stage at all. Both are listed
as mandatory in the reference standard, and the second is also a licensing
question for a corpus that may be published.

Every rule returns a reason string rather than a bare boolean, so a dropped row
can be accounted for instead of vanishing into a count.

A distinct-character-ratio rule was tried and removed. It correlated with
document length at -0.588 and, worse, with script: an alphabet has 26 letters
where Chinese has thousands, so every long English or code document sits near
the threshold while Chinese prose sits far above it. It was deleting 2.2% of
the corpus, concentrated in exactly the English and code material the model is
weakest on. :func:`top_bigram_mass` covers the degenerate case it was meant to
catch, without the bias.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

# Chinese mainland ID: 17 digits plus a checksum character. The checksum is
# verified rather than pattern-matched, because an unvalidated 18-digit pattern
# also matches order numbers and bank references, and dropping real training
# data on a false positive is its own cost.
_ID_CARD = re.compile(r"(?<!\d)(\d{17})([0-9Xx])(?!\d)")
_ID_WEIGHTS = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
_ID_CHECKSUM = "10X98765432"

# Mainland mobile numbers are 11 digits starting 1[3-9].
_PHONE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
# 16-19 digit card numbers, validated by Luhn for the same reason as above.
_BANK_CARD = re.compile(r"(?<!\d)\d{16,19}(?!\d)")

_CJK = re.compile(r"[一-鿿]")


@dataclass(frozen=True)
class QualityThresholds:
    """Frozen quality boundaries.

    ``min_characters`` stands in for the standard's "drop texts under 20
    tokens": tokenising to decide whether to keep a row would mean tokenising
    the corpus twice. On this tokenizer Chinese runs near one token per
    character and English near four characters per token, so 20 characters is
    the conservative equivalent.
    """

    min_characters: int = 20
    # 0.8 is the 99.5th percentile of 60,000 real rows. The earlier 0.5 sat at
    # the 89th percentile and removed 11% of the corpus, with a median dropped
    # length of 729 characters against 269 for kept rows: it was deleting the
    # longest and richest documents, not the degenerate ones.
    max_repetition_ratio: float = 0.8
    max_top_bigram_mass: float = 0.2
    max_replacement_character_ratio: float = 0.01


def _luhn_valid(digits: str) -> bool:
    total = 0
    for index, character in enumerate(reversed(digits)):
        value = int(character)
        if index % 2:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def _id_card_valid(body: str, check: str) -> bool:
    total = sum(int(digit) * weight for digit, weight in zip(body, _ID_WEIGHTS, strict=True))
    return _ID_CHECKSUM[total % 11] == check.upper()


def find_pii(text: str) -> dict[str, int]:
    """Count personal identifiers by kind.

    Checksums are verified for identity and card numbers so that ordinary
    long digit strings are not mistaken for personal data.
    """
    found: dict[str, int] = {}

    identity = sum(1 for body, check in _ID_CARD.findall(text) if _id_card_valid(body, check))
    if identity:
        found["id_card"] = identity

    phones = len(_PHONE.findall(text))
    if phones:
        found["phone"] = phones

    emails = len(_EMAIL.findall(text))
    if emails:
        found["email"] = emails

    cards = sum(
        1
        for candidate in _BANK_CARD.findall(text)
        # An 18-digit identity number would otherwise be counted twice.
        if _luhn_valid(candidate) and not _ID_CARD.fullmatch(candidate)
    )
    if cards:
        found["bank_card"] = cards

    return found


def _collapsed(text: str) -> str:
    """NFKC with whitespace runs collapsed.

    Degeneracy is measured on this rather than the raw text: a formatted poem
    or table carries long whitespace runs that dominate every repetition
    statistic while saying nothing about the content, and both rules below
    would otherwise reject perfectly good documents for their indentation.
    """
    return " ".join(unicodedata.normalize("NFKC", text).split())


def repetition_ratio(text: str) -> float:
    """Fraction of character bigrams that repeat an earlier bigram.

    Length-dependent by construction: on 60,000 real rows this correlates with
    document length at 0.545, because the space of common Chinese bigrams is
    finite and a longer text exhausts it. The threshold is therefore set from
    the measured distribution to catch structural junk only, and
    :func:`top_bigram_mass` carries the length-robust part of the job.
    """
    collapsed = _collapsed(text)
    if len(collapsed) < 3:
        return 0.0
    bigrams = [collapsed[index : index + 2] for index in range(len(collapsed) - 1)]
    return 1.0 - (len(set(bigrams)) / len(bigrams))


def top_bigram_mass(text: str) -> float:
    """Share of all bigrams taken by the single most frequent one.

    Natural language spreads its mass; degenerate text concentrates it. On the
    same 60,000 rows this correlates with length at -0.188, so unlike
    :func:`repetition_ratio` it does not double as a length filter.
    """
    from collections import Counter

    collapsed = _collapsed(text)
    if len(collapsed) < 3:
        return 0.0
    bigrams = [collapsed[index : index + 2] for index in range(len(collapsed) - 1)]
    return Counter(bigrams).most_common(1)[0][1] / len(bigrams)


def quality_reasons(text: str, thresholds: QualityThresholds) -> list[str]:
    """Every quality rule the text fails, or an empty list if it passes."""
    reasons: list[str] = []
    normalized = unicodedata.normalize("NFKC", text).strip()

    if len(normalized) < thresholds.min_characters:
        reasons.append("too_short")

    if normalized:
        replacements = normalized.count("�") / len(normalized)
        if replacements > thresholds.max_replacement_character_ratio:
            reasons.append("mojibake")

    if repetition_ratio(normalized) > thresholds.max_repetition_ratio:
        reasons.append("repetitive")

    if top_bigram_mass(normalized) > thresholds.max_top_bigram_mass:
        reasons.append("degenerate_ngram")

    return reasons


def evaluate_row(
    text: str,
    thresholds: QualityThresholds,
    *,
    drop_on_pii: bool = True,
) -> tuple[bool, dict[str, Any]]:
    """Decide whether to keep a row, with the reasons attached either way."""
    reasons = quality_reasons(text, thresholds)
    pii = find_pii(text)
    if pii and drop_on_pii:
        reasons.append("pii")
    return not reasons, {"reasons": reasons, "pii": pii, "characters": len(text)}


def language_profile(text: str) -> str:
    """Coarse script label, for reporting what a filter pass removed."""
    if not text:
        return "empty"
    cjk = len(_CJK.findall(text))
    ratio = cjk / len(text)
    if ratio > 0.3:
        return "zh"
    if ratio > 0.05:
        return "mixed"
    return "other"
