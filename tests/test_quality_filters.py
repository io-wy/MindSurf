"""Quality and PII filter tests."""

from __future__ import annotations

import pytest

from python_starter.core.quality_filters import (
    QualityThresholds,
    evaluate_row,
    find_pii,
    language_profile,
    quality_reasons,
    repetition_ratio,
)

THRESHOLDS = QualityThresholds()


def test_identity_number_needs_a_valid_checksum() -> None:
    # Real checksum: this is a structurally valid mainland ID.
    assert find_pii("联系人身份证 11010519491231002X 已登记") == {"id_card": 1}
    # Same shape, wrong check character: an order number, not personal data.
    assert "id_card" not in find_pii("订单编号 11010519491231002A 已登记")


def test_an_ordinary_long_number_is_not_read_as_a_card() -> None:
    assert find_pii("参数总量 8935829375928375923 已记录") == {}


def test_phone_and_email_are_found() -> None:
    found = find_pii("联系 13812345678 或 someone@example.com")

    assert found["phone"] == 1
    assert found["email"] == 1


def test_digits_embedded_in_a_longer_run_are_not_phones() -> None:
    assert "phone" not in find_pii("序列 913812345678999 结束")


@pytest.mark.parametrize(
    "text,reason",
    [
        ("太短", "too_short"),
        ("正常的中文段落" + "�" * 30, "mojibake"),
        ("ab" * 200, "repetitive"),
        ("啊" * 300, "low_character_diversity"),
    ],
)
def test_quality_rules_fire(text: str, reason: str) -> None:
    assert reason in quality_reasons(text, THRESHOLDS)


def test_ordinary_prose_passes_every_rule() -> None:
    text = (
        "深度学习模型的训练需要大量算力与高质量语料，"
        "数据管线的每一个环节都会影响最终的模型表现。"
    )

    assert quality_reasons(text, THRESHOLDS) == []


def test_short_text_is_not_penalised_for_low_diversity() -> None:
    # Distinctness is mechanically low when the denominator is small; the rule
    # must not turn that into a quality judgement.
    assert "low_character_diversity" not in quality_reasons("你好你好你好你好你好你好", THRESHOLDS)


def test_repetition_ratio_bounds() -> None:
    assert repetition_ratio("") == 0.0
    assert repetition_ratio("abcdefgh") == 0.0
    assert repetition_ratio("abababab") > 0.5


def test_evaluate_row_reports_why_it_dropped() -> None:
    keep, detail = evaluate_row("联系 13812345678 获取更多信息，欢迎咨询业务。", THRESHOLDS)

    assert keep is False
    assert "pii" in detail["reasons"]
    assert detail["pii"]["phone"] == 1


def test_pii_can_be_reported_without_dropping() -> None:
    keep, detail = evaluate_row(
        "联系 13812345678 获取更多信息，欢迎咨询业务。", THRESHOLDS, drop_on_pii=False
    )

    assert keep is True
    assert detail["pii"]["phone"] == 1


def test_language_profile() -> None:
    assert language_profile("这是一段中文文本内容") == "zh"
    assert language_profile("pure english text here") == "other"
    assert language_profile("") == "empty"
