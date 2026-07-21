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
    top_bigram_mass,
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
        ("ab" * 400, "degenerate_ngram"),
        ("啊" * 300, "degenerate_ngram"),
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


def test_long_english_and_code_are_not_penalised_for_a_small_alphabet() -> None:
    """The second calibration failure, and the more damaging one.

    A distinct-character-ratio rule put every long English or code document
    near its threshold, because an alphabet has 26 letters where Chinese has
    thousands. It was removing 2.2% of the corpus, concentrated in the exact
    domain the model scores worst on.
    """
    # Genuinely varied prose. A repeated paragraph would trip the repetition
    # rule for the right reason and prove nothing about script bias.
    english = " ".join(
        [
            "Can you suggest a code snippet that deletes a specific record from a table?",
            "Certainly. Use a DELETE statement whose WHERE clause identifies the row by",
            "its primary key, and wrap the whole thing in a transaction so an accidental",
            "match can be rolled back before it is committed anywhere permanent.",
            "Before running it against production, check how many rows the equivalent",
            "SELECT returns; a missing predicate silently matches everything.",
            "Foreign keys referencing that row will either block the delete or cascade,",
            "depending on how the constraint was declared when the schema was created.",
            "Soft deletion is often preferable for audit purposes: mark a status column",
            "instead, and exclude those rows from ordinary queries through a view.",
            "Finally, confirm that any cached aggregate derived from the table is",
            "invalidated, otherwise dashboards keep reporting the vanished record.",
        ]
    )

    assert quality_reasons(english, THRESHOLDS) == []


def test_the_surviving_degeneracy_rule_is_script_neutral() -> None:
    """Both removed rules measured inventory coverage, which is script-relative.

    Over 60,000 real rows the bigram-diversity rule dropped 6.07% of English
    and code documents against 0.00% of Chinese. top_bigram_mass measures
    concentration instead: 90th percentile 0.047 for Chinese, 0.048 for
    English, so one threshold means the same thing in both.
    """
    chinese = "深度学习模型的训练需要大量算力与高质量语料，数据管线决定有效样本数量。"
    english = "Training a language model needs compute and a carefully filtered corpus."

    assert abs(top_bigram_mass(chinese) - top_bigram_mass(english)) < 0.1


def test_long_chinese_prose_survives_the_degeneracy_rules() -> None:
    """The case the first calibration got wrong.

    A 0.5 repetition threshold removed 11% of the corpus with a median dropped
    length of 729 characters against 269 kept: it was a length filter wearing a
    quality filter's name, and it deleted the longest documents.
    """
    prose = (
        "假装你是一个财务顾问，帮助我制定有效的理财方案。好的，我可以为您提供以下建议，"
        "以帮助您制定有效的理财方案。首先，您需要制定一个详细的预算计划，以便了解自己的"
        "收入和支出情况，这可以帮助您确定可用于投资的资金量。其次，建立应急储备金，通常"
        "建议覆盖三到六个月的生活开支。第三，根据风险承受能力配置资产，年轻时可适当提高"
        "权益类比例，临近退休则应逐步转向稳健品种。最后，定期复盘并根据人生阶段调整方案。"
    ) * 3

    assert quality_reasons(prose, THRESHOLDS) == []


def test_whitespace_runs_do_not_condemn_a_formatted_document() -> None:
    """Indentation is formatting noise, not content degeneracy."""
    body = "\n".join(
        " " * 20 + line
        for line in ["五月的风轻轻吹动", "恋人的心相约桥头", "心与心距离渐近", "你我并肩看晚霞"]
    )
    poem = "帮我生成一首爱情诗吧！\n" + body

    assert "degenerate_ngram" not in quality_reasons(poem, THRESHOLDS)


def test_padding_is_still_caught() -> None:
    assert "degenerate_ngram" in quality_reasons("txgfjhk" + "l" * 300, THRESHOLDS)


def test_top_bigram_mass_separates_prose_from_padding() -> None:
    prose = "深度学习模型的训练需要大量算力与高质量语料，数据管线决定有效样本数量。"

    assert top_bigram_mass(prose) < 0.2
    assert top_bigram_mass("l" * 300) > 0.9


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
