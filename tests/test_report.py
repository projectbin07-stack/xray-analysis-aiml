import pytest

from core.report import (
    FALLBACK_EXPLANATION,
    build_predictions,
    describe_for_prompt,
    overall_status,
    parse_report,
    risk_level,
)


@pytest.mark.parametrize(
    ("prob", "expected"),
    [
        (0.0, "LOW"),
        (0.45, "LOW"),
        (0.499, "LOW"),
        (0.5, "MODERATE"),  # 0.5 = raw probability at the model's operating point
        (0.7, "MODERATE"),
        (0.71, "HIGH"),
        (1.0, "HIGH"),
    ],
)
def test_risk_level_boundaries(prob, expected):
    assert risk_level(prob) == expected


def test_overall_status_uses_highest_score():
    assert overall_status([("A", 0.2), ("B", 0.1)]) == "NO ELEVATED SCORE"
    assert overall_status([("A", 0.45), ("B", 0.1)]) == "NO ELEVATED SCORE"
    assert overall_status([("A", 0.5), ("B", 0.1)]) == "ELEVATED SCORE"
    assert overall_status([("A", 0.1), ("B", 0.9)]) == "HIGH SCORE"
    assert overall_status([]) == "NO ELEVATED SCORE"


def test_statuses_and_fallback_text_are_not_clinical_directives():
    words = " ".join([*FALLBACK_EXPLANATION.values(), "HIGH SCORE ELEVATED SCORE NO ELEVATED SCORE"])
    for phrase in ("critical", "normal", "attention", "immediately", "consult", "seek"):
        assert phrase not in words.lower()
    assert "not a diagnosis" in FALLBACK_EXPLANATION["recommendation"]


def test_build_predictions_rounds_to_percent():
    out = build_predictions([("Effusion", 0.71234), ("Mass", 0.05)])
    assert out == [
        {"disease": "Effusion", "confidence": 71.23, "risk": "HIGH"},
        {"disease": "Mass", "confidence": 5.0, "risk": "LOW"},
    ]


def test_describe_for_prompt_filters_and_limits():
    preds = [("A", 0.9), ("B", 0.5), ("C", 0.3), ("D", 0.2), ("E", 0.05)]
    assert describe_for_prompt(preds) == "A (90.0%), B (50.0%), C (30.0%)"
    assert describe_for_prompt([("A", 0.05)]) == "No pathology score above 10%"


def test_parse_report_reads_plain_and_markdown_headings():
    text = "Summary: one\n**Meaning:** two\n- Recommendation: see a doctor: soon\nOther: ignored"
    assert parse_report(text) == {
        "summary": "one",
        "meaning": "two",
        "recommendation": "see a doctor: soon",
    }


def test_parse_report_keeps_safe_defaults_for_missing_sections():
    parsed = parse_report("Summary: only this\nMeaning:\n")
    assert parsed["summary"] == "only this"
    assert parsed["meaning"] == FALLBACK_EXPLANATION["meaning"]
    assert parsed["recommendation"] == FALLBACK_EXPLANATION["recommendation"]
    assert parse_report("") == FALLBACK_EXPLANATION
