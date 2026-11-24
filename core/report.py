"""Pure helpers for turning model scores into the API response.

Kept free of PyTorch imports so it can be unit-tested on its own.

Scores are the outputs of torchxrayvision's pretrained DenseNet, which applies its own
sigmoid and then ``op_norm``: a score of 0.5 means the raw probability equals that
pathology's operating-point threshold, below 0.5 it is under the threshold and above 0.5
over it. The bands below are display choices built on that: they are not calibrated
probabilities and carry no clinical meaning.
"""

from __future__ import annotations

from collections.abc import Sequence

OPERATING_POINT = 0.5
HIGH_THRESHOLD = 0.7
MODERATE_THRESHOLD = OPERATING_POINT
REPORT_MIN_PROBABILITY = 0.1

STATUS_HIGH = "HIGH SCORE"
STATUS_ELEVATED = "ELEVATED SCORE"
STATUS_NONE = "NO ELEVATED SCORE"

Prediction = tuple[str, float]

FALLBACK_EXPLANATION = {
    "summary": "The model produced pathology scores for this image.",
    "meaning": "These are uncalibrated outputs of a pretrained research model, not findings.",
    "recommendation": "No recommendation is given: this is research software, not a medical "
    "device, and its scores are not a diagnosis.",
}


def risk_level(probability: float) -> str:
    """Display band for one score: HIGH (> 0.7), MODERATE (>= 0.5, the operating point), LOW."""
    if probability > HIGH_THRESHOLD:
        return "HIGH"
    if probability >= MODERATE_THRESHOLD:
        return "MODERATE"
    return "LOW"


def overall_status(predictions: Sequence[Prediction]) -> str:
    """Status label driven by the highest score. It describes the score, not the patient."""
    if not predictions:
        return STATUS_NONE
    top = max(float(p) for _, p in predictions)
    if top > HIGH_THRESHOLD:
        return STATUS_HIGH
    if top >= MODERATE_THRESHOLD:
        return STATUS_ELEVATED
    return STATUS_NONE


def build_predictions(predictions: Sequence[Prediction]) -> list[dict]:
    return [
        {"disease": name, "confidence": round(float(prob) * 100, 2), "risk": risk_level(float(prob))}
        for name, prob in predictions
    ]


def describe_for_prompt(predictions: Sequence[Prediction], limit: int = 3) -> str:
    """Short text listing the scores worth mentioning to the language model."""
    significant = [(d, float(p)) for d, p in predictions if float(p) > REPORT_MIN_PROBABILITY][:limit]
    if not significant:
        return "No pathology score above 10%"
    return ", ".join(f"{d} ({p * 100:.1f}%)" for d, p in significant)


def parse_report(text: str) -> dict[str, str]:
    """Extract ``Summary:`` / ``Meaning:`` / ``Recommendation:`` lines from LLM output.

    Missing sections fall back to the fixed, non-diagnostic wording.
    """
    result = dict(FALLBACK_EXPLANATION)
    for raw in text.splitlines():
        line = raw.strip().lstrip("*#- ").strip()
        key, sep, value = line.partition(":")
        key = key.strip().strip("*").lower()
        value = value.strip().strip("*").strip()
        if sep and value and key in result:
            result[key] = value
    return result
