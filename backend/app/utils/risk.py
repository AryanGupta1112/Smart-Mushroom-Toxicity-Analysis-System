from __future__ import annotations


def probability_to_risk_score(probability: float) -> int:
    clipped = max(0.0, min(1.0, probability))
    return int(round(clipped * 100))


def risk_level_from_score(score: int) -> str:
    if score <= 39:
        return "Low Risk"
    if score <= 69:
        return "Medium Risk"
    return "High Risk"
