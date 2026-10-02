# src/rules/severity.py
"""Severity is derived from the FINAL diagnosis (and urgency), never from an intermediate label."""
from __future__ import annotations

SEVERITY_BY_DIAGNOSIS = {
    "خراج سني": "High",
    "التهاب عصب غير عكوس": "High",
    "التهاب عصب عكوس": "Moderate",
    "تسوس عميق": "Moderate",
    "تسوس سطحي": "Low",
    "التهاب لثة": "Low",
    "غير واضح حالياً": "Low",
}


def final_severity(diagnosis: str, urgency: str) -> str:
    """An 'Urgent' case is always High severity; otherwise severity follows the diagnosis."""
    if urgency == "Urgent":
        return "High"
    return SEVERITY_BY_DIAGNOSIS.get(diagnosis, "Low")
