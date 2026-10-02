# src/rules/urgency.py
from __future__ import annotations

from typing import Iterable


def _has_any(text: str, terms: Iterable[str]) -> bool:
    return any(t in text for t in terms)


def _negated(text: str, phrase: str) -> bool:
    negators = [
        "ما في",
        "مافي",
        "مفيش",
        "بدون",
        "لا يوجد",
        "مو موجود",
        "ما عندي",
        "ما عندنا",
    ]
    return any((n + " " + phrase) in text for n in negators) or any((n + phrase) in text for n in negators)


def determine_urgency(diagnosis: str, severity: str, text: str) -> str:
    text = (text or "").lower()

    if diagnosis in {"خراج سني", "التهاب عصب غير عكوس"}:
        return "Urgent"

    denies_swelling = _negated(text, "تورم") or _negated(text, "ورم")
    denies_pus = _negated(text, "قيح") or _negated(text, "صديد") or _negated(text, "خراج")

    swelling = _has_any(text, ["تورم", "منتفخ", "انتفاخ", "ورم"]) and not denies_swelling
    pus = _has_any(text, ["قيح", "صديد", "خراج"]) and not denies_pus

    night = _has_any(text, ["ألم بالليل", "يزداد بالليل", "ألم ليلي", "يوقظني من النوم"]) and not (
        _negated(text, "ألم بالليل") or _negated(text, "ألم ليلي")
    )

    unrelieved = _has_any(text, ["ألم مستمر", "لا يهدأ", "لا يزول", "مستمر طول اليوم", "يزداد مع الوقت", "يوجع بقوة"])
    systemic = _has_any(text, ["حمى", "حرارة", "انتفاخ الوجه", "ورم الوجه", "صعوبة فتح الفم", "تعب عام"])

    if denies_swelling and denies_pus:
        if severity == "High" and night and unrelieved:
            return "Urgent"
        return "Non-Urgent"

    if pus:
        return "Urgent"
    if swelling and unrelieved:
        return "Urgent"
    if systemic and (swelling or pus):
        return "Urgent"
    if severity == "High" and night and unrelieved:
        return "Urgent"

    return "Non-Urgent"
