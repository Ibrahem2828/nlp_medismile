# src/symptom_understanding/text_classifier.py
from __future__ import annotations

from typing import Iterable, Tuple


def _has_any(text: str, terms: Iterable[str]) -> bool:
    return any(t in text for t in terms)


def _has_all(text: str, terms: Iterable[str]) -> bool:
    return all(t in text for t in terms)


def _negated(text: str, phrase: str) -> bool:
    """Light negation handling for Arabic dialect/Fusha."""
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


def classify_text(text: str) -> Tuple[str, str]:
    """
    Clinical rule-based classifier (Arabic).
    Returns: (diagnosis_label, severity_level)
    Diagnoses: خراج سني، التهاب عصب غير عكوس، التهاب عصب عكوس، تسوس عميق، تسوس سطحي، التهاب لثة، أو غير واضح.
    """

    text = (text or "").strip()
    if not text:
        return "غير واضح حالياً", "Low"

    # Negation signals
    denies_swelling = _negated(text, "تورم") or _negated(text, "ورم")
    denies_pus = _negated(text, "قيح") or _negated(text, "صديد") or _negated(text, "خراج")

    # Core symptom groups
    swelling_terms = ["تورم", "منتفخ", "انتفاخ", "ورم"]
    pus_terms = ["قيح", "صديد", "خراج"]
    night_terms = ["ألم بالليل", "يزداد بالليل", "ألم ليلي", "يوقظني من النوم"]
    persistent_terms = ["ألم مستمر", "لا يهدأ", "لا يزول", "مستمر طول اليوم", "يزداد مع الوقت"]
    throbbing_terms = ["نبضي", "ينبض", "خافق"]
    severe_pain_terms = ["ألم شديد", "شديد جداً", "لا يحتمل", "يوجع بقوة"]
    stimulus_terms = ["مع البارد", "مع الساخن", "مع الحار", "مع الحلو", "مع المشروبات", "مع الأكل"]
    transient_terms = ["يختفي بسرعة", "يزول بسرعة", "يروح بعد ثواني", "يزول بعد إزالة المحفز"]
    chewing_terms = ["ألم عند المضغ", "ألم مع الضغط", "يوجع عند العض", "يوجع عند الأكل", "ألم عند الضغط"]
    cavity_terms = ["تسوس", "نخر", "ثقب", "حفرة", "تجويف"]
    gum_terms = ["لثة", "اللثة"]
    bleeding_terms = ["نزيف", "دم", "تنزف", "دم عند التفريش", "نزيف عند التفريش"]

    has_swelling = _has_any(text, swelling_terms) and not denies_swelling
    has_pus = _has_any(text, pus_terms) and not denies_pus
    has_night = _has_any(text, night_terms)
    has_persistent = _has_any(text, persistent_terms)
    has_throbbing = _has_any(text, throbbing_terms)
    has_severe = _has_any(text, severe_pain_terms)
    has_stimulus = _has_any(text, stimulus_terms)
    has_transient = _has_any(text, transient_terms)
    has_chewing = _has_any(text, chewing_terms)
    has_cavity = _has_any(text, cavity_terms)
    has_bleeding = _has_any(text, bleeding_terms)
    has_gum = _has_any(text, gum_terms)

    abscess_blocked = denies_swelling and denies_pus

    # 1) Abscess (STRICT)
    if not abscess_blocked:
        if has_pus:
            return "خراج سني", "High"
        if has_swelling and (has_severe or has_persistent):
            return "خراج سني", "High"

    # 2) Irreversible pulpitis (STRICT)
    if has_persistent and (has_night or has_throbbing or has_severe):
        return "التهاب عصب غير عكوس", "High"

    if has_night and has_severe:
        return "التهاب عصب غير عكوس", "High"

    if denies_night := _negated(text, "ألم ليلي"):
        if denies_night and has_transient and has_stimulus:
            return "التهاب عصب عكوس", "Moderate"

    # 3) Reversible pulpitis
    if has_stimulus and has_transient:
        return "التهاب عصب عكوس", "Moderate"

    # 4) Deep caries
    if (has_chewing and _has_any(text, ["سن", "ضرس"])) or (has_cavity and has_chewing):
        return "تسوس عميق", "Moderate"

    # 5) Surface caries
    if has_stimulus and _has_any(text, ["حساسية", "برودة", "بارد", "حلو"]) and not has_persistent and not has_night:
        return "تسوس سطحي", "Low"

    # 6) Gingivitis
    if has_bleeding and has_gum:
        return "التهاب لثة", "Low"

    return "غير واضح حالياً", "Low"
