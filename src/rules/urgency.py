# src/rules/urgency.py
from __future__ import annotations

from typing import List

from src.symptom_understanding.text_utils import has_any, prep_all

CRITICAL_DIAGNOSES = {"خراج سني", "التهاب عصب غير عكوس"}

_SWELLING = prep_all(["تورم", "منتفخ", "انتفاخ", "ورم"])
_PUS = prep_all(["قيح", "صديد", "خراج", "تقيح"])
_PAIN = prep_all(["ألم", "وجع", "يوجع", "يوجعني"])
_UNRELIEVED = prep_all(["مستمر", "لا يهدأ", "لا يزول", "لا يختفي", "لا يتوقف", "طول اليوم",
                        "يزداد مع الوقت", "بقوه", "شديد", "لا يحتمل"])
_NIGHT = prep_all(["بالليل", "ليلي", "ليلا", "يوقظني", "يمنعني من النوم"])
_FEVER = prep_all(["حمى", "حراره", "سخونه", "ارتفاع حراره"])
_FACE_SWELLING = prep_all(["تورم الوجه", "تورم في الوجه", "انتفاخ الوجه", "تورم العين", "تورم الرقبه",
                           "تورم تحت الفك", "ورم الوجه"])

# Red flags escalate to Urgent whatever the model says.
_RED_FLAGS = {
    "trismus": prep_all(["صعوبة فتح الفم", "صعوبة في فتح الفم", "لا استطيع فتح فمي", "تشنج الفك"]),
    "dysphagia": prep_all(["صعوبة البلع", "صعوبة في البلع", "لا استطيع البلع"]),
    "breathing_difficulty": prep_all(["صعوبة التنفس", "صعوبة في التنفس", "ضيق التنفس", "ضيق نفس"]),
    "uncontrolled_bleeding": prep_all(["نزيف لا يتوقف", "نزيف غزير", "نزيف شديد", "لا يتوقف النزيف"]),
    "trauma": prep_all(["ضربة", "حادث", "كسر", "سقطت", "اصطدام", "رضه"]),
    "spreading_facial_swelling": _FACE_SWELLING,
}


def detect_red_flags(text: str) -> List[str]:
    """Return the red-flag codes present in the (normalised) text."""
    text = (text or "").lower()
    flags = [name for name, terms in _RED_FLAGS.items() if has_any(text, terms)]
    swelling = has_any(text, _SWELLING)
    pus = has_any(text, _PUS)
    if has_any(text, _FEVER) and (swelling or pus):
        flags.append("fever_with_swelling_or_pus")
    return flags


def determine_urgency(diagnosis: str, severity: str, text: str) -> str:
    text = (text or "").lower()

    if diagnosis in CRITICAL_DIAGNOSES:
        return "Urgent"
    if detect_red_flags(text):
        return "Urgent"

    swelling = has_any(text, _SWELLING)
    pus = has_any(text, _PUS)
    pain = has_any(text, _PAIN)
    unrelieved = pain and has_any(text, _UNRELIEVED)
    night = pain and has_any(text, _NIGHT)

    if pus:
        return "Urgent"
    if swelling and (pain or unrelieved):
        return "Urgent"
    if severity == "High" and night and unrelieved:
        return "Urgent"

    return "Non-Urgent"
