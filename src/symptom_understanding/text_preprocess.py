# src/symptom_understanding/text_preprocess.py
from __future__ import annotations

import re
from typing import Dict

# Arabic diacritics (tashkeel)
_ARABIC_DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652]")
_TATWEEL = "\u0640"

# Keep Arabic letters/numbers/spaces
_NON_ARABIC_PUNCT = re.compile(r"[^\w\s\u0600-\u06FF]")
_WS = re.compile(r"\s+")

# Common light replacements for dialect/typos
_REPLACEMENTS: Dict[str, str] = {
    "يوجعني": "الم",
    "يوجع": "الم",
    "يوجعك": "الم",
    "وجع": "الم",
    "واجع": "الم",
    "موجع": "الم",
    "حساس": "محفز",
    "حساسية": "محفز",
    "بارد": "بارد",
    "ساخن": "ساخن",
    "حار": "ساخن",
    "خراج": "خراج",
    "تقيح": "قيح",
    "صديد": "قيح",
    "قيح": "قيح",
    "نزيف": "نزيف",
    "تنزف": "نزيف",
    "دم": "نزيف",
    "تورم": "تورم",
    "ورم": "تورم",
    "منتفخ": "تورم",
    "انتفاخ": "تورم",
    "ينتفخ": "تورم",
    "لثة": "لثة",
    "اللثة": "لثة",
    "ضرس": "سن",
    "ضروس": "سن",
    "سناني": "سن",
    "اسناني": "سن",
    "سن": "سن",
}


def _normalize_letters(text: str) -> str:
    """Normalize common Arabic letter variants for more stable matching."""
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ى", "ي").replace("ئ", "ي").replace("ؤ", "و")
    return text


def preprocess_text(text: str) -> str:
    """
    Arabic-safe normalization for dental symptom descriptions.
    Steps:
    - strip, remove diacritics, remove tatweel
    - normalize Arabic letter variants
    - remove punctuation/symbols (keep Arabic letters/numbers)
    - collapse whitespace and lowercase
    - apply light replacements for dialect/typos
    """
    text = (text or "").strip()
    if not text:
        return ""

    text = re.sub(_ARABIC_DIACRITICS, "", text)
    text = text.replace(_TATWEEL, "")
    text = _normalize_letters(text)
    text = re.sub(_NON_ARABIC_PUNCT, " ", text)
    text = re.sub(_WS, " ", text).strip().lower()

    for k, v in _REPLACEMENTS.items():
        text = text.replace(k, v)

    text = re.sub(_WS, " ", text).strip()
    return text
