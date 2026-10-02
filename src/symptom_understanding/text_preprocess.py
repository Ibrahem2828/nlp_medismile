# src/symptom_understanding/text_preprocess.py
from __future__ import annotations

import re
from typing import Dict, List

# Arabic diacritics (tashkeel)
_ARABIC_DIACRITICS = re.compile(r"[\u0617-\u061A\u064B-\u0652]")
_TATWEEL = "\u0640"

# Keep Arabic letters/numbers/spaces
_NON_ARABIC_PUNCT = re.compile(r"[^\w\s\u0600-\u06FF]")
# Same, but also keep the "|" clause-boundary marker.
_NON_ARABIC_PUNCT_KEEP_BOUNDARY = re.compile(r"[^\w\s\u0600-\u06FF|]")
# Arabic-script punctuation (U+060C comma, U+061B semicolon, U+061F question mark, U+066A-066D)
# lives inside the Arabic block, so strip it explicitly.
_ARABIC_PUNCT = re.compile(r"[\u060C\u061B\u061F\u066A-\u066D]")
_WS = re.compile(r"\s+")
# Sentence/clause punctuation, turned into a "|" marker when boundaries are requested.
_BOUNDARY = re.compile(r"[.!?\u060C\u061B\u061F;:,\n\r]+")
BOUNDARY_TOKEN = "|"

# Light dialect/typo canonicalisation. Keys are matched as WHOLE TOKENS (optionally
# carrying a clitic prefix such as و/ب/ال), never as substrings, so a word like
# "قدم" is not rewritten because it contains "دم". Keys are written in normalised
# letter form (see _normalize_letters).
_REPLACEMENTS: Dict[str, str] = {
    "يوجعني": "الم",
    "بيوجعني": "الم",
    "بتوجعني": "الم",
    "وجعني": "الم",
    "يوجعك": "الم",
    "بيوجع": "الم",
    "بتوجع": "الم",
    "يوجع": "الم",
    "وجع": "الم",
    "واجع": "الم",
    "موجع": "الم",
    "حساس": "محفز",
    "حساسيه": "محفز",
    "حار": "ساخن",
    "تقيح": "قيح",
    "صديد": "قيح",
    "تنزف": "نزيف",
    "ينزف": "نزيف",
    "دم": "نزيف",
    "ورم": "تورم",
    "منتفخ": "تورم",
    "انتفاخ": "تورم",
    "ينتفخ": "تورم",
    "ضرس": "سن",
    "ضروس": "سن",
    "سناني": "سن",
    "اسناني": "سن",
}

# Clitic prefixes that may precede a token (longest first).
_PREFIXES: List[str] = ["وال", "بال", "فال", "كال", "لل", "ال", "و", "ف", "ب", "ل", "ك"]


def _normalize_letters(text: str) -> str:
    """Normalize common Arabic letter variants for more stable matching."""
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ى", "ي").replace("ئ", "ي").replace("ؤ", "و")
    text = text.replace("ة", "ه")
    return text


def _canonical_token(token: str) -> str:
    repl = _REPLACEMENTS.get(token)
    if repl is not None:
        return repl
    for prefix in _PREFIXES:
        if token.startswith(prefix):
            repl = _REPLACEMENTS.get(token[len(prefix):])
            if repl is not None:
                return prefix + repl
    return token


def preprocess_text(text: str, keep_boundaries: bool = False) -> str:
    """
    Arabic-safe normalization for dental symptom descriptions.
    Steps:
    - strip, remove diacritics, remove tatweel
    - normalize Arabic letter variants (أ/إ/آ→ا, ى→ي, ة→ه ...)
    - remove punctuation/symbols (keep Arabic letters/numbers)
    - collapse whitespace and lowercase
    - canonicalise dialect/typo tokens (whole-token matching only)

    With keep_boundaries=True sentence punctuation becomes a "|" token. The rule engine
    uses that so a negation ("بدون تورم.") never leaks into the next sentence; the text
    sent to AraBERT / returned to clients keeps the default (no markers).

    The classifier and urgency rules pass their own vocabulary through this same
    function, so text and rules always live in one normalised space.
    """
    text = (text or "").strip()
    if not text:
        return ""

    text = re.sub(_ARABIC_DIACRITICS, "", text)
    if keep_boundaries:
        text = re.sub(_BOUNDARY, f" {BOUNDARY_TOKEN} ", text)
    text = text.replace(_TATWEEL, "")
    text = _normalize_letters(text)
    text = re.sub(_ARABIC_PUNCT, " ", text)
    text = re.sub(_NON_ARABIC_PUNCT_KEEP_BOUNDARY if keep_boundaries else _NON_ARABIC_PUNCT, " ", text)
    text = re.sub(_WS, " ", text).strip().lower()

    tokens = [_canonical_token(tok) for tok in text.split(" ")]
    return re.sub(_WS, " ", " ".join(tokens)).strip()
