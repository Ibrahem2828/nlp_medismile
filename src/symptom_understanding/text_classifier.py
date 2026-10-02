# src/symptom_understanding/text_classifier.py
from __future__ import annotations

from typing import Tuple

from .text_utils import has_any, prep_all

# All vocabulary goes through the same normalisation as the input text
# (see text_preprocess.preprocess_text), so the two can never drift apart.
_SWELLING = prep_all(["تورم", "منتفخ", "انتفاخ", "ورم"])
_PUS = prep_all(["قيح", "صديد", "خراج", "تقيح"])
_PAIN = prep_all(["ألم", "وجع", "يوجع", "يوجعني", "توجع"])
_NIGHT = prep_all(["بالليل", "ليلي", "ليلا", "يوقظني", "يمنعني من النوم", "لا استطيع النوم", "يزداد بالليل"])
_PERSISTENT = prep_all(["مستمر", "مستمره", "متواصل", "لا يهدأ", "لا يزول", "لا يختفي", "لا يروح",
                        "لا يتوقف", "طول اليوم", "يزداد مع الوقت", "دائم"])
_THROBBING = prep_all(["نبضي", "ينبض", "نابض", "خافق", "نبض"])
_SEVERE = prep_all(["شديد", "شديده", "شديد جدا", "لا يحتمل", "بقوه", "قوي", "فظيع", "رهيب", "مبرح"])
# Thermal/sweet triggers only: "حساسية باللثة" (gum tenderness) is NOT a tooth stimulus.
_STIMULUS = prep_all(["بارد", "ساخن", "حار", "حلو", "حلويات", "سكريات", "حلوى", "مشروبات"])
_SURFACE_STIMULUS = prep_all(["بارد", "حلو", "حلويات", "سكريات", "حلوى", "برودة"])
_TRANSIENT = prep_all(["يختفي", "يزول", "يروح", "ثواني", "لحظات", "لحظي", "لحظيا", "مؤقت", "عابر", "سريع"])
_CHEWING = prep_all(["عند المضغ", "مع المضغ", "عند العض", "مع العض", "عند الضغط", "مع الضغط",
                     "عند الاكل", "اثناء الاكل", "المضغ"])
_CAVITY = prep_all(["تسوس", "نخر", "ثقب", "حفره", "تجويف"])
_GUM = prep_all(["لثه", "اللثه", "لثتي", "لثتك"])
# Bleeding while brushing/flossing implies the gums even when the word is missing.
_GUM_CONTEXT = prep_all(["التفريش", "تفريش", "الفرشاه", "فرشاه", "الخيط", "خيط الاسنان", "السواك"])
_BLEEDING = prep_all(["نزيف", "تنزف", "ينزف", "دم"])
_TOOTH = prep_all(["سن", "ضرس"])
_FEVER = prep_all(["حمى", "حراره", "سخونه"])
_FACE_SWELLING = prep_all(["تورم الوجه", "تورم في الوجه", "انتفاخ الوجه", "الوجه منتفخ", "تورم الخد", "تورم في الخد"])


def classify_text(text: str) -> Tuple[str, str]:
    """
    Clinical rule-based classifier (Arabic). `text` must already be normalised with
    `preprocess_text`.
    Returns: (diagnosis_label, severity_level)
    Diagnoses: خراج سني، التهاب عصب غير عكوس، التهاب عصب عكوس، تسوس عميق، تسوس سطحي، التهاب لثة، أو غير واضح.
    Every symptom is checked with proper negation scope ("بدون ورم ولا قيح", "ما في ألم", ...).
    """
    text = (text or "").strip()
    if not text:
        return "غير واضح حالياً", "Low"

    pain = has_any(text, _PAIN)
    swelling = has_any(text, _SWELLING)
    pus = has_any(text, _PUS)
    night = pain and has_any(text, _NIGHT)
    persistent = pain and has_any(text, _PERSISTENT)
    throbbing = pain and has_any(text, _THROBBING)
    severe = pain and has_any(text, _SEVERE)
    stimulus = has_any(text, _STIMULUS)
    transient = has_any(text, _TRANSIENT) and not has_any(text, _PERSISTENT)
    chewing = has_any(text, _CHEWING)
    cavity = has_any(text, _CAVITY)
    bleeding = has_any(text, _BLEEDING)
    gum = has_any(text, _GUM) or has_any(text, _GUM_CONTEXT)
    fever = has_any(text, _FEVER)
    face_swelling = has_any(text, _FACE_SWELLING)

    # 1) Abscess
    if pus:
        return "خراج سني", "High"
    if face_swelling or (swelling and (pain or severe or persistent or fever)):
        return "خراج سني", "High"

    # 2) Irreversible pulpitis
    if persistent and (night or throbbing or severe):
        return "التهاب عصب غير عكوس", "High"
    if night and severe:
        return "التهاب عصب غير عكوس", "High"

    # 3) Reversible pulpitis
    if stimulus and transient:
        return "التهاب عصب عكوس", "Moderate"

    # 4) Deep caries
    if chewing and (pain or cavity or has_any(text, _TOOTH)):
        return "تسوس عميق", "Moderate"

    # 5) Gingivitis (before surface caries: gum symptoms must not be read as tooth sensitivity)
    if bleeding and gum:
        return "التهاب لثة", "Low"

    # 6) Surface caries
    if stimulus and has_any(text, _SURFACE_STIMULUS) and not persistent and not night:
        return "تسوس سطحي", "Low"

    return "غير واضح حالياً", "Low"
