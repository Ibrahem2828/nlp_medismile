# src/symptom_understanding/inference.py
from __future__ import annotations
import time
from .patient_explanation import generate_patient_explanation

from .text_preprocess import preprocess_text
from .text_classifier import classify_text
from src.rules.urgency import detect_red_flags, determine_urgency
from src.rules.severity import final_severity
from .arabert_similarity import AraBERTSemanticClassifier


_ARABERT = AraBERTSemanticClassifier(
    model_name="aubmindlab/bert-base-arabertv2",
    min_score=0.52,
    min_margin=0.05,
)

CRITICAL_DIAGNOSES = {"خراج سني", "التهاب عصب غير عكوس"}

SAFE_UPGRADES = {
    "تسوس سطحي": {"التهاب عصب عكوس"},
    "تسوس عميق": {"التهاب عصب عكوس", "التهاب عصب غير عكوس"},
    "التهاب عصب عكوس": {"التهاب عصب غير عكوس"},
    "التهاب لثة": set(),
    "غير واضح حالياً": {"تسوس سطحي", "تسوس عميق", "التهاب عصب عكوس", "التهاب عصب غير عكوس", "خراج سني", "التهاب لثة"},
}


def _hybrid_decision(rule_diag: str, arabert_label: str, arabert_score: float) -> str:
    if rule_diag in CRITICAL_DIAGNOSES:
        return rule_diag

    if arabert_label == "غير واضح حالياً":
        return rule_diag

    if rule_diag == "غير واضح حالياً":
        return arabert_label

    allowed = SAFE_UPGRADES.get(rule_diag, set())
    if arabert_label in allowed:
        return arabert_label

    return rule_diag


def analyze_symptoms(text: str):
    started = time.time()
    clean_text = preprocess_text(text)
    # The rules also see sentence boundaries, so a negation never leaks into the next sentence.
    rule_text = preprocess_text(text, keep_boundaries=True)

    rule_diag, rule_sev = classify_text(rule_text)

    sug = _ARABERT.suggest(clean_text)

    final_diag = _hybrid_decision(rule_diag, sug.label, sug.score)

    urgency = determine_urgency(
        diagnosis=final_diag,
        severity=rule_sev,
        text=rule_text,
    )
    # Severity follows the FINAL diagnosis; an Urgent case is always High.
    final_sev = final_severity(final_diag, urgency)
    final_sev_lower = final_sev.lower()
    red_flags = detect_red_flags(rule_text)
    patient_explanation = generate_patient_explanation(
        diagnosis=final_diag,
        severity=final_sev,
        urgency=urgency,
    )

    # Confidence: اعتمد على AraBERT إن وجد، مع قصّ القيم العالية لحماية الدمج.
    raw_confidence = float(sug.score) if sug.label != "غير واضح حالياً" else 0.5
    confidence = max(0.0, min(0.9, raw_confidence))  # clipping

    confidence_level = "low"
    if confidence >= 0.75:
        confidence_level = "high"
    elif confidence >= 0.5:
        confidence_level = "medium"

    suspected_conditions = [
        {
            "label": final_diag,
            "confidence": round(confidence, 3),
        }
    ]
    if sug.second_label:
        suspected_conditions.append(
            {
                "label": sug.second_label,
                "confidence": round(float(sug.second_score or 0.0), 3),
            }
        )

    inference_time_ms = int((time.time() - started) * 1000)
    tokens_length = len(clean_text.split())

    return {
        "clean_text": clean_text,
        "rule_diagnosis": rule_diag,
        "severity": final_sev,
        "severity_level": final_sev_lower,
        "arabert_diagnosis": sug.label,
        "arabert_score": round(float(sug.score), 3),
        "arabert_second": sug.second_label,
        "arabert_second_score": round(float(sug.second_score), 3) if sug.second_score is not None else None,
        "diagnosis": final_diag,
        "urgency": urgency,
        "red_flags": red_flags,
        "confidence": round(confidence, 3),
        "confidence_level": confidence_level,
        "model_version": "arabert_symptoms_v1",
        "normalized_text": clean_text,
        "patient_explanation": patient_explanation,
        "suspected_conditions": suspected_conditions,
        "metadata": {
            "tokens_length": tokens_length,
            "inference_time_ms": inference_time_ms,
        },
    }
