# src/api/main.py

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
from fastapi import status

from src.symptom_understanding.inference import analyze_symptoms

from src.api.schemas import (
    SymptomAnalysisRequest,
    SymptomAnalysisResponse,
    SymptomAnalyzePayload,
    SymptomAnalyzeResponse,
    SuspectedCondition,
)

app = FastAPI(
    title="Medismile AI Engine",
    description="AI-powered dental symptom triage service (AraBERT microservice)",
    version="1.0.0",
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
UI_DIR = BASE_DIR / "ui"

app.mount("/static", StaticFiles(directory=UI_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(UI_DIR / "index.html")


class TextRequest(BaseModel):
    text: str


@app.post("/api/analyze-text")
def analyze_text(req: TextRequest):
    if not req.text or len(req.text.strip()) < 4:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Text too short")
    if req.text.strip().isdigit():
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Text must not be numeric only")

    result = analyze_symptoms(req.text)

    return {
        "text": result["clean_text"],
        "diagnosis": result["diagnosis"],
        "severity": result["severity_level"],
        "urgency": result["urgency"],
        "rule_diagnosis": result["rule_diagnosis"],
        "arabert_diagnosis": result["arabert_diagnosis"],
        "arabert_score": result["arabert_score"],
        "arabert_second": result["arabert_second"],
        "arabert_second_score": result["arabert_second_score"],
        "patient_explanation": result["patient_explanation"],
        "model_version": result["model_version"],
        "confidence": result["confidence"],
        "confidence_level": result["confidence_level"],
        "metadata": result["metadata"],
    }


@app.post(
    "/analyze/symptoms",
    response_model=SymptomAnalysisResponse,
)
def analyze_symptoms_system(payload: SymptomAnalysisRequest):
    """
    Stable endpoint for Django.
    Uses the SAME AI engine, but returns a clean contract.
    """

    try:
        result = analyze_symptoms(payload.text)

        return SymptomAnalysisResponse(
            diagnosis_label=result["diagnosis"],
            confidence_level="medium",
            severity_level=result["severity"],
            urgency_level=result["urgency"],
            patient_explanation=result["patient_explanation"],
            recommendations=None,
            metadata={
                "rule_diagnosis": result["rule_diagnosis"],
                "arabert": {
                    "label": result["arabert_diagnosis"],
                    "score": result["arabert_score"],
                    "second": result["arabert_second"],
                    "second_score": result["arabert_second_score"],
                },
                "model_version": result["model_version"],
                "confidence": result["confidence"],
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Fusion-facing contract (Model #2 → Model #3)
# ============================================================


@app.post(
    "/analyze-symptoms",
    response_model=SymptomAnalyzeResponse,
)
def analyze_symptoms_contract(payload: SymptomAnalyzePayload):
    """
    Contract for backend → AraBERT (Model #2) → Fusion (Model #3).
    Accepts case_id and symptoms_text, returns suspected_conditions list with confidence,
    severity, and model_version. No final decision or image fusion happens here.
    """

    result = analyze_symptoms(payload.symptoms_text)

    return SymptomAnalyzeResponse(
        case_id=payload.case_id,
        suspected_conditions=[
            SuspectedCondition(
                name=item["label"],
                confidence=item["confidence"],
            )
            for item in result["suspected_conditions"]
        ],
        severity=result["severity_level"],
        model_version=result["model_version"],
        primary_condition=result["diagnosis"],
        confidence_level=result["confidence_level"],
        normalized_text=result["normalized_text"],
        metadata=result["metadata"],
        note="النتائج تعتمد على الأعراض النصية فقط وليست تشخيصاً نهائياً",
    )


# ============================================================
# Health Check
# ============================================================


@app.get("/health")
def health_check():
    """
    Lightweight health check for Spaces/Backend orchestration.
    Does not expose patient data or raw text.
    """
    return {
        "status": "ok",
        "model_loaded": True,
        "model_version": "arabert_symptoms_v1",
    }
