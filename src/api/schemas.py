# src/api/schemas.py

from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional, List


class SymptomAnalysisRequest(BaseModel):
    text: str


class SymptomAnalysisResponse(BaseModel):
    diagnosis_label: str
    confidence_level: str
    severity_level: str
    urgency_level: str
    patient_explanation: str
    recommendations: Optional[str]
    metadata: Dict[str, Any]


class SymptomAnalyzePayload(BaseModel):
    case_id: Optional[str] = Field(default=None, description="Optional case tracking ID")
    symptoms_text: str = Field(..., min_length=4, description="Raw symptom text (Arabic)")
    request_id: Optional[str] = Field(default=None, description="Optional request correlation ID")

    @validator("symptoms_text")
    def validate_symptoms_text(cls, v: str) -> str:
        cleaned = (v or "").strip()
        if len(cleaned) < 4:
            raise ValueError("symptoms_text too short")
        if cleaned.isdigit():
            raise ValueError("symptoms_text must not be numeric only")
        return cleaned


class SuspectedCondition(BaseModel):
    name: str
    confidence: float


class SymptomAnalyzeResponse(BaseModel):
    case_id: Optional[str]
    suspected_conditions: List[SuspectedCondition]
    severity: str
    model_version: str
    primary_condition: str
    confidence_level: str
    normalized_text: str
    metadata: Dict[str, Any]
    note: str = "النتائج تعتمد على الأعراض النصية فقط وليست تشخيصاً نهائياً"
