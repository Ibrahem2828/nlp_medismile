# Medismile – AraBERT Symptom Understanding Model

This Space hosts an AraBERT-based NLP model for analyzing Arabic dental symptoms and providing AI-assisted preliminary diagnostic signals. It is an **Inference-only** microservice; no training runs inside the Space.

## Endpoint
`POST /analyze-symptoms`

### Input
```json
{
  "symptoms_text": "أعاني من ألم شديد في الضرس مع تورم",
  "case_id": "UUID-123",      // optional
  "request_id": "REQ-1"       // optional
}
```

### Output
- `primary_condition`
- `suspected_conditions` (label + confidence)
- `severity` (lower-case)
- `confidence_level` (low | medium | high)
- `normalized_text`
- `metadata` (tokens_length, inference_time_ms)
- `model_version`

Example response:
```json
{
  "case_id": "UUID-123",
  "primary_condition": "تسوس عميق",
  "suspected_conditions": [
    {"name": "تسوس عميق", "confidence": 0.78},
    {"name": "التهاب عصب عكوس", "confidence": 0.41}
  ],
  "severity": "high",
  "confidence_level": "medium",
  "normalized_text": "الم شديد في الضرس مع تورم",
  "model_version": "arabert_symptoms_v1",
  "metadata": { "tokens_length": 9, "inference_time_ms": 110 },
  "note": "النتائج تعتمد على الأعراض النصية فقط وليست تشخيصاً نهائياً"
}
```

## Health Check
`GET /health` → `{"status":"ok","model_loaded":true,"model_version":"arabert_symptoms_v1"}`

## Running locally
```bash
python -m venv env
./env/Scripts/activate
pip install -r requirements.txt
uvicorn src.api.main:app --host 0.0.0.0 --port 7860
```
On Hugging Face Spaces, the Dockerfile runs uvicorn on port **7860** (mandatory).

## Safety Disclaimer
⚠️ This model provides AI-assisted preliminary analysis only. It is NOT a medical diagnosis and does not replace professional dental evaluation or imaging. Final decision rests with clinicians and image-based models (Fusion Engine).
