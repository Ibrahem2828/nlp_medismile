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

## Rule engine & tests
The rule layer (`src/symptom_understanding/text_preprocess.py`, `text_utils.py`, `text_classifier.py`, `src/rules/`) is covered by offline unit tests (no model download):
```bash
python -m unittest discover -s tests -t .
```
Behaviour guaranteed by the tests:
- rules and input share one normalised form (hamza, `ة`, dialect words such as `وجع`/`يوجعني` → `الم`), matched by whole words only;
- negation has a proper scope (`بدون ورم ولا قيح`, `ما في تورم بس ألم شديد`, `ولا يوجد ألم أو تورم`) and sentence boundaries stop it;
- severity is derived from the final diagnosis and is always `high` when urgency is `Urgent`;
- red flags (fever with swelling, trismus, dysphagia, breathing difficulty, trauma, uncontrolled bleeding, spreading facial swelling) force `Urgent` and are returned in `red_flags`;
- non-Arabic input is rejected with HTTP 422.

## Evaluation
Accuracy is measured against a clinician-labelled, leakage-free gold set with `python -m evaluation.run_eval`
(urgent recall, per-class F1, confusion matrix, calibration, quality gate). See [`evaluation/README.md`](evaluation/README.md)
and the annotation guidelines in [`evaluation/guidelines_ar.md`](evaluation/guidelines_ar.md).
