# دليل تقني لمشروع Medismile AraBERT

## المكونات الرئيسة
- واجهة API: `src/api/main.py` تخدم FastAPI مع نقاط `/api/analyze-text`, `/analyze/symptoms`, `/analyze-symptoms`, و`/health`، وتقدم الملفات الثابتة من `ui/`.
- مخططات التحقق: `src/api/schemas.py` تضبط بنية الطلب/الاستجابة، وتتحقق من طول النص وعدم كونه أرقاماً فقط.
- طبقة فهم الأعراض: داخل `src/symptom_understanding/` وتشمل التنقية اللغوية، التصنيف القاعدي، تطابق AraBERT الدلالي، توليد شرح المريض، وتقدير الثقة.
- قواعد الإلحاح: `src/rules/urgency.py` لتحديد Urgent/Non-Urgent بناءً على الأعراض والشدّة والإنكار.
- واجهة الويب: مجلد `ui/` (HTML/CSS/JS) مع إدخال نصي أو إملاء صوتي عبر متصفح المستخدم.
- النموذج المخزن: `models/arabert_finetune/` يحوي ملفات نموذج مضبوطة، لكنه غير مستعمل في مسار الاستدلال الحالي الذي يعتمد `aubmindlab/bert-base-arabertv2` مع بروتوتايب نصي.

## مسار المعالجة (analyze_symptoms)
1) التحقق الأولي في API: رفض النصوص الفارغة، الأقصر من 4 أحرف، أو الأرقام فقط (`TextRequest.validate`, `SymptomAnalyzePayload.validator`).
2) التنقية اللغوية `preprocess_text`: إزالة التشكيل والـ`tatweel`، تطبيع بعض الحروف العربية، حذف الرموز غير العربية، دمج المسافات، ثم استبدالات خفيفة للهجات/الأخطاء الإملائية.
3) التصنيف القاعدي `classify_text`: بحث عن مفردات تورّم/قيح/ألم ليلي/ألم نابض/تحفيز بالبرد والحرارة/نزيف لثة... مع دعم إنكار الأعراض البسيط. يعيد تشخيصاً ابتدائياً (مثل abscess أو pulpitis أو caries أو gingivitis) مع مستوى شدّة High/Moderate/Low.
4) مطابقة AraBERT الدلالية `AraBERTSemanticClassifier.suggest`: يحوّل النص إلى تمثيل (mean-pooling) عبر `aubmindlab/bert-base-arabertv2` ويقارن كوسينياً مع متجهات بروتوتايب مخزّنة لكل تشخيص. يطبّق عتبات `min_score=0.52` و`min_margin=0.05` لإرجاع "تشخيص غير مؤكد" عند التشابه الضعيف أو الفارق الضئيل.
5) دمج rule + AraBERT `_hybrid_decision`: يعطي أولوية للتشاخيص الحرجة (مثل الخراج)، يسمح بترقيات آمنة محددة مسبقاً، وي otherwise يبقي نتيجة القاعدة إذا كانت AraBERT غير مؤكدة أو تقترح حالة خارج القائمة البيضاء.
6) تقدير الإلحاح `determine_urgency`: يختبر تورّم/قيح/ألم ليلي/ألم غير مهدأ/أعراض جهازية مع مراعاة الإنكار اللغوي لتصنيف "Urgent" أو "Non-Urgent".
7) توليد شرح المريض `generate_patient_explanation`: نص عربي مبسّط يختلف حسب الشدّة/الإلحاح، لتوجيه المستخدم دون مصطلحات طبية معقدة.
8) الثقة والمستويات: الثقة الخام مشتقة من درجة AraBERT (0.0-0.9 بعد القص)، ثم مستوى ثقة low/medium/high بناءً على العتبات 0.5 و0.75.
9) النتائج الإضافية: قائمة `suspected_conditions` (تشخيص أساسي + ثاني إن وجد)، نص منقّح `normalized_text`, نسخة من التشخيص القاعدي ونتيجة AraBERT، زمن الاستدلال بالملّي ثانية، وطول النص.

## نقاط النهاية (FastAPI)
- `POST /api/analyze-text`: واجهة تفاعلية للواجهة الأمامية؛ تعيد كل الحقول التقنية + شرح المريض.
- `POST /analyze/symptoms`: عقدة مستقرة لـ Django تعيد هيكل `SymptomAnalysisResponse` المنظّم (تشخيص/شدّة/إلحاح/شرح/metadata).
- `POST /analyze-symptoms`: عقدة Model #2 → Model #3 (Fusion) تعيد `suspected_conditions` مع الثقة، التشخيص الأساسي، الشدّة، مستوى الثقة، النص المنقّح، والـmetadata.
- `GET /health`: فحص خفيف يعيد الحالة وإصدار النموذج.

## واجهة المستخدم `ui/`
- `index.html`: صفحة RTL بالعربية مع شعار وأزرار بدء/إيقاف التسجيل الصوتي وزر التحليل.
- `app.js`: يستخدم Web Speech API (SpeechRecognition) للنسخ الصوتي إلى العربية (ar-SA) وإلحاقه في حقل النص. يستدعي `/api/analyze-text` ويعرض شرح المريض، الشدّة، الإلحاح، والتفاصيل التقنية (تشخيص القاعدة/AraBERT والدرجات).
- `style.css`: تصميم عصري مع تدرجات وأنيمايشن دخول وطوَاف بسيط للشعار، وتحسينات استجابة للشاشات الصغيرة.

## النمذجة والضبط
- بروتوتايب AraBERT: عبارات ممثلة لكل فئة تشخيصية محفوظة داخل `AraBERTSemanticClassifier.prototypes`، يُبنى منها فهرس embeddings عند التهيئة.
- الضبط الدلالي: لا يوجد fine-tuning في كود الاستدلال الحالي؛ يعتمد على التشابه مع بروتوتايب. مجلد `models/arabert_finetune/` يمكن توظيفه مستقبلاً لكن غير مستدعى في `inference.py`.
- القص والهوامش: أي نتيجة أقل من `min_score` أو بفارق أقل من `min_margin` تُعامل كـ "تشخيص غير مؤكد" وتُرجَع النتيجة القاعدية.

## قواعد الإلحاح
- تورّم/قيح أو أعراض جهازية مع تورّم/قيح ⇒ Urgent.
- شدّة High مع ألم ليلي وألم غير مهدأ ⇒ Urgent حتى مع إنكار التورّم/القيح.
- إنكار صريح للتورّم/القيح مع غياب المحفزات الخطرة ⇒ Non-Urgent.

## التشغيل والنشر
- المتطلبات: `requirements.txt` يشمل torch CPU، transformers، camel-tools، FastAPI/Uvicorn، وwhisper مع ffmpeg.
- محلياً: `python -m venv env && ./env/Scripts/activate && pip install -r requirements.txt && uvicorn src.api.main:app --host 0.0.0.0 --port 7860`.
- Docker: `Dockerfile` يبني على `python:3.10-slim`، يثبت ffmpeg، ينسخ المشروع، ويشغّل Uvicorn على 7860 كمستخدم غير جذري.
- Render: `render.yaml` يحدد خدمة ويب بالخطة المجانية وتشغيل Uvicorn على المنفذ 10000.

## ملاحظات وحدود
- `speech_to_text.py` غلاف جاهز لـ Whisper مع تخزين نتائج اختياري، لكن الاستدعاء الحقيقي معلّق (`self._model` معلق تعليقاً) ولا يُستخدم في الـAPI/الواجهة.
- API لا تتحقق من حجم الحمل أو معدل الطلب؛ يُنصح بإضافة معدل/مصادقة عند النشر العام.
- الترميز في بعض النصوص (README/MODEL2_ARABERT.md) يحوي محارف غير واضحة؛ يفضّل ضبط UTF-8 وإعادة الصياغة عند الحاجة.
- تشخيصات وألفاظ Arabic hard-coded؛ لإضافة فئات جديدة عدّل `prototypes`, `classify_text`, و`SAFE_UPGRADES`/`CRITICAL_DIAGNOSES`.

## خرائط الملفات السريعة
- API وواجهات: `src/api/main.py`, `src/api/schemas.py`, `ui/index.html`, `ui/app.js`, `ui/style.css`.
- المنطق الطبي/اللغوي: `src/symptom_understanding/text_preprocess.py`, `text_classifier.py`, `arabert_similarity.py`, `inference.py`, `patient_explanation.py`, وقواعد الإلحاح `src/rules/urgency.py`.
- النماذج والتهيئة: `models/arabert_finetune/` (غير مستخدمة حالياً)، `requirements.txt`, `Dockerfile`, `render.yaml`, `runtime.txt`.
