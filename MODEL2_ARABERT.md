# خطة تطوير النموذج الثاني: AraBERT لتحليل الأعراض النصية

## الهدف والمخرجات
- الهدف: فهم وصف المريض للأعراض (فصحى ولهجات)، استخراج مؤشرات مرضية، وتقديم تشخيص نصي أولي مع درجة شدة وإلحاح.
- المدخلات: نص عربي حر (أو نص ناتج من تحويل صوتي لاحقاً).
- المخرجات (عبر `/api/analyze-text`, `/analyze/symptoms`, `/analyze-symptoms`):
  ```json
  {
    "diagnosis": "<التشخيص النهائي بعد الدمج>",
    "severity": "High|Moderate|Low",
    "urgency": "Urgent|Non-Urgent",
    "rule_diagnosis": "<تشخيص القواعد>",
    "arabert_diagnosis": "<تشخيص AraBERT>",
    "arabert_score": 0.x,
    "arabert_second": "<أفضل بديل>",
    "arabert_second_score": 0.x,
    "patient_explanation": "نص مبسط للمريض",
    "model_version": "arabert_symptoms_v1"
  }
  ```
  ولعقدة الاندماج مع النموذج الثالث (`/analyze-symptoms`):
  ```json
  {
    "case_id": "UUID",
    "suspected_conditions": [{"name": "caries_deep", "confidence": 0.84}],
    "severity": "high",
    "model_version": "arabert_symptoms_v1",
    "note": "النتائج تعتمد على الأعراض النصية فقط وليست تشخيصاً نهائياً"
  }
  ```

## خط الأنابيب الحالي (داخل الشيفرة)
1. تنظيف النص: `src/symptom_understanding/text_preprocess.py` يزيل التشكيل والتمطيط، يوحّد أشكال الحروف، يحذف الرموز غير العربية، ويطبّق قاموس استبدالات خفيف للهجات.
2. تصنيف قاعدي Rule-based: `src/symptom_understanding/text_classifier.py` يبحث عن مجموعات أعراض (تورم/قيح/ألم ليلي/ألم مستمر/ألم مع المحفز/نزيف لثوي...) مع نفي بسيط. يخرج واحدة من ست فئات أساسية (خراج، التهابات عصب، تسوس عميق/سطحي، التهاب لثة، أو غير واضح) ويعيد شدة High/Moderate/Low.
3. طبقة AraBERT الدلالية: `src/symptom_understanding/arabert_similarity.py` تولّد متجه Mean Pooling من `aubmindlab/bert-base-arabertv2` وتقيس تشابه كوني مع بروتوتايب لكل فئة. العتبات: `min_score=0.52`, `min_margin=0.05`.
4. دمج القرار: `_hybrid_decision` في `src/symptom_understanding/inference.py` يثبت التشخيص إذا كان قاعدياً حرجاً، يسمح بـ AraBERT عند الغموض أو في “ترقيات آمنة”، وإلا يبقي القاعدة. الشدة حالياً من القواعد. confidence يُشتق من درجة AraBERT إن كانت واضحة، وإلا 0.5.
5. تقدير الإلحاح: `src/rules/urgency.py` يصنف عاجل/غير عاجل حسب التشخيص والشدة وإشارات التورم/القيح/الألم الليلي/الأعراض الجهازية مع احترام النفي.
6. شرح للمريض: `src/symptom_understanding/patient_explanation.py` ينتج نصوصاً مبسطة حسب الشدة/الإلحاح.
7. (اختياري) صوت إلى نص: `src/symptom_understanding/speech_to_text.py` غلاف لـ Whisper مع تحقق من الملفات وتخزين مؤقت اختياري.
8. واجهات: `src/api/main.py` يعرّض:
   - `/api/analyze-text`: للواجهة التجريبية.
   - `/analyze/symptoms`: عقدة Django مستقرة، تعيد `SymptomAnalysisResponse` مع metadata.
   - `/analyze-symptoms`: عقدة الربط مع النموذج الثالث (Fusion) تعيد `suspected_conditions`, `severity`, `model_version`, وتنبيه طبي صريح.

## الربط المعماري (Model #2 ↔ Backend ↔ Model #3)
- الواجهة الأمامية ترسل النص إلى الباك-إند فقط.
- الباك-إند يتحقق من المستخدم/الحالة، يرسل النص إلى AraBERT (Model #2) عبر `/analyze-symptoms`، يستلم النتائج كنقاط تشخيصية داعمة، ثم يمررها للنموذج الثالث (Fusion Engine) لدمجها مع نتائج الصور.
- أولوية القرار للصور؛ النص عامل داعم. يجب ألا يوصف أي مخرج من Model #2 بأنه “تشخيص نهائي”.

## اعتبارات السلامة الطبية
- تضمين تنبيه دائم: “النتائج تعتمد على الأعراض النصية فقط وليست تشخيصاً نهائياً”.
- عدم استخدام مصطلح “تشخيص نهائي”.
- قصّ/تخفيض الثقة عند الدمج (Confidence Clipping) إن لزم.

## خطة تطوير متدرجة
- **المرحلة 1: تنظيف النص**: استكمال استبدالات اللهجات، اختبارات وحدات، وضمان UTF-8.
- **المرحلة 2: إعداد البيانات**: Dataset متوازن (الأعمدة text, normalized_text, label, severity) يغطي: gingivitis, caries_superficial, caries_deep, abscess, periodontitis, tooth_sensitivity, normal_condition.
- **المرحلة 3: Fine-tuning AraBERT**: تصنيف متعدد الفئات، Cross Entropy، وزن فئات، مراقبة F1 خاصة للفئات الحرجة (abscess/periodontitis).
- **المرحلة 4: تحسين الفهم الطبي**: keyword boosting، وضبط الثقة rule-based، وتوسيع بروتوتايب AraBERT بإشراف طبي.

## أعمال هندسية
- ملء بيانات التدريب وتشغيل `train.py` لحفظ نموذج Fine-tuned في `models/arabert_finetune`.
- إضافة تقييمات و`eval_metrics.json` مع F1/Accuracy.
- تحديث توثيق Hugging Face عند رفع النموذج (README يوضح الدور المحدود وتنبيه السلامة).

## ملاحظات مخاطر
- أي نص عربي يجب أن يكون UTF-8 لتفادي فساد الحروف.
- المطابقات الحرفية تعتمد على نصوص عربية واضحة؛ أي ترميز تالف سيعطل القواعد.
- عند الدمج مع الصور، يجب ألا تُستخدم ثقة AraBERT وحدها لاتخاذ قرار نهائي.
