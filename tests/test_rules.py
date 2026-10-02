"""Rule-engine regression tests (no model download needed).

Run from the repository root:  python -m unittest discover -s tests -t .
"""
import unittest

from src.rules.severity import final_severity
from src.rules.urgency import detect_red_flags, determine_urgency
from src.symptom_understanding.text_classifier import classify_text
from src.symptom_understanding.text_preprocess import preprocess_text
from src.symptom_understanding.text_utils import arabic_letter_ratio, has_any

ABSCESS = "خراج سني"
IRREV = "التهاب عصب غير عكوس"
REV = "التهاب عصب عكوس"
DEEP = "تسوس عميق"
SURF = "تسوس سطحي"
GINGI = "التهاب لثة"
UNCLEAR = "غير واضح حالياً"


def run(text):
    clean = preprocess_text(text, keep_boundaries=True)
    diag, sev = classify_text(clean)
    urgency = determine_urgency(diag, sev, clean)
    return diag, urgency, final_severity(diag, urgency), clean


# (text, expected diagnosis, expected urgency)
CASES = [
    # --- previously broken by the hamza / وجع normalisation mismatch -------------
    ("ألم شديد مستمر يوقظني من النوم", IRREV, "Urgent"),
    ("عندي تورم وألم", ABSCESS, "Urgent"),
    ("ألم مستمر ونبضي", IRREV, "Urgent"),
    ("وجع شديد مستمر بالليل", IRREV, "Urgent"),
    ("يوجعني ضرسي وجع شديد مستمر لا يهدأ", IRREV, "Urgent"),
    ("ألم نابض ومستمر طول اليوم", IRREV, "Urgent"),
    ("الألم بالليل شديد ولا يحتمل", IRREV, "Urgent"),
    ("ألم لا يحتمل يمنعني من النوم", IRREV, "Urgent"),
    ("ألم مستمر وينبض في الضرس", IRREV, "Urgent"),
    ("ألم شديد ومستمر في الضرس السفلي", IRREV, "Urgent"),
    # --- abscess ---------------------------------------------------------------
    ("في قيح حول السن", ABSCESS, "Urgent"),
    ("تورم في الخد وألم شديد", ABSCESS, "Urgent"),
    ("عندي خراج في اللثة", ABSCESS, "Urgent"),
    ("يخرج صديد من اللثة", ABSCESS, "Urgent"),
    ("ورم في اللثة مع وجع", ABSCESS, "Urgent"),
    ("الوجه منتفخ وفيه ألم", ABSCESS, "Urgent"),
    ("تقيح حول الضرس", ABSCESS, "Urgent"),
    # --- negation (previously produced a false abscess) ------------------------
    ("بدون ورم ولا قيح وعندي ألم عند المضغ", DEEP, "Non-Urgent"),
    ("ما في تورم ولا قيح بس ألم عند الضغط على السن", DEEP, "Non-Urgent"),
    ("لا يوجد تورم ولا صديد والألم عند المضغ", DEEP, "Non-Urgent"),
    ("ما في ورم ولا قيح والسن فيه حفرة وألم عند الأكل", DEEP, "Non-Urgent"),
    ("بدون ألم تلقائي أو تورم. ألم لحظي مع البارد ثم يزول", REV, "Non-Urgent"),
    ("ألم قوي في الضرس منذ يومين. لا توجد تورم أو قيح", UNCLEAR, "Non-Urgent"),
    ("عم حس بوجع قوي بالضرس. مو ورم ولا قيح", UNCLEAR, "Non-Urgent"),
    ("لا يوجد ورم. عندي قيح حول السن", ABSCESS, "Urgent"),
    ("ما في تورم. تورم ظهر اليوم مع ألم", ABSCESS, "Urgent"),
    ("أريد فحصا دوريا. ولا يوجد ألم أو تورم. آخر فحص منذ ستة أشهر", UNCLEAR, "Non-Urgent"),
    ("أريد تقييما دوريا. ولا يوجد ألم عند المضغ", UNCLEAR, "Non-Urgent"),
    ("بدي متابعة. وما في حساسية مع البارد أو الحار", UNCLEAR, "Non-Urgent"),
    ("بدون تورم", UNCLEAR, "Non-Urgent"),
    ("لا يوجد قيح", UNCLEAR, "Non-Urgent"),
    ("لا يوجد ألم", UNCLEAR, "Non-Urgent"),
    ("ما في وجع ولا تورم", UNCLEAR, "Non-Urgent"),
    ("ما في تورم بس ألم شديد", UNCLEAR, "Non-Urgent"),
    ("لا يوجد ألم ليلي", UNCLEAR, "Non-Urgent"),
    ("ما عندي ألم مستمر", UNCLEAR, "Non-Urgent"),
    ("بدون ألم شديد", UNCLEAR, "Non-Urgent"),
    # --- reversible pulpitis ---------------------------------------------------
    ("ألم مع البارد يختفي بسرعة", REV, "Non-Urgent"),
    ("حساسية للمشروبات الساخنة تزول بعد ثواني", REV, "Non-Urgent"),
    ("الضرس يوجعني مع الحار ويروح بعد لحظات", REV, "Non-Urgent"),
    ("لما أشرب ماء بارد يوجع ثواني ويروح", REV, "Non-Urgent"),
    ("ألم مؤقت مع الحلو", REV, "Non-Urgent"),
    # --- deep caries -----------------------------------------------------------
    ("ألم عند المضغ في السن", DEEP, "Non-Urgent"),
    ("يوجع عند العض على الضرس", DEEP, "Non-Urgent"),
    ("في حفرة كبيرة في السن وألم عند الأكل", DEEP, "Non-Urgent"),
    ("ألم عند الضغط على السن", DEEP, "Non-Urgent"),
    ("ألم أثناء الأكل في ضرس مسوس", DEEP, "Non-Urgent"),
    ("تسوس عميق وألم مع المضغ", DEEP, "Non-Urgent"),
    # --- surface caries --------------------------------------------------------
    ("حساسية خفيفة من الحلو", SURF, "Non-Urgent"),
    ("بقعة بنية وحساسية من البارد", SURF, "Non-Urgent"),
    ("أحس بحساسية عند الحلويات", SURF, "Non-Urgent"),
    # --- gingivitis ------------------------------------------------------------
    ("نزيف في اللثة عند التفريش", GINGI, "Non-Urgent"),
    ("اللثة تنزف كل يوم", GINGI, "Non-Urgent"),
    ("دم من اللثة عند استخدام الخيط", GINGI, "Non-Urgent"),
    ("لثتي حمراء وتنزف", GINGI, "Non-Urgent"),
    ("نزيف عند التفريش ومعها حساسية في اللثة", GINGI, "Non-Urgent"),
    ("دم عند استخدام الخيط", GINGI, "Non-Urgent"),
    ("ما في دم بس اللثة حمراء", UNCLEAR, "Non-Urgent"),
    # --- unclear / non-dental --------------------------------------------------
    ("", UNCLEAR, "Non-Urgent"),
    ("مرحبا", UNCLEAR, "Non-Urgent"),
    ("عندي صداع", UNCLEAR, "Non-Urgent"),
    ("ألم في قدمي", UNCLEAR, "Non-Urgent"),
    ("tooth pain", UNCLEAR, "Non-Urgent"),
    ("اشتريت ندم", UNCLEAR, "Non-Urgent"),
    ("عندي ألم في القدم والندم", UNCLEAR, "Non-Urgent"),
    # --- red flags (urgent whatever the diagnosis) -----------------------------
    ("عندي حرارة وتورم في اللثة", ABSCESS, "Urgent"),
    ("صعوبة في البلع وألم في السن", UNCLEAR, "Urgent"),
    ("صعوبة التنفس مع ألم الأسنان", UNCLEAR, "Urgent"),
    ("لا أستطيع فتح فمي بعد قلع الضرس", UNCLEAR, "Urgent"),
    ("نزيف لا يتوقف بعد خلع السن", UNCLEAR, "Urgent"),
    ("سقطت وانكسر السن", UNCLEAR, "Urgent"),
    ("تورم الوجه وألم", ABSCESS, "Urgent"),
    ("تعرضت لحادث وألم في الفك", UNCLEAR, "Urgent"),
]


class RuleEngineTests(unittest.TestCase):
    def test_cases(self):
        for text, want_diag, want_urgency in CASES:
            with self.subTest(text=text):
                diag, urgency, _sev, clean = run(text)
                self.assertEqual(diag, want_diag, f"diagnosis for {clean!r}")
                self.assertEqual(urgency, want_urgency, f"urgency for {clean!r}")

    def test_severity_is_never_low_when_urgent(self):
        for text, _d, _u in CASES:
            with self.subTest(text=text):
                _diag, urgency, severity, _ = run(text)
                if urgency == "Urgent":
                    self.assertEqual(severity, "High")

    def test_severity_follows_final_diagnosis(self):
        self.assertEqual(final_severity(ABSCESS, "Urgent"), "High")
        self.assertEqual(final_severity(DEEP, "Non-Urgent"), "Moderate")
        self.assertEqual(final_severity(SURF, "Non-Urgent"), "Low")
        self.assertEqual(final_severity(IRREV, "Urgent"), "High")
        self.assertEqual(final_severity(REV, "Non-Urgent"), "Moderate")


class PreprocessTests(unittest.TestCase):
    def test_rule_terms_and_text_share_one_normal_form(self):
        self.assertEqual(preprocess_text("ألم"), preprocess_text("الم"))
        self.assertEqual(preprocess_text("يوجعني"), "الم")
        self.assertEqual(preprocess_text("ضرس"), "سن")

    def test_whole_word_replacement_only(self):
        self.assertEqual(preprocess_text("قدم"), "قدم")
        self.assertEqual(preprocess_text("ندم"), "ندم")
        self.assertEqual(preprocess_text("دم"), "نزيف")
        self.assertEqual(preprocess_text("بالدم"), "بالنزيف")

    def test_taa_marbuta_and_punctuation(self):
        self.assertEqual(preprocess_text("اللثة،  تنزف!!"), "اللثه نزيف")

    def test_empty_and_none(self):
        self.assertEqual(preprocess_text(""), "")
        self.assertEqual(preprocess_text(None), "")


class NegationScopeTests(unittest.TestCase):
    def test_negation_covers_coordinated_terms(self):
        text = preprocess_text("بدون ورم ولا قيح")
        self.assertFalse(has_any(text, ["تورم"]))
        self.assertFalse(has_any(text, ["قيح"]))

    def test_but_clause_ends_negation(self):
        text = preprocess_text("ما في تورم بس ألم شديد")
        self.assertFalse(has_any(text, ["تورم"]))
        self.assertTrue(has_any(text, [preprocess_text("ألم")]))

    def test_affirmative_not_negated(self):
        self.assertTrue(has_any(preprocess_text("عندي تورم"), ["تورم"]))


class RedFlagTests(unittest.TestCase):
    def test_fever_alone_is_not_a_flag(self):
        self.assertEqual(detect_red_flags(preprocess_text("عندي حرارة")), [])

    def test_fever_with_swelling(self):
        self.assertIn("fever_with_swelling_or_pus",
                      detect_red_flags(preprocess_text("حرارة وتورم في الخد")))

    def test_negated_flag_is_ignored(self):
        self.assertEqual(detect_red_flags(preprocess_text("ما في صعوبة في البلع")), [])


class InputValidationTests(unittest.TestCase):
    def test_arabic_ratio(self):
        self.assertGreater(arabic_letter_ratio("ألم في السن"), 0.9)
        self.assertEqual(arabic_letter_ratio("tooth pain"), 0.0)
        self.assertEqual(arabic_letter_ratio("12345"), 0.0)


if __name__ == "__main__":
    unittest.main()
