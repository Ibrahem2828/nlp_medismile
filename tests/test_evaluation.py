"""Tests for the evaluation framework itself (it will judge the model, so it must be right)."""
import csv
import json
import os
import tempfile
import unittest

from evaluation import agreement, dataset_checks, make_sheet, run_eval
from evaluation.labels import LABELS
from evaluation.metrics import (accuracy, cohen_kappa, confusion_matrix, expected_calibration_error,
                                macro_f1, per_class, risk_coverage, urgency_report, wilson_interval)

ABSCESS, DEEP, UNCLEAR = "خراج سني", "تسوس عميق", "غير واضح حالياً"


class MetricsTests(unittest.TestCase):
    def test_per_class_and_macro_f1(self):
        y_true = ["a", "a", "a", "b", "b", "c"]
        y_pred = ["a", "a", "b", "b", "b", "a"]
        stats = per_class(y_true, y_pred, ["a", "b", "c"])
        self.assertAlmostEqual(stats["a"]["precision"], 2 / 3)
        self.assertAlmostEqual(stats["a"]["recall"], 2 / 3)
        self.assertAlmostEqual(stats["b"]["precision"], 2 / 3)
        self.assertAlmostEqual(stats["b"]["recall"], 1.0)
        self.assertEqual(stats["c"]["f1"], 0.0)
        self.assertAlmostEqual(accuracy(y_true, y_pred), 4 / 6)
        self.assertAlmostEqual(macro_f1(stats), (2 / 3 + 0.8 + 0.0) / 3)

    def test_macro_f1_ignores_classes_absent_from_gold(self):
        stats = per_class(["a", "a"], ["a", "a"], ["a", "z"])
        self.assertEqual(macro_f1(stats), 1.0)

    def test_confusion_matrix(self):
        m = confusion_matrix(["a", "a", "b"], ["a", "b", "b"], ["a", "b"])
        self.assertEqual(m, {"a": {"a": 1, "b": 1}, "b": {"a": 0, "b": 1}})

    def test_wilson_interval_is_wide_for_small_samples(self):
        lo, hi = wilson_interval(10, 10)
        self.assertLess(lo, 0.75)  # 10/10 does NOT prove 100%
        self.assertEqual(hi, 1.0)
        self.assertEqual(wilson_interval(0, 0), (0.0, 0.0))

    def test_cohen_kappa(self):
        self.assertEqual(cohen_kappa(["x", "y", "x", "y"], ["x", "y", "x", "y"]), 1.0)
        self.assertAlmostEqual(cohen_kappa(["x", "x", "y", "y"], ["x", "y", "x", "y"]), 0.0)
        self.assertAlmostEqual(cohen_kappa(["x", "x", "x", "y"], ["x", "x", "y", "y"]), 0.5)

    def test_urgency_report_counts_missed_urgent_cases(self):
        gold = ["Urgent", "Urgent", "Non-Urgent", "Non-Urgent", "Urgent"]
        pred = ["Urgent", "Non-Urgent", "Urgent", "Non-Urgent", "Urgent"]
        r = urgency_report(gold, pred)
        self.assertEqual((r["tp"], r["fn"], r["fp"], r["tn"]), (2, 1, 1, 1))
        self.assertAlmostEqual(r["recall"], 2 / 3)
        self.assertAlmostEqual(r["specificity"], 0.5)

    def test_calibration_and_risk_coverage(self):
        # perfectly calibrated: 0.9 confident and right 9 of 10
        conf = [0.9] * 10
        ok = [True] * 9 + [False]
        self.assertAlmostEqual(expected_calibration_error(conf, ok), 0.0, places=6)
        # over-confident
        self.assertAlmostEqual(expected_calibration_error([0.95] * 4, [True, False, False, False]), 0.7, places=6)
        curve = risk_coverage([0.9, 0.8, 0.2, 0.1], [True, True, False, False], points=4)
        self.assertEqual(curve[0]["accuracy"], 1.0)
        self.assertEqual(curve[-1]["accuracy"], 0.5)


class DatasetCheckTests(unittest.TestCase):
    def test_numbers_do_not_make_texts_different(self):
        rows = [{"text": "ألم في السن منذ 3 أيام", "split": "dev"},
                {"text": "ألم في السن منذ 5 أيام", "split": "test"}]
        leak = dataset_checks.split_leakage(rows)
        self.assertEqual(len(leak["exact"]), 1)

    def test_near_duplicate_across_splits_is_detected(self):
        a = "عندي ألم شديد في الضرس السفلي مع تورم في الخد منذ يومين"
        b = "عندي ألم شديد في الضرس السفلي مع تورم في الخد منذ يومين تقريبا"
        rows = [{"text": a, "split": "dev"}, {"text": b, "split": "test"},
                {"text": "نزيف عند التفريش", "split": "test"}]
        leak = dataset_checks.split_leakage(rows, threshold=0.8)
        self.assertEqual(len(leak["near"]), 1)
        self.assertEqual(leak["exact"], [])

    def test_shared_group_id_across_splits_is_leakage(self):
        rows = [{"text": "نص أول", "split": "dev", "group_id": "p1"},
                {"text": "نص ثان مختلف تماما", "split": "test", "group_id": "p1"}]
        self.assertEqual(dataset_checks.split_leakage(rows)["groups"], ["p1"])

    def test_same_split_duplicates_are_not_leakage(self):
        rows = [{"text": "نزيف عند التفريش", "split": "dev"}, {"text": "نزيف عند التفريش", "split": "dev"}]
        leak = dataset_checks.split_leakage(rows)
        self.assertEqual((leak["exact"], leak["near"], leak["groups"]), ([], [], []))


class SheetTests(unittest.TestCase):
    def test_dedupe_and_deterministic_sheet(self):
        pool = [{"text": "ألم في الضرس منذ 2 يوم"}, {"text": "ألم في الضرس منذ 7 يوم"},
                {"text": "نزيف في اللثة عند التفريش"}, {"text": ""}, {"text": "تورم في الخد وألم شديد"}]
        s1 = make_sheet.build_sheet(pool, n=10, seed=5)
        s2 = make_sheet.build_sheet(pool, n=10, seed=5)
        self.assertEqual([r["text"] for r in s1], [r["text"] for r in s2])
        self.assertEqual(len(s1), 3)  # near-duplicate and empty rows dropped
        self.assertEqual({c for c in make_sheet.SHEET_COLUMNS}, set(s1[0]))

    def test_assign_splits_keeps_near_duplicates_together(self):
        base = "عندي ألم شديد في الضرس السفلي مع تورم في الخد منذ يومين"
        rows = [{"text": base}, {"text": base + " تقريبا"}, {"text": "نزيف عند التفريش في اللثة"},
                {"text": "ألم مع البارد يزول بسرعة"}, {"text": "حفرة في السن وألم عند المضغ"},
                {"text": "اشعر بصعوبة في البلع"}]
        out = make_sheet.assign_splits(rows, test_fraction=0.5, seed=1, threshold=0.8)
        self.assertEqual(out[0]["split"], out[1]["split"])
        self.assertTrue(all(r["split"] in ("dev", "test") for r in out))
        self.assertEqual(dataset_checks.split_leakage(out, threshold=0.8)["near"], [])

    def test_group_id_forces_same_split(self):
        rows = [{"text": "نص أ", "group_id": "g"}, {"text": "نص مختلف تماما ب", "group_id": "g"},
                {"text": "آخر لا علاقة له", "group_id": ""}]
        out = make_sheet.assign_splits(rows, seed=3)
        self.assertEqual(out[0]["split"], out[1]["split"])


class AgreementTests(unittest.TestCase):
    ROWS = [
        {"label_a": ABSCESS, "label_b": ABSCESS, "urgency_a": "Urgent", "urgency_b": "Urgent"},
        {"label_a": DEEP, "label_b": ABSCESS, "urgency_a": "Non-Urgent", "urgency_b": "Urgent"},
        {"label_a": DEEP, "label_b": DEEP, "urgency_a": "Non-Urgent", "urgency_b": "Non-Urgent"},
        {"label_a": "", "label_b": DEEP, "urgency_a": "", "urgency_b": ""},  # unfinished row ignored
    ]

    def test_agreement_and_disagreements(self):
        result = agreement.agreement(self.ROWS)
        self.assertEqual(result["label"]["n"], 3)
        self.assertAlmostEqual(result["label"]["percent_agreement"], 2 / 3)
        self.assertEqual(len(agreement.disagreements(self.ROWS)), 1)
        self.assertIn("poor", agreement.interpret(0.3))
        self.assertIn("good", agreement.interpret(0.75))


def _write_gold(path, rows):
    cols = ["id", "text", "label", "urgency", "red_flags", "split", "group_id", "source"]
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


GOLD = [
    {"id": "1", "text": "ألم شديد مستمر يوقظني من النوم", "label": "التهاب عصب غير عكوس", "urgency": "Urgent", "split": "test"},
    {"id": "2", "text": "عندي تورم وألم", "label": ABSCESS, "urgency": "Urgent", "split": "test"},
    {"id": "3", "text": "بدون ورم ولا قيح وعندي ألم عند المضغ", "label": DEEP, "urgency": "Non-Urgent", "split": "test"},
    {"id": "4", "text": "نزيف في اللثة عند التفريش", "label": "التهاب لثة", "urgency": "Non-Urgent", "split": "test"},
    {"id": "5", "text": "صعوبة في البلع وألم في السن", "label": UNCLEAR, "urgency": "Urgent",
     "red_flags": "dysphagia", "split": "test"},
    {"id": "6", "text": "أريد فحصا روتينيا ولا يوجد ألم", "label": UNCLEAR, "urgency": "Non-Urgent", "split": "dev"},
]


class RunEvalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.gold = os.path.join(self.tmp.name, "gold.csv")
        self.out = os.path.join(self.tmp.name, "reports")
        _write_gold(self.gold, GOLD)

    def tearDown(self):
        self.tmp.cleanup()

    def test_rules_engine_end_to_end(self):
        code = run_eval.main([self.gold, "--engine", "rules", "--out", self.out, "--min-urgent-recall", "0.95"])
        self.assertEqual(code, 0)
        data = json.load(open(os.path.join(self.out, "report.json"), encoding="utf-8"))
        self.assertEqual(data["result"]["n"], 5)  # only the test split
        self.assertEqual(data["result"]["accuracy"], 1.0)
        self.assertEqual(data["result"]["urgency"]["recall"], 1.0)
        self.assertEqual(data["result"]["red_flag_recall"], 1.0)
        self.assertTrue(any("only 5 scored cases" in w for w in data["warnings"]))
        self.assertTrue(os.path.exists(os.path.join(self.out, "errors.csv")))

    def test_gate_fails_when_urgent_cases_are_missed(self):
        preds = os.path.join(self.tmp.name, "preds.csv")
        with open(preds, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "pred_label", "pred_urgency", "confidence"])
            for g in GOLD:
                w.writerow([g["id"], UNCLEAR, "Non-Urgent", "0.5"])  # a model that never raises an alarm
        code = run_eval.main([self.gold, "--predictions", preds, "--out", self.out])
        self.assertEqual(code, 1)
        data = json.load(open(os.path.join(self.out, "report.json"), encoding="utf-8"))
        self.assertTrue(any("urgent recall" in f for f in data["failures"]))
        self.assertEqual(data["result"]["under_triage"], 3)
        self.assertIn("ece", data["result"])

    def test_leakage_fails_the_gate(self):
        rows = GOLD + [{"id": "7", "text": "عندي تورم وألم", "label": ABSCESS, "urgency": "Urgent", "split": "dev"}]
        _write_gold(self.gold, rows)
        code = run_eval.main([self.gold, "--engine", "rules", "--out", self.out])
        self.assertEqual(code, 1)

    def test_unknown_label_is_rejected(self):
        _write_gold(self.gold, GOLD + [{"id": "9", "text": "x y z", "label": "تشخيص جديد", "urgency": "Urgent", "split": "test"}])
        with self.assertRaises(SystemExit):
            run_eval.main([self.gold, "--engine", "rules", "--out", self.out])

    def test_markdown_contains_the_key_sections(self):
        result = run_eval.score([{"label": ABSCESS, "urgency": "Urgent", "text": "t"}] * 2,
                                [{"label": ABSCESS, "urgency": "Urgent"}, {"label": DEEP, "urgency": "Non-Urgent"}])
        md = run_eval.render_markdown(result, "x", "test", [], [])
        for needle in ("accuracy", "urgent recall", "Per class", "Confusion matrix", "PASS"):
            self.assertIn(needle, md)
        self.assertEqual(result["under_triage"], 1)

    def test_labels_constant_matches_service_vocabulary(self):
        from src.symptom_understanding.text_classifier import classify_text
        from src.symptom_understanding.text_preprocess import preprocess_text
        for text, _ in [("ألم شديد مستمر بالليل", 0), ("تورم وألم", 0), ("نزيف في اللثة عند التفريش", 0),
                        ("ألم مع البارد يختفي بسرعة", 0), ("حساسية من الحلو", 0), ("ألم عند المضغ", 0), ("", 0)]:
            self.assertIn(classify_text(preprocess_text(text, keep_boundaries=True))[0], LABELS)


if __name__ == "__main__":
    unittest.main()
