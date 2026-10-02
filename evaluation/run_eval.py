"""Evaluate the symptom service against a clinician-labelled gold set.

Usage (from the repository root):
    python -m evaluation.run_eval evaluation/data/gold.csv --engine rules
    python -m evaluation.run_eval evaluation/data/gold.csv --engine api --api-url https://nlp.medismile.xn--mgbaab0cxheq.tech
    python -m evaluation.run_eval evaluation/data/gold.csv --engine full          # needs torch + AraBERT
    python -m evaluation.run_eval evaluation/data/gold.csv --predictions preds.csv # id,pred_label,pred_urgency[,confidence]

Gold CSV columns: id, text, label, urgency, [red_flags], split, [group_id], [source].
Only the `--split` rows (default: test) are scored; the gate fails (exit code 1) on cross-split
leakage or when a quality threshold is missed.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.request
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from . import dataset_checks
from .labels import LABELS, NON_URGENT, UNCLEAR, URGENT
from .metrics import (accuracy, confusion_matrix, expected_calibration_error, macro_f1, per_class,
                      risk_coverage, urgency_report, wilson_interval)

Prediction = Dict[str, object]  # keys: label, urgency, confidence (optional), red_flags (optional)


# ----------------------------------------------------------------------------- engines
def engine_rules(text: str) -> Prediction:
    """Rule engine only (no model download)."""
    from src.rules.severity import final_severity
    from src.rules.urgency import detect_red_flags, determine_urgency
    from src.symptom_understanding.text_classifier import classify_text
    from src.symptom_understanding.text_preprocess import preprocess_text

    rule_text = preprocess_text(text, keep_boundaries=True)
    label, severity = classify_text(rule_text)
    urgency = determine_urgency(label, severity, rule_text)
    return {"label": label, "urgency": urgency, "confidence": None,
            "red_flags": detect_red_flags(rule_text), "severity": final_severity(label, urgency)}


def engine_full(text: str) -> Prediction:
    """Rules + AraBERT, exactly as served (imports torch/transformers)."""
    from src.symptom_understanding.inference import analyze_symptoms

    result = analyze_symptoms(text)
    return {"label": result["diagnosis"], "urgency": result["urgency"],
            "confidence": result["confidence"], "red_flags": result.get("red_flags", [])}


def make_api_engine(base_url: str, timeout: int = 90) -> Callable[[str], Prediction]:
    def call(text: str) -> Prediction:
        body = json.dumps({"text": text}, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(base_url.rstrip("/") + "/api/analyze-text", body,
                                     {"Content-Type": "application/json; charset=utf-8"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        return {"label": data["diagnosis"], "urgency": data["urgency"],
                "confidence": data.get("confidence"), "red_flags": data.get("red_flags", [])}
    return call


# ----------------------------------------------------------------------------- scoring
def score(gold: Sequence[Dict[str, str]], preds: Sequence[Prediction]) -> Dict[str, object]:
    y_true = [g["label"] for g in gold]
    y_pred = [str(p["label"]) for p in preds]
    labels = [l for l in LABELS if l in set(y_true) | set(y_pred)] or list(LABELS)
    classes = per_class(y_true, y_pred, labels)
    n = len(gold)
    correct_n = sum(1 for t, p in zip(y_true, y_pred) if t == p)

    gold_urg = [g["urgency"] for g in gold]
    pred_urg = [str(p["urgency"]) for p in preds]
    urg = urgency_report(gold_urg, pred_urg)

    answered = [i for i, p in enumerate(y_pred) if p != UNCLEAR]
    result: Dict[str, object] = {
        "n": n,
        "accuracy": accuracy(y_true, y_pred),
        "accuracy_ci95": wilson_interval(correct_n, n),
        "macro_f1": macro_f1(classes),
        "per_class": classes,
        "confusion": confusion_matrix(y_true, y_pred, labels),
        "urgency": urg,
        "abstain_rate": 1 - len(answered) / n if n else 0.0,
        "accuracy_when_answered": (sum(1 for i in answered if y_true[i] == y_pred[i]) / len(answered)) if answered else 0.0,
        "over_triage": sum(1 for g, p in zip(gold_urg, pred_urg) if g == NON_URGENT and p == URGENT),
        "under_triage": sum(1 for g, p in zip(gold_urg, pred_urg) if g == URGENT and p != URGENT),
        "labels": labels,
    }
    confidences = [p.get("confidence") for p in preds]
    if all(isinstance(c, (int, float)) for c in confidences) and confidences:
        ok = [t == p for t, p in zip(y_true, y_pred)]
        result["ece"] = expected_calibration_error([float(c) for c in confidences], ok)
        result["risk_coverage"] = risk_coverage([float(c) for c in confidences], ok)

    flagged = [(g.get("red_flags", ""), p.get("red_flags") or []) for g, p in zip(gold, preds) if g.get("red_flags")]
    if flagged:
        hit = sum(1 for want, got in flagged if all(f in got for f in want.split(";") if f))
        result["red_flag_recall"] = hit / len(flagged)
        result["red_flag_cases"] = len(flagged)
    return result


def reliability_warnings(result: Dict[str, object]) -> List[str]:
    warnings = []
    if result["n"] < 100:
        warnings.append(f"only {result['n']} scored cases: confidence intervals are wide, treat numbers as indicative")
    for label, stats in result["per_class"].items():
        if 0 < stats["support"] < 20:
            warnings.append(f"class '{label}' has only {int(stats['support'])} cases: its F1 is unreliable")
    if result["urgency"]["positives"] < 30:
        warnings.append(f"only {result['urgency']['positives']} urgent cases: urgent-recall CI is very wide")
    return warnings


def gate(result: Dict[str, object], leakage: Dict[str, object], min_urgent_recall: float,
         min_macro_f1: float, min_accuracy: float) -> List[str]:
    failures = []
    if leakage["exact"] or leakage["near"] or leakage["groups"]:
        failures.append("train/test leakage detected (exact, near-duplicate or shared group_id)")
    if result["urgency"]["recall"] < min_urgent_recall:
        failures.append(f"urgent recall {result['urgency']['recall']:.3f} < {min_urgent_recall}")
    if result["macro_f1"] < min_macro_f1:
        failures.append(f"macro-F1 {result['macro_f1']:.3f} < {min_macro_f1}")
    if result["accuracy"] < min_accuracy:
        failures.append(f"accuracy {result['accuracy']:.3f} < {min_accuracy}")
    return failures


# ----------------------------------------------------------------------------- reporting
def render_markdown(result: Dict[str, object], engine: str, split: str, warnings: Sequence[str],
                    failures: Sequence[str]) -> str:
    lo, hi = result["accuracy_ci95"]
    u = result["urgency"]
    ulo, uhi = u["recall_ci95"]
    lines = [
        f"# Evaluation report - engine `{engine}`, split `{split}`", "",
        f"- cases: **{result['n']}**",
        f"- accuracy: **{result['accuracy']:.3f}** (95% CI {lo:.3f}-{hi:.3f})",
        f"- macro-F1: **{result['macro_f1']:.3f}**",
        f"- urgent recall: **{u['recall']:.3f}** (95% CI {ulo:.3f}-{uhi:.3f}), missed urgent cases: {u['fn']} of {u['positives']}",
        f"- urgent precision: {u['precision']:.3f}, specificity: {u['specificity']:.3f}, over-triage: {result['over_triage']}",
        f"- abstain (unclear) rate: {result['abstain_rate']:.1%}, accuracy when answered: {result['accuracy_when_answered']:.3f}",
    ]
    if "ece" in result:
        lines.append(f"- calibration error (ECE): {result['ece']:.3f}")
    if "red_flag_recall" in result:
        lines.append(f"- red-flag recall: {result['red_flag_recall']:.3f} over {result['red_flag_cases']} cases")
    lines += ["", "## Per class", "", "| label | precision | recall | F1 | support |", "|---|---|---|---|---|"]
    for label, s in result["per_class"].items():
        lines.append(f"| {label} | {s['precision']:.3f} | {s['recall']:.3f} | {s['f1']:.3f} | {int(s['support'])} |")
    labels = result["labels"]
    lines += ["", "## Confusion matrix (rows = gold, columns = predicted)", "",
              "| gold \\ pred | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for t in labels:
        lines.append(f"| {t} | " + " | ".join(str(result["confusion"].get(t, {}).get(p, 0)) for p in labels) + " |")
    if "risk_coverage" in result:
        lines += ["", "## Risk-coverage (answer only the most confident cases)", "", "| coverage | accuracy |", "|---|---|"]
        for point in result["risk_coverage"]:
            lines.append(f"| {point['coverage']:.0%} | {point['accuracy']:.3f} |")
    if warnings:
        lines += ["", "## Reliability warnings", ""] + [f"- {w}" for w in warnings]
    lines += ["", "## Quality gate", ""]
    lines += [f"- FAIL: {f}" for f in failures] if failures else ["- PASS"]
    return "\n".join(lines) + "\n"


def write_errors(path: str, gold: Sequence[Dict[str, str]], preds: Sequence[Prediction]) -> int:
    rows = []
    for g, p in zip(gold, preds):
        if g["label"] != p["label"] or g["urgency"] != p["urgency"]:
            rows.append({"id": g.get("id", ""), "text": g["text"], "gold_label": g["label"], "pred_label": p["label"],
                         "gold_urgency": g["urgency"], "pred_urgency": p["urgency"],
                         "confidence": p.get("confidence", "")})
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "text", "gold_label", "pred_label", "gold_urgency",
                                                "pred_urgency", "confidence"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def load_predictions(path: str, gold: Sequence[Dict[str, str]]) -> List[Prediction]:
    by_id = {r["id"]: r for r in dataset_checks.read_rows(path)}
    missing = [g["id"] for g in gold if g["id"] not in by_id]
    if missing:
        raise SystemExit(f"predictions missing for {len(missing)} ids, e.g. {missing[:3]}")
    out = []
    for g in gold:
        r = by_id[g["id"]]
        conf = r.get("confidence", "")
        out.append({"label": r["pred_label"], "urgency": r["pred_urgency"],
                    "confidence": float(conf) if conf not in ("", None) else None})
    return out


def main(argv: Iterable[str] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gold")
    ap.add_argument("--engine", choices=["rules", "full", "api"], default="rules")
    ap.add_argument("--api-url")
    ap.add_argument("--predictions", help="evaluate an existing predictions CSV instead of running an engine")
    ap.add_argument("--split", default="test", choices=["dev", "test", "all"])
    ap.add_argument("--out", default="evaluation/reports")
    ap.add_argument("--min-urgent-recall", type=float, default=0.95)
    ap.add_argument("--min-macro-f1", type=float, default=0.0)
    ap.add_argument("--min-accuracy", type=float, default=0.0)
    args = ap.parse_args(argv)

    rows = dataset_checks.read_rows(args.gold)
    labelled = [r for r in rows if r.get("label") and r.get("urgency")]
    unknown = {r["label"] for r in labelled} - set(LABELS)
    if unknown:
        raise SystemExit(f"unknown labels in gold set: {sorted(unknown)}")
    scored = [r for r in labelled if args.split == "all" or r.get("split") == args.split]
    if not scored:
        raise SystemExit(f"no labelled rows in split '{args.split}'")

    if args.predictions:
        preds = load_predictions(args.predictions, scored)
        engine_name = f"predictions:{os.path.basename(args.predictions)}"
    else:
        if args.engine == "api":
            if not args.api_url:
                raise SystemExit("--api-url is required with --engine api")
            engine = make_api_engine(args.api_url)
        else:
            engine = engine_rules if args.engine == "rules" else engine_full
        preds = [engine(r["text"]) for r in scored]
        engine_name = args.engine

    result = score(scored, preds)
    leakage = dataset_checks.split_leakage(labelled)
    warnings = reliability_warnings(result)
    failures = gate(result, leakage, args.min_urgent_recall, args.min_macro_f1, args.min_accuracy)

    os.makedirs(args.out, exist_ok=True)
    report = render_markdown(result, engine_name, args.split, warnings, failures)
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as fh:
        fh.write(report)
    with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as fh:
        json.dump({"result": result, "warnings": warnings, "failures": failures}, fh, ensure_ascii=False, indent=2)
    errors = write_errors(os.path.join(args.out, "errors.csv"), scored, preds)

    print(report)
    print(f"{errors} misclassified cases written to {os.path.join(args.out, 'errors.csv')}")
    return 1 if failures else 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Arabic output on Windows consoles
    sys.exit(main())
