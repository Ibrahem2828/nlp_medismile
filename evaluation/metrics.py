"""Dependency-free evaluation metrics (stdlib only)."""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def confusion_matrix(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> Dict[str, Dict[str, int]]:
    matrix = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        matrix.setdefault(t, {q: 0 for q in labels})
        matrix[t][p] = matrix[t].get(p, 0) + 1
    return matrix


def per_class(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """Precision / recall / F1 / support per label (labels absent from gold get support 0)."""
    out: Dict[str, Dict[str, float]] = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out[label] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
    return out


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true) if y_true else 0.0


def macro_f1(per_class_scores: Dict[str, Dict[str, float]]) -> float:
    """Macro-F1 over the classes that actually occur in the gold set."""
    scores = [v["f1"] for v in per_class_scores.values() if v["support"] > 0]
    return sum(scores) / len(scores) if scores else 0.0


def wilson_interval(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    """95% Wilson score interval; honest about small samples."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def expected_calibration_error(confidences: Sequence[float], correct: Sequence[bool], bins: int = 10) -> float:
    n = len(confidences)
    if n == 0:
        return 0.0
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confidences) if (lo <= c < hi) or (b == bins - 1 and c == 1.0)]
        if not idx:
            continue
        acc = sum(1 for i in idx if correct[i]) / len(idx)
        conf = sum(confidences[i] for i in idx) / len(idx)
        total += len(idx) / n * abs(acc - conf)
    return total


def risk_coverage(confidences: Sequence[float], correct: Sequence[bool], points: int = 10) -> List[Dict[str, float]]:
    """Accuracy when only the most confident `coverage` fraction of cases is answered."""
    order = sorted(range(len(confidences)), key=lambda i: confidences[i], reverse=True)
    out = []
    for k in range(1, points + 1):
        take = max(1, round(len(order) * k / points)) if order else 0
        chosen = order[:take]
        acc = sum(1 for i in chosen if correct[i]) / len(chosen) if chosen else 0.0
        out.append({"coverage": take / len(order) if order else 0.0, "accuracy": acc})
    return out


def cohen_kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Cohen's kappa between two annotators (1.0 perfect, 0 chance-level)."""
    n = len(a)
    if n == 0:
        return 0.0
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    cats = set(a) | set(b)
    expected = sum((sum(1 for x in a if x == c) / n) * (sum(1 for y in b if y == c) / n) for c in cats)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def urgency_report(gold: Sequence[str], pred: Sequence[str], positive: str = "Urgent") -> Dict[str, object]:
    """Safety view: how many truly urgent cases were missed (under-triage)."""
    tp = sum(1 for g, p in zip(gold, pred) if g == positive and p == positive)
    fn = sum(1 for g, p in zip(gold, pred) if g == positive and p != positive)
    fp = sum(1 for g, p in zip(gold, pred) if g != positive and p == positive)
    tn = sum(1 for g, p in zip(gold, pred) if g != positive and p != positive)
    pos = tp + fn
    neg = tn + fp
    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "recall": tp / pos if pos else 0.0,
        "recall_ci95": wilson_interval(tp, pos),
        "precision": tp / (tp + fp) if (tp + fp) else 0.0,
        "specificity": tn / neg if neg else 0.0,
        "positives": pos,
    }
