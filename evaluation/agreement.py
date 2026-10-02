"""Inter-annotator agreement and the list of cases that need adjudication.

Usage (from the repository root):
    python -m evaluation.agreement evaluation/data/annotation_sheet.csv --disagreements evaluation/data/to_adjudicate.csv

Reports percent agreement and Cohen's kappa for the diagnosis and the urgency label
(target: kappa >= 0.70 before trusting the gold set; below 0.60 the guidelines need work).
"""
from __future__ import annotations

import argparse
import sys
from typing import Dict, Iterable, List, Sequence

from .dataset_checks import read_rows
from .make_sheet import SHEET_COLUMNS, write_csv
from .metrics import cohen_kappa


def _both(rows: Sequence[Dict[str, str]], a: str, b: str):
    pairs = [(r.get(a, "").strip(), r.get(b, "").strip()) for r in rows]
    return [(x, y) for x, y in pairs if x and y]


def agreement(rows: Sequence[Dict[str, str]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for name in ("label", "urgency"):
        pairs = _both(rows, f"{name}_a", f"{name}_b")
        a = [x for x, _ in pairs]
        b = [y for _, y in pairs]
        out[name] = {
            "n": len(pairs),
            "percent_agreement": sum(1 for x, y in pairs if x == y) / len(pairs) if pairs else 0.0,
            "kappa": cohen_kappa(a, b),
        }
    return out


def disagreements(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    out = []
    for r in rows:
        la, lb = r.get("label_a", "").strip(), r.get("label_b", "").strip()
        ua, ub = r.get("urgency_a", "").strip(), r.get("urgency_b", "").strip()
        if (la and lb and la != lb) or (ua and ub and ua != ub):
            out.append(r)
    return out


def interpret(kappa: float) -> str:
    if kappa >= 0.80:
        return "almost perfect"
    if kappa >= 0.70:
        return "good - acceptable for a gold set"
    if kappa >= 0.60:
        return "moderate - tighten the guidelines"
    return "poor - do not use as ground truth"


def main(argv: Iterable[str] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv")
    ap.add_argument("--disagreements", help="write the cases that need adjudication to this CSV")
    args = ap.parse_args(argv)
    rows = read_rows(args.csv)
    result = agreement(rows)
    for name, stats in result.items():
        print(f"{name:8s} n={stats['n']:4d}  agreement={stats['percent_agreement']:.1%}  "
              f"kappa={stats['kappa']:.3f} ({interpret(stats['kappa'])})")
    todo = disagreements(rows)
    print(f"cases needing adjudication: {len(todo)}")
    if args.disagreements:
        write_csv(args.disagreements, todo, SHEET_COLUMNS)
        print(f"wrote {args.disagreements}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Arabic output on Windows consoles
    sys.exit(main())
