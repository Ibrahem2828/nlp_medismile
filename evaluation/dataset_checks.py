"""Dataset hygiene checks: duplicates, near-duplicates and train/test leakage.

Usage (from the repository root):
    python -m evaluation.dataset_checks evaluation/data/gold.csv
Exit code is 1 when the same (or near-same) text appears in both splits.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from src.symptom_understanding.text_preprocess import preprocess_text

_DIGITS = re.compile(r"\d+")


def normalise(text: str) -> str:
    """Normalise like the service does and drop digits (templates often differ only by numbers)."""
    return _DIGITS.sub("", preprocess_text(text or "")).strip()


def tokens(text: str) -> Set[str]:
    return set(normalise(text).split())


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def read_rows(path: str) -> List[Dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def exact_duplicates(texts: Sequence[str]) -> Dict[str, List[int]]:
    seen: Dict[str, List[int]] = defaultdict(list)
    for i, text in enumerate(texts):
        seen[normalise(text)].append(i)
    return {k: v for k, v in seen.items() if len(v) > 1 and k}


def near_duplicate_pairs(texts: Sequence[str], threshold: float = 0.9, limit: int = 5000) -> List[Tuple[int, int, float]]:
    """Pairs of texts with Jaccard token overlap >= threshold (inverted index keeps it fast)."""
    sets = [tokens(t) for t in texts]
    index: Dict[str, List[int]] = defaultdict(list)
    pairs: List[Tuple[int, int, float]] = []
    for i, toks in enumerate(sets):
        candidates: Counter = Counter()
        for tok in toks:
            for j in index[tok]:
                candidates[j] += 1
        for j in candidates:
            score = jaccard(toks, sets[j])
            if score >= threshold:
                pairs.append((j, i, score))
                if len(pairs) >= limit:
                    return pairs
        for tok in toks:
            index[tok].append(i)
    return pairs


def split_leakage(rows: Sequence[Dict[str, str]], threshold: float = 0.9) -> Dict[str, object]:
    """Texts (exact/near) or group_ids shared between different splits."""
    texts = [r.get("text", "") for r in rows]
    splits = [r.get("split", "") for r in rows]
    cross_exact = []
    for _key, idxs in exact_duplicates(texts).items():
        if len({splits[i] for i in idxs}) > 1:
            cross_exact.append(idxs)
    cross_near = [(a, b, s) for a, b, s in near_duplicate_pairs(texts, threshold) if splits[a] != splits[b]]
    groups: Dict[str, Set[str]] = defaultdict(set)
    for r in rows:
        if r.get("group_id"):
            groups[r["group_id"]].add(r.get("split", ""))
    cross_groups = sorted(g for g, s in groups.items() if len(s) > 1)
    return {"exact": cross_exact, "near": cross_near, "groups": cross_groups}


def summary(rows: Sequence[Dict[str, str]], threshold: float = 0.9) -> Dict[str, object]:
    texts = [r.get("text", "") for r in rows]
    lengths = sorted(len(t.split()) for t in texts)
    labels = Counter(r.get("label", "") for r in rows)
    splits = Counter(r.get("split", "") for r in rows)
    sources = Counter(r.get("source", "") for r in rows)
    dup = exact_duplicates(texts)
    return {
        "rows": len(rows),
        "unique_normalised_texts": len({normalise(t) for t in texts}),
        "exact_duplicate_groups": len(dup),
        "near_duplicate_pairs": len(near_duplicate_pairs(texts, threshold)),
        "label_counts": dict(labels),
        "split_counts": dict(splits),
        "source_counts": dict(sources),
        "words_min_median_max": (lengths[0], lengths[len(lengths) // 2], lengths[-1]) if lengths else (0, 0, 0),
        "leakage": split_leakage(rows, threshold),
    }


def main(argv: Iterable[str] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", help="gold/annotated CSV (columns: text, label, split, group_id, source)")
    ap.add_argument("--near-dup-threshold", type=float, default=0.9)
    args = ap.parse_args(argv)
    rows = read_rows(args.csv)
    info = summary(rows, args.near_dup_threshold)
    leak = info.pop("leakage")
    for key, value in info.items():
        print(f"{key}: {value}")
    print(f"cross-split exact duplicates: {len(leak['exact'])}")
    print(f"cross-split near duplicates : {len(leak['near'])}")
    print(f"cross-split group ids       : {len(leak['groups'])}")
    return 1 if (leak["exact"] or leak["near"] or leak["groups"]) else 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Arabic output on Windows consoles
    sys.exit(main())
