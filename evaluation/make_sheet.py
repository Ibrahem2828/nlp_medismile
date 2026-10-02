"""Build an annotation sheet for clinicians from a pool of candidate texts.

Usage (from the repository root):
    python -m evaluation.make_sheet pool.csv --out evaluation/data/annotation_sheet.csv --n 400

`pool.csv` needs a `text` column (an optional `source` column is kept). The sheet is
de-duplicated (exact + near duplicates, digits ignored), shuffled with a fixed seed and
gets empty annotator columns. Assign dev/test splits AFTER adjudication with
`assign_splits` so that near-identical texts always land in the same split.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from typing import Dict, Iterable, List, Sequence

from .dataset_checks import jaccard, normalise, read_rows, tokens

SHEET_COLUMNS = [
    "id", "text", "source", "group_id",
    "label_a", "urgency_a", "red_flags_a",
    "label_b", "urgency_b", "red_flags_b",
    "label", "urgency", "red_flags", "split", "notes",
]


def deduplicate(texts: Sequence[str], threshold: float = 0.9) -> List[int]:
    """Indices of texts to keep: first occurrence of every exact/near-duplicate cluster."""
    kept: List[int] = []
    kept_tokens: List[set] = []
    seen_exact = set()
    for i, text in enumerate(texts):
        key = normalise(text)
        if not key or key in seen_exact:
            continue
        toks = tokens(text)
        if any(jaccard(toks, other) >= threshold for other in kept_tokens):
            continue
        seen_exact.add(key)
        kept.append(i)
        kept_tokens.append(toks)
    return kept


def build_sheet(rows: Sequence[Dict[str, str]], n: int, seed: int = 13, threshold: float = 0.9) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    shuffled = [rows[i] for i in order]
    keep = deduplicate([r.get("text", "") for r in shuffled], threshold)[:n]
    sheet = []
    for new_id, idx in enumerate(keep, start=1):
        row = shuffled[idx]
        sheet.append({
            **{c: "" for c in SHEET_COLUMNS},
            "id": f"C{new_id:04d}",
            "text": row.get("text", "").strip(),
            "source": row.get("source", ""),
            "group_id": row.get("group_id", ""),
        })
    return sheet


def assign_splits(rows: List[Dict[str, str]], test_fraction: float = 0.5, seed: int = 13,
                  threshold: float = 0.9) -> List[Dict[str, str]]:
    """Fill `split`, keeping near-duplicate texts (and equal group_id) in the SAME split."""
    parent = list(range(len(rows)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        parent[find(a)] = find(b)

    toks = [tokens(r.get("text", "")) for r in rows]
    by_group: Dict[str, int] = {}
    for i, r in enumerate(rows):
        g = r.get("group_id", "")
        if g:
            if g in by_group:
                union(i, by_group[g])
            else:
                by_group[g] = i
    for i in range(len(rows)):
        for j in range(i):
            if jaccard(toks[i], toks[j]) >= threshold:
                union(i, j)

    clusters: Dict[int, List[int]] = {}
    for i in range(len(rows)):
        clusters.setdefault(find(i), []).append(i)
    ids = sorted(clusters)
    random.Random(seed).shuffle(ids)
    target_test = test_fraction * len(rows)
    test_count = 0
    for cid in ids:
        split = "test" if test_count < target_test else "dev"
        if split == "test":
            test_count += len(clusters[cid])
        for i in clusters[cid]:
            rows[i]["split"] = split
    return rows


def write_csv(path: str, rows: Sequence[Dict[str, str]], columns: Sequence[str] = SHEET_COLUMNS) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Iterable[str] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pool", help="CSV with a `text` column")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--near-dup-threshold", type=float, default=0.9)
    args = ap.parse_args(argv)
    sheet = build_sheet(read_rows(args.pool), args.n, args.seed, args.near_dup_threshold)
    write_csv(args.out, sheet)
    print(f"wrote {len(sheet)} unique texts to {args.out}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Arabic output on Windows consoles
    sys.exit(main())
