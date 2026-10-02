# src/symptom_understanding/text_utils.py
"""Shared matching helpers (word-boundary matching + negation scope) for the rules."""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable, List, Tuple

from .text_preprocess import preprocess_text

_AR = "ء-ي"
_PREFIX = r"(?:وال|بال|فال|كال|لل|ال|و|ف|ب|ل|ك)?"
_SUFFIX = r"(?:ات|ين|ان|ها|ه|ي|ك|ت)?"

# Phrases that negate the symptom that follows them (already in normalised form).
_NEGATORS_RAW = [
    "بدون", "ما في", "مافي", "ما فيه", "مفيش", "لا يوجد", "لايوجد", "ما عندي",
    "ماعندي", "ما عندنا", "مو موجود", "ما اعاني من", "لا اعاني من", "ما اشعر ب",
    "ليس", "ليس لدي", "عدم", "خالي من", "مش", "مو", "غير", "ولا", "لا توجد", "ما توجد",
]
# Words that end a negation scope ("no swelling BUT severe pain"). "|" is the sentence
# boundary marker produced by preprocess_text(keep_boundaries=True).
_SCOPE_BREAKERS = {"|", "بس", "لكن", "ولكن", "الا", "انما", "عندي", "وعندي", "فيه", "وفيه",
                   "كمان", "وكمان", "اشعر", "واشعر", "اعاني", "واعاني", "احس", "واحس"}
# "ولا X" only negates the item right after it (it continues a negation, it does not open one).
_ADJACENT_ONLY = {("ولا",)}
# A negator covers the next 3 tokens; "بدون الم تلقائي او تورم" (explicit "أو") reaches 4.
_BASE_NEGATION_WINDOW = 3
_NEGATION_WINDOW = 4


def prep(term: str) -> str:
    return preprocess_text(term)


def _with_conjunction(negator: Tuple[str, ...]) -> Tuple[str, ...]:
    """'لا يوجد' -> 'ولا يوجد' (a negation opening a new coordinated clause)."""
    first = negator[0]
    return negator if first.startswith("و") else ("و" + first,) + negator[1:]


_NEGATORS: List[Tuple[str, ...]] = []
for _raw in _NEGATORS_RAW:
    _tokens = tuple(prep(_raw).split())
    for _variant in (_tokens, _with_conjunction(_tokens)):
        if _variant not in _NEGATORS:
            _NEGATORS.append(_variant)


@lru_cache(maxsize=2048)
def _pattern(term: str) -> "re.Pattern[str]":
    return re.compile(rf"(?<![{_AR}]){_PREFIX}{re.escape(term)}{_SUFFIX}(?![{_AR}])")


def _is_negated(text: str, start: int) -> bool:
    tokens = text[:start].split()
    if not tokens:
        return False
    for end in range(1, _NEGATION_WINDOW + 1):  # negator ends `end` tokens before the term
        if end > len(tokens):
            break
        between = tokens[len(tokens) - end + 1:] if end > 1 else []
        if any(tok in _SCOPE_BREAKERS for tok in between):
            continue
        if end > _BASE_NEGATION_WINDOW and "او" not in between:
            continue  # only an explicit "أو" coordination reaches the extended window
        for negator in _NEGATORS:
            if negator in _ADJACENT_ONLY and end != 1:
                continue
            length = len(negator)
            first = len(tokens) - end - length + 1
            if first >= 0 and tuple(tokens[first:first + length]) == negator:
                return True
    return False


def has_any(text: str, terms: Iterable[str], *, respect_negation: bool = True) -> bool:
    """True if any (already normalised) term occurs as whole word(s) and is not negated."""
    for term in terms:
        for match in _pattern(term).finditer(text):
            if not respect_negation or not _is_negated(text, match.start()):
                return True
    return False


def prep_all(terms: Iterable[str]) -> List[str]:
    return [prep(t) for t in terms]


def arabic_letter_ratio(text: str) -> float:
    letters = re.findall(r"[A-Za-zء-ي]", text or "")
    if not letters:
        return 0.0
    return sum(1 for c in letters if "ء" <= c <= "ي") / len(letters)
