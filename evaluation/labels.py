"""Label vocabulary shared by every evaluation tool (kept free of heavy imports)."""
from __future__ import annotations

UNCLEAR = "غير واضح حالياً"

# Diagnoses the service can output, plus the "unclear / out of scope" abstention.
LABELS = [
    "خراج سني",
    "التهاب عصب غير عكوس",
    "التهاب عصب عكوس",
    "تسوس عميق",
    "تسوس سطحي",
    "التهاب لثة",
    UNCLEAR,
]

URGENT = "Urgent"
NON_URGENT = "Non-Urgent"
URGENCY_VALUES = [URGENT, NON_URGENT]

RED_FLAG_CODES = [
    "trismus",
    "dysphagia",
    "breathing_difficulty",
    "uncontrolled_bleeding",
    "trauma",
    "spreading_facial_swelling",
    "fever_with_swelling_or_pus",
]

SPLITS = ("dev", "test")
