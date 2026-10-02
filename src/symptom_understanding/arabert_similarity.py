# src/symptom_understanding/arabert_similarity.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModel


@dataclass
class ArabertSuggestion:
    label: str
    score: float
    second_label: Optional[str] = None
    second_score: Optional[float] = None


class AraBERTSemanticClassifier:
    """
    AraBERT encoder-based semantic similarity classifier (no fine-tuning here).
    Uses mean-pooling embeddings and cosine similarity to prototypes.
    """

    def __init__(
        self,
        model_name: str = "aubmindlab/bert-base-arabertv2",
        device: Optional[str] = None,
        min_score: float = 0.52,
        min_margin: float = 0.05,
    ) -> None:
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.min_score = min_score
        self.min_margin = min_margin

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

        # Prototypes per label (can be expanded with clinician feedback)
        self.prototypes: Dict[str, List[str]] = {
            "خراج سني": [
                "تورم في الوجه مع قيح وألم شديد في الضرس",
                "انتفاخ وانتشار قيح حول اللثة مع ألم لا يحتمل",
                "ورم وصديد في نفس السن وألم يزيد مع الضغط",
            ],
            "التهاب عصب غير عكوس": [
                "ألم شديد مستمر يوقظني من النوم",
                "ألم نبضي لا يهدأ ويزيد بالليل",
                "ألم حارق قوي طوال اليوم ولا يزول",
            ],
            "التهاب عصب عكوس": [
                "ألم مع البارد يختفي بسرعة بعد ثواني",
                "حساسية مع الحلو أو البارد ثم يزول الألم",
                "يوجع مع المشروبات الباردة لكن يرتاح بعد إزالتها",
            ],
            "تسوس عميق": [
                "ألم عند المضغ أو الضغط على السن مع حفرة واضحة",
                "تجويف في الضرس وألم قوي عند الأكل",
                "ثقب في السن يسبب ألم مع العض",
            ],
            "تسوس سطحي": [
                "حساسية بسيطة مع البارد أو الحلو وتزول بسرعة",
                "ألم خفيف مع العصير البارد لكن لا يستمر",
                "وخز مع الحلويات يذهب بعد لحظات",
            ],
            "التهاب لثة": [
                "نزيف عند التفريش مع لثة حمراء",
                "دم من اللثة وانتفاخ بسيط",
                "اللثة تنزف بسهولة مع ريحة فم",
            ],
            "غير واضح حالياً": [
                "الأعراض غير واضحة",
                "لا أستطيع وصف الألم بدقة",
            ],
        }

        self.prototype_embeddings: Dict[str, torch.Tensor] = {}
        self._build_prototype_index()

    @torch.no_grad()
    def _encode(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        ).to(self.device)

        outputs = self.model(**inputs)
        last_hidden = outputs.last_hidden_state  # [B, T, H]
        mask = inputs["attention_mask"].unsqueeze(-1).float()  # [B, T, 1]

        summed = torch.sum(last_hidden * mask, dim=1)
        counted = torch.clamp(mask.sum(dim=1), min=1e-9)
        emb = summed / counted  # [B, H]

        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb[0]

    def _build_prototype_index(self) -> None:
        for label, texts in self.prototypes.items():
            embs = [self._encode(t) for t in texts]
            avg = torch.stack(embs, dim=0).mean(dim=0)
            avg = torch.nn.functional.normalize(avg, p=2, dim=0)
            self.prototype_embeddings[label] = avg

    @staticmethod
    def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(torch.dot(a, b).clamp(-1, 1).item())

    def suggest(self, text: str) -> ArabertSuggestion:
        query = self._encode(text)

        scored: List[Tuple[str, float]] = []
        for label, emb in self.prototype_embeddings.items():
            scored.append((label, self._cosine(query, emb)))

        scored.sort(key=lambda x: x[1], reverse=True)
        best_label, best_score = scored[0]
        second_label, second_score = scored[1] if len(scored) > 1 else (None, None)

        if best_score < self.min_score:
            return ArabertSuggestion(label="غير واضح حالياً", score=best_score, second_label=second_label, second_score=second_score)

        if second_score is not None and (best_score - second_score) < self.min_margin:
            return ArabertSuggestion(label="غير واضح حالياً", score=best_score, second_label=second_label, second_score=second_score)

        return ArabertSuggestion(label=best_label, score=best_score, second_label=second_label, second_score=second_score)
