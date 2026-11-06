"""Semantic post-processing for over-segmented question decompositions."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Literal, Protocol, Sequence

from qralia.rag.question_decomposition import (
    AtomicQuestion,
    DecompositionResult,
    ROOT,
    _search_form,
)


DEFAULT_INPUT = ROOT / "outputs/question-decomposition-qwen35-v0.1.json"
DEFAULT_OUTPUT = ROOT / "outputs/question-decomposition-qwen35-fusion-v0.2-exploratory.json"


class EmbeddingBackend(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


# Each prototype deliberately contains the three scripts evaluated in v0.1.
# These are semantic anchors, not training examples and not routing decisions.
INTENT_PROTOTYPES = {
    "invoice.total_due": (
        "montant total à payer combien régler facture "
        "كم يجب أن أدفع المبلغ الإجمالي شحال خاصني نخلص المبلغ كامل"
    ),
    "invoice.due_date": (
        "date limite échéance dernier jour pour payer "
        "تاريخ الاستحقاق آخر أجل فوقاش آخر نهار"
    ),
    "invoice.identifier": (
        "numéro de facture numéro de contrat identifiant "
        "رقم الفاتورة رقم العقد نمرة الفاتورة نمرة الكونطرا"
    ),
    "invoice.service_bundle": (
        "eau et électricité présentes ensemble sur la même facture "
        "الماء والكهرباء في نفس الفاتورة الما والضو مجموعين"
    ),
    "invoice.amount_composition": (
        "composition du montant taxes frais inclus "
        "هل يشمل المبلغ الضرائب والرسوم واش المبلغ فيه الضريبة والمصاريف"
    ),
    "official.payment_channels": (
        "payer en agence en ligne canal ou lieu de paiement "
        "الأداء في الوكالة أو عبر الإنترنت نخلص فالوكالة ولا فالإنترنت"
    ),
    "official.payment_methods": (
        "payer par carte chèque espèces moyen de paiement "
        "الدفع نقدا أو بالشيك نخلص بالكارت ولا بالشيك"
    ),
    "official.late_payment": (
        "conséquence retard de paiement que se passe-t-il "
        "ماذا يحدث عند التأخر شنو يوقع إلا تعطلت"
    ),
}


DIMENSION_PATTERNS = {
    "amount": re.compile(r"\b(combien|montant|total)\b|كم|شحال|المبلغ"),
    "location": re.compile(r"\b(où|agence)\b|أين|فين|وكالة|الوكالة"),
    "time": re.compile(r"\b(quand|date|échéance|délai)\b|متى|تاريخ|أجل|فوقاش|نهار"),
    "method": re.compile(r"\b(comment|méthode|moyen)\b|كيف|كيفاش"),
    "payment_method": re.compile(
        r"\b(chèque|cheque|carte|espèces|especes|virement)\b|"
        r"شيك|بالشيك|كارت|بالكارت|نقدا|نقداً|تحويل"
    ),
    "identifier": re.compile(r"\b(numéro|identifiant|contrat)\b|رقم|نمرة|العقد|الكونطرا"),
    "consequence": re.compile(r"se passe|arrive|conséquence|ماذا يحدث|شنو يوقع|يحدث|تأخرت|تعطلت"),
}
OPTION_CONNECTOR = re.compile(r"\b(ou|ou bien|soit)\b|أو|ولا")
FRAGMENT_PREFIX = re.compile(r"^(et|ou|en|par|avec|dans|و|أو|ولا|ب|في|ف)")


@dataclass(frozen=True)
class MergeSignals:
    left_index: int
    right_index: int
    left_top_intent: str
    right_top_intent: str
    same_top_intent: bool
    intent_similarity: float
    sentence_similarity: float
    option_connector: bool
    shared_predicate: bool
    incomplete_fragment: bool
    different_question_dimension: bool
    merge: bool
    reason: str


@dataclass(frozen=True)
class FusionResult:
    original: DecompositionResult
    final: DecompositionResult
    pair_signals: tuple[MergeSignals, ...]
    applied: bool


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return float(sum(a * b for a, b in zip(left, right)))


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    denominator = math.sqrt(_dot(left, left) * _dot(right, right))
    return _dot(left, right) / denominator if denominator else 0.0


def _dimensions(text: str) -> set[str]:
    normalized = _search_form(text)
    return {name for name, pattern in DIMENSION_PATTERNS.items() if pattern.search(normalized)}


def _span_between(question: str, left: str, right: str) -> str:
    normalized_question = _search_form(question)
    normalized_left = _search_form(left)
    normalized_right = _search_form(right)
    left_start = normalized_question.find(normalized_left)
    right_start = normalized_question.find(normalized_right, max(left_start, 0))
    if left_start < 0 or right_start < 0:
        return normalized_question
    return normalized_question[left_start + len(normalized_left) : right_start]


def _is_incomplete_fragment(need: AtomicQuestion) -> bool:
    source = _search_form(need.source_span)
    words = source.split()
    return bool(FRAGMENT_PREFIX.search(source)) or (len(words) <= 4 and not _dimensions(source))


class SemanticQuestionFusion:
    """Merge adjacent needs only when several independent signals agree."""

    def __init__(
        self,
        embeddings: EmbeddingBackend,
        *,
        sentence_similarity_threshold: float = 0.78,
        intent_similarity_threshold: float = 0.90,
        policy: Literal["broad", "conservative"] = "broad",
    ) -> None:
        if policy not in {"broad", "conservative"}:
            raise ValueError(f"Politique de fusion inconnue : {policy}")
        self.embeddings = embeddings
        self.sentence_similarity_threshold = sentence_similarity_threshold
        self.intent_similarity_threshold = intent_similarity_threshold
        self.policy = policy
        self._intent_names = tuple(INTENT_PROTOTYPES)
        self._prototype_vectors = embeddings.embed_documents(
            [INTENT_PROTOTYPES[name] for name in self._intent_names]
        )

    def _intent_profile(self, vector: Sequence[float]) -> tuple[str, list[float]]:
        scores = [_cosine(vector, prototype) for prototype in self._prototype_vectors]
        best = max(range(len(scores)), key=scores.__getitem__)
        # Negative cosine values carry no useful evidence in this small profile.
        profile = [max(score, 0.0) for score in scores]
        return self._intent_names[best], profile

    def fuse(self, question: str, decomposition: DecompositionResult) -> FusionResult:
        if len(decomposition.needs) < 2:
            return FusionResult(decomposition, decomposition, (), False)

        needs = decomposition.needs
        vectors = self.embeddings.embed_documents(
            [need.standalone_question for need in needs]
        )
        intent_data = [self._intent_profile(vector) for vector in vectors]
        original_dimensions = _dimensions(question)
        parents = list(range(len(needs)))

        def find(index: int) -> int:
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parents[right_root] = left_root

        analyses: list[MergeSignals] = []
        for left_index in range(len(needs) - 1):
            right_index = left_index + 1
            left, right = needs[left_index], needs[right_index]
            left_intent, left_profile = intent_data[left_index]
            right_intent, right_profile = intent_data[right_index]
            left_dimensions = _dimensions(left.standalone_question)
            right_dimensions = _dimensions(right.standalone_question)
            different_dimension = bool(
                left_dimensions and right_dimensions and left_dimensions.isdisjoint(right_dimensions)
            )
            sentence_similarity = _cosine(vectors[left_index], vectors[right_index])
            intent_similarity = _cosine(left_profile, right_profile)
            between = _span_between(question, left.source_span, right.source_span)
            option_connector = bool(
                OPTION_CONNECTOR.search(between)
                or OPTION_CONNECTOR.search(_search_form(right.source_span))
            )
            incomplete = _is_incomplete_fragment(right)
            same_intent = left_intent == right_intent
            shared_predicate = (
                same_intent
                and sentence_similarity >= self.sentence_similarity_threshold
                and intent_similarity >= self.intent_similarity_threshold
            )

            merge = False
            reason = "independent_needs"
            if different_dimension:
                reason = "different_question_dimensions"
            elif self.policy == "conservative":
                if (
                    option_connector
                    and same_intent
                    and intent_similarity >= self.intent_similarity_threshold
                ):
                    merge = True
                    reason = "conservative_explicit_alternative_same_intent"
            elif option_connector and (same_intent or len(original_dimensions) <= 1):
                merge = True
                reason = "same_need_alternatives"
            elif incomplete and len(original_dimensions) <= 1:
                merge = True
                reason = "dependent_fragment"
            elif shared_predicate and len(original_dimensions) <= 1:
                merge = True
                reason = "same_intent_shared_predicate"

            if merge:
                union(left_index, right_index)
            analyses.append(
                MergeSignals(
                    left_index=left_index,
                    right_index=right_index,
                    left_top_intent=left_intent,
                    right_top_intent=right_intent,
                    same_top_intent=same_intent,
                    intent_similarity=round(intent_similarity, 4),
                    sentence_similarity=round(sentence_similarity, 4),
                    option_connector=option_connector,
                    shared_predicate=shared_predicate,
                    incomplete_fragment=incomplete,
                    different_question_dimension=different_dimension,
                    merge=merge,
                    reason=reason,
                )
            )

        groups: dict[int, list[int]] = {}
        for index in range(len(needs)):
            groups.setdefault(find(index), []).append(index)

        final_needs: list[AtomicQuestion] = []
        for indexes in groups.values():
            if len(indexes) == 1:
                final_needs.append(needs[indexes[0]])
            else:
                # The original wording is the safest lossless representation.
                # Routing is a later stage and may create a shorter search query.
                final_needs.append(AtomicQuestion(question, question))
        final = DecompositionResult(len(final_needs) > 1, tuple(final_needs))
        return FusionResult(decomposition, final, tuple(analyses), len(final_needs) < len(needs))


def _load_decomposition(payload: dict[str, Any]) -> DecompositionResult:
    needs = tuple(AtomicQuestion(**item) for item in payload["needs"])
    return DecompositionResult(bool(payload["is_composite"]), needs)


def compare_report(
    baseline_report: dict[str, Any],
    fusion: SemanticQuestionFusion,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for record in baseline_report["records"]:
        result = fusion.fuse(record["question"], _load_decomposition(record["prediction"]))
        final_count = len(result.final.needs)
        count_correct = final_count == record["expected_need_count"]
        composite_correct = result.final.is_composite == record["expected_is_composite"]
        strict_correct = (
            count_correct
            and composite_correct
            and record["marker_coverage"] == 1.0
            and record["valid"]
        )
        records.append(
            {
                "case_id": record["case_id"],
                "language": record["language"],
                "question": record["question"],
                "expected_need_count": record["expected_need_count"],
                "baseline_need_count": len(record["prediction"]["needs"]),
                "final_need_count": final_count,
                "baseline_strict_correct": record["strict_correct"],
                "fusion_strict_correct": strict_correct,
                "fusion_applied": result.applied,
                "final_prediction": asdict(result.final),
                "pair_signals": [asdict(signal) for signal in result.pair_signals],
            }
        )

    def accuracy(items: list[dict[str, Any]], key: str) -> float:
        return mean(float(item[key]) for item in items)

    by_language = {}
    for language in sorted({record["language"] for record in records}):
        group = [record for record in records if record["language"] == language]
        by_language[language] = {
            "case_count": len(group),
            "baseline_strict_accuracy": accuracy(group, "baseline_strict_correct"),
            "fusion_strict_accuracy": accuracy(group, "fusion_strict_correct"),
        }
    summary = {
        "evaluation_status": "exploratory_same_observed_set",
        "case_count": len(records),
        "fusion_policy": fusion.policy,
        "baseline_strict_accuracy": accuracy(records, "baseline_strict_correct"),
        "fusion_strict_accuracy": accuracy(records, "fusion_strict_correct"),
        "fusion_applied_count": sum(bool(item["fusion_applied"]) for item in records),
        "improved_count": sum(
            not item["baseline_strict_correct"] and item["fusion_strict_correct"]
            for item in records
        ),
        "regressed_count": sum(
            item["baseline_strict_correct"] and not item["fusion_strict_correct"]
            for item in records
        ),
        "by_language": by_language,
    }
    return {"summary": summary, "records": records}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare la décomposition brute et sa fusion.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--policy", choices=("broad", "conservative"), default="broad")
    args = parser.parse_args()

    from qralia.rag.semantic_evaluation import Qwen3LangChainEmbeddings

    baseline = json.loads(args.input.read_text(encoding="utf-8"))
    embeddings = Qwen3LangChainEmbeddings(device=args.device, batch_size=16)
    report = compare_report(
        baseline,
        SemanticQuestionFusion(embeddings, policy=args.policy),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Rapport écrit dans {args.output}")


if __name__ == "__main__":
    main()
