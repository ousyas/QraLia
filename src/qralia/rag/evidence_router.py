"""Evidence-gated routing between validated invoice fields and official RAG.

Embeddings identify which *field definition* a question targets. They never
embed, infer, or alter the protected value itself.  Every semantic suggestion
must pass deterministic document or corpus gates before it becomes evidence.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from statistics import mean
from typing import Any, Protocol, Sequence

from qralia.rag.question_decomposition import ROOT, _search_form


DEFAULT_CASES = ROOT / "data/evaluation/evidence_routing.v0.1.jsonl"
DEFAULT_OUTPUT = ROOT / "outputs/evidence-routing-qwen3-v0.1.json"


class EmbeddingBackend(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


FIELD_CATALOG: dict[str, dict[str, Any]] = {
    "total_due": {
        "description": "montant total exact à payer sur cette facture; المبلغ الإجمالي الواجب أداؤه; شحال خاصني نخلص المجموع",
        "aliases": (
            "montant total", "montant à payer", "total à payer", "le total", "quel montant",
            "combien payer", "combien dois-je payer", "combien je dois payer",
            "المبلغ الإجمالي", "ما المبلغ", "شحال خاصني نخلص", "شحال المجموع",
        ),
    },
    "due_date": {
        "description": "date limite ou échéance imprimée pour payer; تاريخ الاستحقاق; آخر نهار للخلاص",
        "aliases": ("date d'échéance", "date limite", "تاريخ الاستحقاق", "آخر أجل", "آخر نهار", "فوقاش نخلص"),
    },
    "invoice_number": {
        "description": "numéro ou identifiant unique de la facture; رقم الفاتورة; نمرة الفاكتورة",
        "aliases": ("numéro de facture", "رقم الفاتورة", "نمرة الفاتورة", "نمرة الفاكتورة"),
    },
    "contract_number": {
        "description": "numéro du contrat ou abonnement client; رقم العقد; نمرة الكونطرا",
        "aliases": ("numéro de contrat", "رقم العقد", "نمرة الكونطرا", "رقم الكونطرا"),
    },
    "subtotal_excl_tax": {
        "description": "montant hors taxes HT avant les taxes; المبلغ دون احتساب الضرائب; الثمن بلا ضريبة",
        "aliases": ("hors taxes", "montant ht", "دون احتساب الضرائب", "بلا ضريبة"),
    },
    "tax_amount": {
        "description": "montant des taxes ou TVA imprimé sur la facture; مبلغ الضرائب; شحال الضريبة",
        "aliases": ("montant des taxes", "tva", "مبلغ الضرائب", "شحال الضريبة", "الضريبة"),
    },
    "cash_total": {
        "description": "total à régler en espèces incluant éventuellement le timbre; مبلغ الأداء نقدا; مجموع الكاش مع التمبر",
        "aliases": ("total en espèces", "montant en espèces", "الأداء نقدا", "مبلغ الكاش", "الكاش"),
    },
    "tariff_code": {
        "description": "tarif ou catégorie tarifaire indiqué sur la facture; التعريفة; الطاريف المكتوب",
        "aliases": ("tarif indiqué", "code tarif", "التعريفة", "الطاريف"),
    },
}

TRUSTED_SOURCE_ROLES = frozenset({"operator_primary", "legal_primary", "regulator"})


@dataclass(frozen=True)
class EvidencePolicy:
    invoice_threshold: float = 0.72
    rag_threshold: float = 0.70
    invoice_override_threshold: float = 0.86
    rag_override_threshold: float = 0.86
    field_margin: float = 0.04
    invoice_rank_floor: float = 0.25
    rank_margin_enabled: bool = True
    unknown_margin: float = 0.12
    min_ocr_confidence: float = 0.85
    rag_relative_ratio: float = 0.75
    max_rag_chunks: int = 3


@dataclass(frozen=True)
class InvoiceEvidence:
    field_id: str
    semantic_similarity: float
    alias_match: bool
    extraction_confidence: float
    confidence: float


@dataclass(frozen=True)
class RagEvidence:
    chunk_id: str
    reranker_score: float
    hybrid_confidence: float
    confidence: float


@dataclass(frozen=True)
class EvidenceDecision:
    route: str
    selected_invoice_fields: tuple[str, ...]
    selected_rag_chunks: tuple[str, ...]
    document_confidence: float
    retrieval_confidence: float
    decision_confidence: float
    reasons: tuple[str, ...]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return float(sum(a * b for a, b in zip(left, right)))


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    denominator = math.sqrt(_dot(left, left) * _dot(right, right))
    return _dot(left, right) / denominator if denominator else 0.0


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class InvoiceFieldMatcher:
    """Match a question to field definitions while keeping values untouched."""

    def __init__(self, embeddings: EmbeddingBackend) -> None:
        self.embeddings = embeddings
        self.field_ids = tuple(FIELD_CATALOG)
        self.prototype_vectors = embeddings.embed_documents(
            [FIELD_CATALOG[field_id]["description"] for field_id in self.field_ids]
        )
        # Threshold calibration replays each calibration question many times.
        # Cache only the semantic representation, never document values.
        self._query_vectors: dict[str, list[float]] = {}

    def match(
        self,
        question: str,
        fields: Sequence[dict[str, Any]],
        policy: EvidencePolicy,
    ) -> list[InvoiceEvidence]:
        if question not in self._query_vectors:
            self._query_vectors[question] = self.embeddings.embed_query(question)
        query_vector = self._query_vectors[question]
        normalized_question = _search_form(question)
        evidence: list[InvoiceEvidence] = []
        for field in fields:
            field_id = str(field.get("field_id", ""))
            if field_id not in FIELD_CATALOG:
                continue
            # Hard gates: semantic similarity can never revive an unsafe field.
            if field.get("status") != "validated":
                continue
            extraction_confidence = _clamp(field.get("extraction_confidence", 0.0))
            if extraction_confidence < policy.min_ocr_confidence:
                continue
            if not field.get("provenance_present", False):
                continue
            if field.get("validation_errors"):
                continue

            index = self.field_ids.index(field_id)
            similarity = _clamp(_cosine(query_vector, self.prototype_vectors[index]))
            alias_match = any(
                _search_form(alias) in normalized_question
                for alias in FIELD_CATALOG[field_id]["aliases"]
            )
            confidence = _clamp(
                0.78 * similarity
                + 0.17 * float(alias_match)
                + 0.05 * extraction_confidence
            )
            evidence.append(
                InvoiceEvidence(
                    field_id=field_id,
                    semantic_similarity=similarity,
                    alias_match=alias_match,
                    extraction_confidence=extraction_confidence,
                    confidence=confidence,
                )
            )
        return sorted(evidence, key=lambda item: (-item.confidence, item.field_id))


def _rag_candidate_allowed(candidate: dict[str, Any], context: dict[str, str]) -> bool:
    audiences = candidate.get("audiences", [])
    zone = candidate.get("service_zone_id")
    return bool(
        candidate.get("validity_status") == "active"
        and candidate.get("provider_id") == context.get("provider_id")
        and candidate.get("service_type") == context.get("service_type")
        and zone in {context.get("service_zone_id"), "national"}
        and context.get("audience") in audiences
        and candidate.get("source_role") in TRUSTED_SOURCE_ROLES
        and candidate.get("risk_level") in {"low", "medium"}
    )


def _select_rag_evidence(
    candidates: Sequence[dict[str, Any]],
    context: dict[str, str],
    policy: EvidencePolicy,
) -> list[RagEvidence]:
    eligible: list[RagEvidence] = []
    for candidate in candidates:
        if not _rag_candidate_allowed(candidate, context):
            continue
        reranker = _clamp(candidate.get("reranker_score", 0.0))
        hybrid = _clamp(candidate.get("hybrid_confidence", 0.0))
        confidence = _clamp(0.85 * reranker + 0.15 * hybrid)
        eligible.append(RagEvidence(str(candidate["chunk_id"]), reranker, hybrid, confidence))
    eligible.sort(key=lambda item: (-item.confidence, item.chunk_id))
    if not eligible or eligible[0].confidence < policy.rag_threshold:
        return []
    best = eligible[0].confidence
    return [
        item for item in eligible[: policy.max_rag_chunks]
        if item.confidence >= best * policy.rag_relative_ratio
    ]


class LiveRagEvidenceProvider:
    """Adapter from the real hybrid index and cross-encoder to router inputs."""

    def __init__(self, hybrid_index: Any, reranker: Any, *, rerank_k: int = 20, top_k: int = 5) -> None:
        self.hybrid_index = hybrid_index
        self.reranker = reranker
        self.rerank_k = rerank_k
        self.top_k = top_k

    def retrieve(self, question: str, context: dict[str, str]) -> list[dict[str, Any]]:
        from qralia.rag.reranker_evaluation import rerank_with_scores
        from qralia.rag.semantic_evaluation import searchable_text

        result = self.hybrid_index.search_components(
            question,
            domain=context.get("domain", "utilities"),
            provider_id=context["provider_id"],
            service_type=context["service_type"],
            service_zone_id=context["service_zone_id"],
            audience=context["audience"],
            top_k=self.rerank_k,
        )
        candidates = result["hybrid"]
        pairs = [
            (question, searchable_text(self.hybrid_index.by_id[item["chunk_id"]]))
            for item in candidates
        ]
        scores = self.reranker.score(pairs)
        ranked = rerank_with_scores(candidates, scores, top_k=self.top_k)
        enriched = []
        for item in ranked:
            metadata = self.hybrid_index.by_id[item["chunk_id"]]["metadata"]
            enriched.append({
                **item,
                "service_type": metadata["service_type"],
                "audiences": metadata["audiences"],
                "validity_status": metadata["validity_status"],
                "source_role": metadata["source_role"],
                "risk_level": metadata["risk_level"],
            })
        return enriched


class EvidenceRouter:
    """Make a conservative route decision from independently gated evidence."""

    def __init__(self, matcher: InvoiceFieldMatcher, policy: EvidencePolicy | None = None) -> None:
        self.matcher = matcher
        self.policy = policy or EvidencePolicy()

    def decide(
        self,
        *,
        question: str,
        qwen_hint: str,
        invoice_fields: Sequence[dict[str, Any]],
        rag_candidates: Sequence[dict[str, Any]],
        context: dict[str, str],
    ) -> EvidenceDecision:
        if qwen_hint not in {"invoice", "official_rag", "both", "unknown"}:
            raise ValueError(f"Indication Qwen invalide : {qwen_hint}")
        invoice = self.matcher.match(question, invoice_fields, self.policy)
        rag = _select_rag_evidence(rag_candidates, context, self.policy)

        invoice_best = invoice[0] if invoice else None
        document_confidence = invoice_best.confidence if invoice_best else 0.0
        retrieval_confidence = rag[0].confidence if rag else 0.0
        invoice_ambiguous = bool(
            len(invoice) > 1
            and not invoice[0].alias_match
            and invoice[0].confidence - invoice[1].confidence < self.policy.field_margin
        )
        rank_supported = bool(
            invoice_best
            and self.policy.rank_margin_enabled
            and qwen_hint in {"invoice", "both"}
            and document_confidence >= self.policy.invoice_rank_floor
            and not invoice_ambiguous
        )
        invoice_strong = bool(
            invoice_best
            and not invoice_ambiguous
            and (
                document_confidence >= self.policy.invoice_threshold
                or invoice_best.alias_match
                or rank_supported
            )
        )
        rag_strong = bool(rag)
        reasons: list[str] = []
        if invoice_ambiguous:
            reasons.append("ambiguous_invoice_field")
        if not invoice_strong:
            reasons.append("no_strong_invoice_evidence")
        if not rag_strong:
            reasons.append("no_strong_rag_evidence")

        route = "unknown"
        if invoice_strong and rag_strong:
            if qwen_hint == "both":
                route = "both"
                reasons.append("both_channels_confirmed_and_requested")
            elif qwen_hint == "invoice":
                route = "invoice"
                reasons.append("invoice_hint_confirmed")
            elif qwen_hint == "official_rag":
                route = "official_rag"
                reasons.append("rag_hint_confirmed")
            elif abs(document_confidence - retrieval_confidence) >= self.policy.unknown_margin:
                route = "invoice" if document_confidence > retrieval_confidence else "official_rag"
                reasons.append("strong_evidence_overrides_unknown_hint")
            else:
                reasons.append("conflicting_strong_evidence")
        elif invoice_strong:
            if qwen_hint == "invoice":
                route = "invoice"
                reasons.append("invoice_hint_confirmed")
            elif qwen_hint != "both" and (
                document_confidence >= self.policy.invoice_override_threshold
                or invoice_best.alias_match
            ):
                route = "invoice"
                reasons.append(
                    "validated_invoice_alias_overrides_hint"
                    if invoice_best.alias_match
                    else "invoice_evidence_overrides_hint"
                )
            else:
                reasons.append("missing_required_rag_evidence" if qwen_hint == "both" else "invoice_not_strong_enough_to_override")
        elif rag_strong:
            if qwen_hint == "official_rag":
                route = "official_rag"
                reasons.append("rag_hint_confirmed")
            elif qwen_hint != "both" and retrieval_confidence >= self.policy.rag_override_threshold:
                route = "official_rag"
                reasons.append("rag_evidence_overrides_hint")
            else:
                reasons.append("missing_required_invoice_evidence" if qwen_hint == "both" else "rag_not_strong_enough_to_override")
        else:
            reasons.append("safe_abstention")

        selected_fields = (invoice_best.field_id,) if route in {"invoice", "both"} and invoice_best else ()
        selected_chunks = tuple(item.chunk_id for item in rag) if route in {"official_rag", "both"} else ()
        if route == "invoice":
            decision_confidence = document_confidence
        elif route == "official_rag":
            decision_confidence = retrieval_confidence
        elif route == "both":
            decision_confidence = min(document_confidence, retrieval_confidence)
        else:
            decision_confidence = 1.0 - max(document_confidence, retrieval_confidence)
        return EvidenceDecision(
            route=route,
            selected_invoice_fields=selected_fields,
            selected_rag_chunks=selected_chunks,
            document_confidence=document_confidence,
            retrieval_confidence=retrieval_confidence,
            decision_confidence=_clamp(decision_confidence),
            reasons=tuple(reasons),
        )

    @staticmethod
    def aggregate(decisions: Sequence[EvidenceDecision]) -> EvidenceDecision:
        """Combine independently routed atomic needs without losing evidence."""
        if not decisions:
            raise ValueError("Il faut au moins une décision atomique.")
        routes = {item.route for item in decisions}
        if "unknown" in routes:
            route = "unknown"
        elif "both" in routes or {"invoice", "official_rag"}.issubset(routes):
            route = "both"
        else:
            route = next(iter(routes))
        fields = tuple(dict.fromkeys(
            field for item in decisions for field in item.selected_invoice_fields
        ))
        chunks = tuple(dict.fromkeys(
            chunk for item in decisions for chunk in item.selected_rag_chunks
        ))
        document_scores = [item.document_confidence for item in decisions if item.selected_invoice_fields]
        retrieval_scores = [item.retrieval_confidence for item in decisions if item.selected_rag_chunks]
        confidence = min(item.decision_confidence for item in decisions)
        return EvidenceDecision(
            route=route,
            selected_invoice_fields=fields if route != "unknown" else (),
            selected_rag_chunks=chunks if route != "unknown" else (),
            document_confidence=min(document_scores) if document_scores else 0.0,
            retrieval_confidence=min(retrieval_scores) if retrieval_scores else 0.0,
            decision_confidence=confidence if route != "unknown" else 1.0 - max(confidence, 0.0),
            reasons=tuple(reason for item in decisions for reason in item.reasons) + ("atomic_decisions_aggregated",),
        )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    routes = ("invoice", "official_rag", "both", "unknown")
    per_route = {}
    recalls = []
    for route in routes:
        positives = [item for item in records if item["expected_route"] == route]
        recall = mean(item["route_correct"] for item in positives) if positives else 0.0
        recalls.append(recall)
        per_route[route] = {"count": len(positives), "recall": recall}
    answerable = [item for item in records if item["expected_route"] != "unknown"]
    abstentions = [item for item in records if item["expected_route"] == "unknown"]
    answered = [item for item in records if item["decision"]["route"] != "unknown"]
    return {
        "case_count": len(records),
        "route_accuracy": mean(item["route_correct"] for item in records),
        "macro_route_recall": mean(recalls),
        "invoice_field_accuracy": mean(item["invoice_field_correct"] for item in records),
        "rag_chunk_recall": mean(item["rag_chunk_recall"] for item in records),
        "answerable_route_recall": mean(item["route_correct"] for item in answerable) if answerable else 0.0,
        "correct_abstention_rate": mean(item["route_correct"] for item in abstentions) if abstentions else 0.0,
        "answered_route_precision": mean(item["route_correct"] for item in answered) if answered else 1.0,
        "false_abstention_rate": mean(item["decision"]["route"] == "unknown" for item in answerable) if answerable else 0.0,
        "unsafe_route_rate": mean(
            item["decision"]["route"] != "unknown" and not item["route_correct"]
            for item in records
        ),
        "per_route": per_route,
    }


def evaluate(router: EvidenceRouter, cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    records = []
    for case in cases:
        if case.get("needs"):
            atomic = [
                router.decide(
                    question=need["question"], qwen_hint=need["qwen_hint"],
                    invoice_fields=case["invoice_fields"],
                    rag_candidates=need.get("rag_candidates", []), context=case["context"],
                )
                for need in case["needs"]
            ]
            decision = router.aggregate(atomic)
        else:
            decision = router.decide(
                question=case["question"], qwen_hint=case["qwen_hint"],
                invoice_fields=case["invoice_fields"],
                rag_candidates=case["rag_candidates"], context=case["context"],
            )
        expected_field = case.get("expected_field_id")
        expected_chunks = set(case.get("expected_chunk_ids", []))
        selected_chunks = set(decision.selected_rag_chunks)
        records.append({
            "case_id": case["case_id"], "split": case["split"], "language": case["language"],
            "question": case["question"], "qwen_hint": case.get("qwen_hint", "decomposed"),
            "expected_route": case["expected_route"], "expected_field_id": expected_field,
            "expected_chunk_ids": sorted(expected_chunks), "decision": asdict(decision),
            "route_correct": decision.route == case["expected_route"],
            "invoice_field_correct": (not expected_field and not decision.selected_invoice_fields) or decision.selected_invoice_fields == (expected_field,),
            "rag_chunk_recall": (len(expected_chunks & selected_chunks) / len(expected_chunks)) if expected_chunks else float(not selected_chunks),
        })
    return {"metrics": _metrics(records), "records": records}


def calibrate_policy(matcher: InvoiceFieldMatcher, cases: Sequence[dict[str, Any]]) -> tuple[EvidencePolicy, dict[str, Any]]:
    candidates = []
    for invoice_threshold, rag_threshold, override in product(
        (0.62, 0.68, 0.74, 0.80), (0.58, 0.66, 0.74, 0.82), (0.80, 0.86, 0.92)
    ):
        policy = EvidencePolicy(
            invoice_threshold=invoice_threshold,
            rag_threshold=rag_threshold,
            invoice_override_threshold=max(invoice_threshold, override),
            rag_override_threshold=max(rag_threshold, override),
        )
        report = evaluate(EvidenceRouter(matcher, policy), cases)
        metrics = report["metrics"]
        candidates.append((policy, metrics))
    # Optimize balanced route quality first; ties favor stricter safe thresholds.
    return max(
        candidates,
        key=lambda item: (
            item[1]["macro_route_recall"], item[1]["route_accuracy"],
            item[0].invoice_threshold + item[0].rag_threshold,
            item[0].invoice_override_threshold + item[0].rag_override_threshold,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibre et évalue le routeur de preuves QraLia.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cuda")
    args = parser.parse_args()
    from qralia.rag.semantic_evaluation import Qwen3LangChainEmbeddings

    cases = _read_jsonl(args.cases)
    calibration = [case for case in cases if case["split"] == "calibration"]
    test = [case for case in cases if case["split"] == "test"]
    challenge = [case for case in cases if case["split"] == "challenge"]
    holdout_v2 = [case for case in cases if case["split"] == "holdout_v2"]
    embeddings = Qwen3LangChainEmbeddings(device=args.device, batch_size=16, max_seq_length=256)
    matcher = InvoiceFieldMatcher(embeddings)
    policy, calibration_metrics = calibrate_policy(matcher, calibration)
    test_report = evaluate(EvidenceRouter(matcher, policy), test)
    challenge_report = evaluate(EvidenceRouter(matcher, policy), challenge) if challenge else None
    holdout_v2_report = evaluate(EvidenceRouter(matcher, policy), holdout_v2) if holdout_v2 else None
    report = {
        "schema_version": "0.1.0",
        "evaluation_status": "calibration_and_held_out_test",
        "embedding_model": embeddings.model_name,
        "policy": asdict(policy),
        "calibration": calibration_metrics,
        "test": test_report,
        "challenge": challenge_report,
        "holdout_v2": holdout_v2_report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"policy": asdict(policy), "calibration": calibration_metrics, "test": test_report["metrics"], "challenge": challenge_report["metrics"] if challenge_report else None, "holdout_v2": holdout_v2_report["metrics"] if holdout_v2_report else None}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
