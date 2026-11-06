from qralia.rag.evidence_router import (
    EvidencePolicy,
    EvidenceRouter,
    InvoiceEvidence,
    InvoiceFieldMatcher,
)


CONTEXT = {
    "provider_id": "onee",
    "service_type": "electricity",
    "service_zone_id": "onee_be_national",
    "audience": "residential",
}


class FixedMatcher:
    def __init__(self, evidence):
        self.evidence = evidence

    def match(self, _question, _fields, _policy):
        return list(self.evidence)


def invoice(confidence=0.95):
    return InvoiceEvidence("total_due", 0.95, True, 0.98, confidence)


def rag_candidate(**overrides):
    candidate = {
        "chunk_id": "onee-payment",
        "reranker_score": 0.95,
        "hybrid_confidence": 0.90,
        "provider_id": "onee",
        "service_type": "electricity",
        "service_zone_id": "onee_be_national",
        "audiences": ["residential"],
        "validity_status": "active",
        "source_role": "operator_primary",
        "risk_level": "medium",
    }
    candidate.update(overrides)
    return candidate


def decide(matcher, hint, rag=()):
    return EvidenceRouter(matcher).decide(
        question="question",
        qwen_hint=hint,
        invoice_fields=[],
        rag_candidates=rag,
        context=CONTEXT,
    )


def test_both_requires_two_independently_validated_channels() -> None:
    result = decide(FixedMatcher([invoice()]), "both", [rag_candidate()])
    assert result.route == "both"
    assert result.selected_invoice_fields == ("total_due",)
    assert result.selected_rag_chunks == ("onee-payment",)


def test_missing_rag_for_both_causes_abstention() -> None:
    result = decide(FixedMatcher([invoice()]), "both")
    assert result.route == "unknown"
    assert "missing_required_rag_evidence" in result.reasons


def test_strong_invoice_evidence_can_correct_unknown_hint() -> None:
    result = decide(FixedMatcher([invoice(0.94)]), "unknown")
    assert result.route == "invoice"
    assert "validated_invoice_alias_overrides_hint" in result.reasons


def test_wrong_provider_and_stale_chunks_are_hard_filtered() -> None:
    result = decide(
        FixedMatcher([]),
        "official_rag",
        [
            rag_candidate(provider_id="srm_rsk", reranker_score=0.99),
            rag_candidate(chunk_id="stale", validity_status="stale", reranker_score=0.99),
        ],
    )
    assert result.route == "unknown"
    assert result.selected_rag_chunks == ()


def test_press_context_is_not_admissible_evidence() -> None:
    result = decide(
        FixedMatcher([]),
        "official_rag",
        [rag_candidate(source_role="press_context", risk_level="high")],
    )
    assert result.route == "unknown"


class TinyEmbeddings:
    def embed_documents(self, texts):
        return [[1.0, 0.0] if index == 0 else [0.0, 1.0] for index, _ in enumerate(texts)]

    def embed_query(self, _text):
        return [1.0, 0.0]


def test_invalid_ocr_field_is_never_revived_by_perfect_similarity() -> None:
    matcher = InvoiceFieldMatcher(TinyEmbeddings())
    fields = [{
        "field_id": "total_due",
        "status": "candidate",
        "extraction_confidence": 1.0,
        "provenance_present": True,
        "validation_errors": [],
    }]
    assert matcher.match("montant total", fields, EvidencePolicy()) == []


def test_missing_provenance_blocks_a_validated_field() -> None:
    matcher = InvoiceFieldMatcher(TinyEmbeddings())
    fields = [{
        "field_id": "total_due",
        "status": "validated",
        "extraction_confidence": 1.0,
        "provenance_present": False,
        "validation_errors": [],
    }]
    assert matcher.match("montant total", fields, EvidencePolicy()) == []


def test_common_total_question_disambiguates_total_due_from_cash_total() -> None:
    matcher = InvoiceFieldMatcher(TinyEmbeddings())
    fields = [
        {
            "field_id": field_id,
            "status": "validated",
            "extraction_confidence": 0.98,
            "provenance_present": True,
            "validation_errors": [],
        }
        for field_id in ("total_due", "cash_total")
    ]
    matches = matcher.match("Combien dois-je payer ?", fields, EvidencePolicy())
    assert matches[0].field_id == "total_due"
    assert matches[0].alias_match is True


def test_rank_margin_can_accept_a_paraphrase_when_hint_and_ocr_agree() -> None:
    evidence = [
        InvoiceEvidence("total_due", 0.50, False, 0.98, 0.44),
        InvoiceEvidence("tax_amount", 0.30, False, 0.98, 0.28),
    ]
    result = decide(FixedMatcher(evidence), "invoice")
    assert result.route == "invoice"


def test_rank_margin_never_overrides_unknown_hint() -> None:
    evidence = [InvoiceEvidence("total_due", 0.50, False, 0.98, 0.44)]
    result = decide(FixedMatcher(evidence), "unknown")
    assert result.route == "unknown"


def test_atomic_invoice_and_rag_decisions_aggregate_to_both() -> None:
    invoice_result = decide(FixedMatcher([invoice()]), "invoice")
    rag_result = decide(FixedMatcher([]), "official_rag", [rag_candidate()])
    result = EvidenceRouter.aggregate([invoice_result, rag_result])
    assert result.route == "both"
    assert result.selected_invoice_fields == ("total_due",)
    assert result.selected_rag_chunks == ("onee-payment",)
