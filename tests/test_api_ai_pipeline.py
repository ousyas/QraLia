"""Garde-fous unitaires du chemin RAG/LLM, sans charger les modèles."""

from datetime import datetime, timedelta, timezone

from qralia.api.ai_pipeline import (
    LocalEvidenceRuntime,
    _explicit_invoice_decision,
    _fallback_need,
    _generation_errors,
    _reconcile_source_hint,
    detect_language,
)
from qralia.rag.evidence_router import EvidenceDecision
from qralia.api.store import InvoiceRecord
from qralia.domain.extracted_invoice import ExtractedInvoice
from qralia.domain.invoice_evidence import InvoiceEvidence
from qralia.rag.generation_evaluation import AnswerClaim, GroundedAnswer, render_answer_from_claims


def _invoice_evidence() -> InvoiceEvidence:
    return InvoiceEvidence(
        evidence_id="invoice.total_check",
        invoice_id="synthetic-1",
        field="total_check",
        value=438.70,
        display_value="438,70",
        text="Montant total à régler : 438,70",
        confidence=0.98,
        page=1,
    )


def _answer(text: str, citation: str = "invoice.total_check", language: str = "fr") -> GroundedAnswer:
    return render_answer_from_claims(GroundedAnswer(
        answer="",
        claims=[AnswerClaim(text=text, citations=[citation])],
        abstain=False,
        abstention_reason=None,
        language=language,
    ))


def test_generation_guard_accepts_exact_protected_value():
    evidence = _invoice_evidence()
    answer = _answer("Le montant à payer est 438,70.")
    assert _generation_errors(answer, [evidence], []) == []


def test_generation_guard_rejects_modified_value_and_unknown_citation():
    evidence = _invoice_evidence()
    answer = _answer("Le montant à payer est 439,70.", citation="invoice.unknown")
    assert _generation_errors(answer, [evidence], []) == [
        "invalid_citation",
        "invented_or_modified_number",
    ]


def test_generation_guard_rejects_wrong_script():
    evidence = _invoice_evidence()
    answer = _answer("Le montant est 438,70.", language="ar")
    assert "wrong_output_script" in _generation_errors(answer, [evidence], [])


def test_language_detection_is_conservative():
    assert detect_language("Quel est le montant ?", "auto") == "fr"
    assert detect_language("شحال خاصني نخلص؟", "auto") == "ar"
    assert detect_language("chhal khasni nkhlles", "auto") == "darija_latn"
    assert detect_language("أي سؤال", "darija_arab") == "darija_arab"


def test_explicit_invoice_alias_uses_validated_field_without_embeddings():
    evidence = _invoice_evidence()
    now = datetime.now(timezone.utc)
    record = InvoiceRecord(
        public_id="public-1",
        invoice=ExtractedInvoice.model_construct(),
        validation={"valid": True},
        evidence=(evidence,),
        created_at=now,
        expires_at=now + timedelta(minutes=30),
    )
    decision = _explicit_invoice_decision(
        "Combien dois-je payer ?",
        "invoice",
        record,
    )
    assert decision is not None
    assert decision.route == "invoice"
    assert decision.selected_invoice_fields == ("total_due",)


def test_reconcile_false_both_for_explicit_official_procedure():
    question = "À quel ordre dois-je écrire le chèque pour payer ma facture ONEE ?"
    assert _reconcile_source_hint(question, "both") == "official_rag"


def test_reconcile_keeps_real_mixed_need():
    question = "Combien dois-je payer et où puis-je payer ?"
    assert _reconcile_source_hint(question, "both") == "both"


def test_reconcile_does_not_override_non_both_qwen_hint():
    question = "À quel ordre dois-je écrire le chèque ?"
    assert _reconcile_source_hint(question, "official_rag") == "official_rag"


def test_rag_retrieval_runs_even_when_qwen_hint_is_invoice():
    class Provider:
        def __init__(self):
            self.called = False

        def retrieve(self, question, context):
            self.called = True
            return [{"chunk_id": "official-proof"}]

    class Router:
        def decide(self, **kwargs):
            assert kwargs["rag_candidates"] == [{"chunk_id": "official-proof"}]
            return EvidenceDecision(
                route="official_rag",
                selected_invoice_fields=(),
                selected_rag_chunks=("official-proof",),
                document_confidence=0.0,
                retrieval_confidence=0.95,
                decision_confidence=0.95,
                reasons=("strong_evidence_overrides_hint",),
            )

    provider = Provider()
    runtime = LocalEvidenceRuntime([])
    runtime._matcher = object()
    runtime._provider = provider
    runtime._router = Router()
    record = type("Record", (), {"validation": {}, "evidence": ()})()

    decision = runtime.decide(
        question="question procédurale",
        qwen_hint="invoice",
        record=record,
        allow_rag=True,
    )

    assert provider.called is True
    assert decision.route == "official_rag"


def test_mixed_amount_and_payment_location_cannot_take_invoice_fast_path():
    question = "Combien dois-je payer, et où puis-je régler cette somme en espèces ?"
    need = _fallback_need(question)

    assert need.source_hint == "both"


def test_cash_amount_alone_remains_an_invoice_question():
    need = _fallback_need("Combien dois-je payer en espèces ?")

    assert need.source_hint == "invoice"
